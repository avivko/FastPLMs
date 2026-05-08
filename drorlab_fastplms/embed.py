#!/usr/bin/env python3
"""Embed protein sequences from CSV or FASTA using a FastPLMs checkpoint (run inside Docker/Singularity image)."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

# Allow `python path/to/drorlab_fastplms/embed.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR in sys.path:
    sys.path.remove(_REPO_ROOT_STR)
sys.path.insert(0, _REPO_ROOT_STR)

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM

from drorlab_fastplms.cli_common import (
    apply_attn_backend_after_load,
    configure_hf_token,
    default_device,
    is_e1_config,
    model_config_with_attn,
    resolve_torch_dtype,
    try_entrypoint_setup,
)
from drorlab_fastplms.e1_context import (
    build_e1_row_strings,
    e1_multiseq_has_context,
    normalize_e1_multiseq_string,
    prepare_e1_inputs_for_runtime,
    reduce_e1_multiseq_context_to_budget,
    validate_e1_embed_inputs,
)
from drorlab_fastplms.embed_batch_timing import patch_embed_timing
from drorlab_fastplms.io_utils import load_csv_as_records, load_sequences_csv, load_sequences_fasta
from drorlab_fastplms.zarr_export import (
    choose_resume_db_path,
    convert_db_to_zarr,
    default_db_path_for_zarr,
    export_embeddings_to_zarr,
)


def parse_pooling(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Embed sequences with FastPLMs (MaskedLM + EmbeddingMixin).")
    p.add_argument("--model", required=True, help="HuggingFace model id, e.g. Synthyra/ESM2-8M")
    p.add_argument("--input", required=True, help="Path to .csv or .fasta/.fa")
    p.add_argument(
        "--seq-col",
        default=None,
        help="CSV: column for query sequence (required for most CSV inputs; omit for FASTA)",
    )
    p.add_argument("--id-col", default=None, help="Optional CSV column for row ids (writes manifest next to output)")
    p.add_argument("--output", required=True, help="Output .pth, .db (SQLite), or .zarr")
    p.add_argument("--pooling", default="mean", help="Comma-separated pooling strategies (default: mean)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--full-embeddings", action="store_true", help="Per-residue embeddings in .pth")
    p.add_argument(
        "--attn-backend",
        default="kernels_flash",
        help="Attention backend: auto (fastest available: flash→flex→sdpa), sdpa (exact/reproducible), flex, kernels_flash (default: kernels_flash)",
    )
    p.add_argument("--dtype", default="float16", help="Model load dtype: bfloat16, float16, float32")
    p.add_argument("--no-entrypoint-setup", action="store_true", help="Skip entrypoint_setup import")
    p.add_argument(
        "--e1-combined-col",
        default=None,
        help="E1 only: CSV column already containing comma-separated multi-sequence string",
    )
    p.add_argument(
        "--e1-context-cols",
        nargs="*",
        default=None,
        help="E1 only: context columns in order, before --seq-col as query (space-separated)",
    )
    p.add_argument(
        "--e1-reduced-max-token-length",
        type=int,
        default=None,
        help="E1 multiseq only: approximate reduced token budget; randomly drop context segments until it fits",
    )
    p.add_argument(
        "--e1-context-drop-seed",
        type=int,
        default=0,
        help="Seed for random E1 context dropping when --e1-reduced-max-token-length is set",
    )
    p.add_argument(
        "--timing",
        action="store_true",
        help=(
            "Print timing for CSV load, E1 preprocessing, model load, total embed wall time, and "
            "per-batch forward_s + since_prev_forward_s (CPU/IO between batches). "
            "Uses a drorlab-only wrap of model._embed (works with padding=longest / no torch.compile)."
        ),
    )
    p.add_argument(
        "--timing-batch-log-every",
        type=int,
        default=50,
        help="With --timing, log a line every N batches (0 = only final summary).",
    )
    p.add_argument(
        "--zarr-resume-from-db",
        nargs="?",
        const="__AUTO__",
        default=None,
        help=(
            "When output is .zarr, pre-seed/resume by converting from a SQLite .db first. "
            "Optional value: explicit DB path. If provided without value, default is <output_stem>.db. "
            "If a newer staged DB exists on scratch, it is preferred automatically."
        ),
    )
    args = p.parse_args(argv)
    stage_t0 = time.perf_counter()

    if not args.no_entrypoint_setup:
        try_entrypoint_setup()
    configure_hf_token()

    inp = args.input
    ext = os.path.splitext(inp)[1].lower()
    cfg = model_config_with_attn(args.model, args.attn_backend)
    e1 = is_e1_config(cfg)

    ids: list[str] | None = None
    sequences: list[str]

    if ext in (".fasta", ".fa", ".faa", ".fna"):
        if args.e1_combined_col or (args.e1_context_cols and len(args.e1_context_cols) > 0):
            print("Warning: E1 context flags ignored for FASTA input.", file=sys.stderr)
        sequences, _ = load_sequences_fasta(inp)
    elif ext == ".csv":
        csv_allows_no_seq_col = bool(e1 and args.e1_combined_col)
        if not csv_allows_no_seq_col and args.seq_col is None:
            print("--seq-col is required for this CSV input", file=sys.stderr)
            return 2
        if e1 and (args.e1_combined_col or (args.e1_context_cols is not None and len(args.e1_context_cols) > 0)):
            selected_cols = []
            if args.e1_combined_col:
                selected_cols.append(args.e1_combined_col)
            selected_cols.extend(args.e1_context_cols or [])
            if args.seq_col:
                selected_cols.append(args.seq_col)
            if args.id_col:
                selected_cols.append(args.id_col)
            selected_cols = list(dict.fromkeys(selected_cols))
            records = load_csv_as_records(inp, usecols=selected_cols)
            sequences = []
            row_ids: list[str] = []
            for i, row in enumerate(records):
                sequences.append(
                    build_e1_row_strings(
                        combined_col=args.e1_combined_col,
                        context_cols=args.e1_context_cols or [],
                        query_col=args.seq_col or "",
                        row=row,
                    )
                )
                if args.id_col:
                    rid = row.get(args.id_col)
                    row_ids.append("" if rid is None or (isinstance(rid, float) and pd.isna(rid)) else str(rid))
                else:
                    row_ids.append(str(i))
            ids = row_ids if args.id_col else None
        else:
            sequences, ids = load_sequences_csv(inp, args.seq_col, args.id_col)
    else:
        print(f"Unsupported input extension {ext!r}; use .csv or .fasta", file=sys.stderr)
        return 2

    use_e1_context_mode = bool(args.e1_combined_col or (args.e1_context_cols and len(args.e1_context_cols) > 0))
    if args.timing:
        print(f"[timing] load_input_s={time.perf_counter() - stage_t0:.3f}")

    if e1:
        e1_prep_t0 = time.perf_counter()
        sequences = [normalize_e1_multiseq_string(s) for s in sequences]
        if use_e1_context_mode:
            no_context_count = sum(1 for s in sequences if not e1_multiseq_has_context(s))
            print(
                f"E1 context mode: {no_context_count}/{len(sequences)} rows have no context (query-only).",
                file=sys.stderr,
            )
        if args.e1_reduced_max_token_length is not None:
            rng = random.Random(args.e1_context_drop_seed)
            noop_budget_count = 0
            no_context_count_after_reduce = 0
            try:
                reduced_sequences: list[str] = []
                for s in sequences:
                    reduced, was_noop, was_no_context = reduce_e1_multiseq_context_to_budget(
                        s,
                        reduced_max_token_length=args.e1_reduced_max_token_length,
                        rng=rng,
                    )
                    if was_noop and not was_no_context:
                        noop_budget_count += 1
                    if was_no_context:
                        no_context_count_after_reduce += 1
                    reduced_sequences.append(reduced)
                sequences = reduced_sequences
            except ValueError as e:
                print(str(e), file=sys.stderr)
                return 2
            if noop_budget_count > 0:
                print(
                    "Warning: --e1-reduced-max-token-length is >= estimated full multiseq token length "
                    f"for {noop_budget_count}/{len(sequences)} rows; context dropping was a no-op for those rows.",
                    file=sys.stderr,
                )
            if no_context_count_after_reduce > 0:
                print(
                    "Note: --e1-reduced-max-token-length skipped reduction for "
                    f"{no_context_count_after_reduce}/{len(sequences)} query-only rows (no context present).",
                    file=sys.stderr,
                )
        sequences = prepare_e1_inputs_for_runtime(sequences, truncate=True, max_len=args.max_len)
        try:
            validate_e1_embed_inputs(sequences, row_labels=ids)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 2
        if args.timing:
            print(f"[timing] e1_preprocess_s={time.perf_counter() - e1_prep_t0:.3f}")

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    use_sql = out_path.lower().endswith(".db")
    use_zarr = out_path.lower().endswith(".zarr")

    dtype = resolve_torch_dtype(args.dtype)
    device = default_device()
    load_model_t0 = time.perf_counter()
    print(f"Loading {args.model} on {device} dtype={dtype} attn_backend={args.attn_backend} ...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        config=cfg,
        trust_remote_code=True,
        dtype=dtype,
    )
    apply_attn_backend_after_load(model, args.attn_backend, cfg)
    model.eval()
    model.to(device)
    if args.timing:
        print(f"[timing] model_load_s={time.perf_counter() - load_model_t0:.3f}")

    tokenizer = None if e1 else model.tokenizer
    pooling = parse_pooling(args.pooling)

    kwargs = dict(
        sequences=sequences,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        truncate=False if e1 else True,
        full_embeddings=args.full_embeddings,
        embed_dtype=torch.float32,
        pooling_types=pooling,
        save=not use_sql and not use_zarr,
        save_path=out_path if (not use_sql and not use_zarr) else "embeddings.pth",
        sql=use_sql,
        sql_db_path=out_path if use_sql else "embeddings.db",
        padding="longest",
    )

    sequence_to_id: dict[str, str] | None = None
    if ids is not None and len(ids) == len(sequences):
        sequence_to_id = {}
        for seq, sid in zip(sequences, ids):
            if seq not in sequence_to_id:
                sequence_to_id[seq] = sid

    restore_timing: Callable[[], None] | None = None
    if args.timing:
        restore_timing = patch_embed_timing(
            model,
            cuda_sync=True,
            log_every=max(0, args.timing_batch_log_every),
        )

    embed_t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            if use_zarr:
                if args.zarr_resume_from_db is not None:
                    db_hint = (
                        default_db_path_for_zarr(out_path)
                        if args.zarr_resume_from_db == "__AUTO__"
                        else args.zarr_resume_from_db
                    )
                    db_source = choose_resume_db_path(db_hint, prefer_staged=True)
                    if db_source is None:
                        print(
                            f"[zarr-resume] No resume DB found at {db_hint} (or staged equivalent). Skipping DB pre-seed."
                        )
                    else:
                        print(f"[zarr-resume] Pre-seeding from DB: {db_source}")
                        convert_db_to_zarr(
                            db_path=db_source,
                            zarr_path=out_path,
                            manifest_path=os.path.splitext(out_path.rstrip("/"))[0] + "_manifest.csv",
                            batch_size=2048,
                            timing=args.timing,
                        )
                manifest = os.path.splitext(out_path.rstrip("/"))[0] + "_manifest.csv"
                export_embeddings_to_zarr(
                    model=model,
                    sequences=sequences,
                    sequence_to_id=sequence_to_id,
                    tokenizer=tokenizer,
                    save_path=out_path,
                    manifest_path=manifest,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    truncate=False if e1 else True,
                    full_embeddings=args.full_embeddings,
                    embed_dtype=torch.float32,
                    pooling_types=pooling,
                    timing=args.timing,
                )
            else:
                model.embed_dataset(**kwargs)
    finally:
        if restore_timing is not None:
            restore_timing()
    if args.timing:
        print(f"[timing] embed_total_s={time.perf_counter() - embed_t0:.3f}")

    if args.id_col and ids is not None and len(ids) == len(sequences) and not use_sql and not use_zarr:
        manifest = os.path.splitext(out_path)[0] + "_manifest.csv"
        pd.DataFrame({"id": ids, "sequence": sequences}).to_csv(manifest, index=False)
        print(f"Wrote manifest {manifest}")

    if use_sql:
        print(f"Wrote SQLite embeddings to {out_path}")
    elif use_zarr:
        print(f"Wrote Zarr embeddings to {out_path}")
    else:
        print(f"Wrote embeddings to {out_path} ({len(sequences)} sequences; keys are sequence strings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
