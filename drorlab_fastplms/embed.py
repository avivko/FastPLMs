#!/usr/bin/env python3
"""Embed protein sequences from CSV or FASTA using a FastPLMs checkpoint (run inside Docker/Singularity image)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python path/to/drorlab_fastplms/embed.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
from drorlab_fastplms.e1_context import build_e1_row_strings
from drorlab_fastplms.io_utils import load_csv_as_records, load_sequences_csv, load_sequences_fasta


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
    p.add_argument("--output", required=True, help="Output .pth or .db (SQLite)")
    p.add_argument("--pooling", default="mean", help="Comma-separated pooling strategies (default: mean)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--full-embeddings", action="store_true", help="Per-residue embeddings in .pth")
    p.add_argument(
        "--attn-backend",
        default="auto",
        help="Attention backend: auto (fastest available: flash→flex→sdpa), sdpa (exact/reproducible), flex, kernels_flash (default: auto)",
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
    args = p.parse_args(argv)

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
            records = load_csv_as_records(inp)
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

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    use_sql = out_path.lower().endswith(".db")

    dtype = resolve_torch_dtype(args.dtype)
    device = default_device()
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

    tokenizer = None if e1 else model.tokenizer
    pooling = parse_pooling(args.pooling)

    kwargs = dict(
        sequences=sequences,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        truncate=True,
        full_embeddings=args.full_embeddings,
        embed_dtype=torch.float32,
        pooling_types=pooling,
        save=not use_sql,
        save_path=out_path if not use_sql else "embeddings.pth",
        sql=use_sql,
        sql_db_path=out_path if use_sql else "embeddings.db",
        padding="longest",
    )

    with torch.inference_mode():
        model.embed_dataset(**kwargs)

    if args.id_col and ids is not None and len(ids) == len(sequences) and not use_sql:
        manifest = os.path.splitext(out_path)[0] + "_manifest.csv"
        pd.DataFrame({"id": ids, "sequence": sequences}).to_csv(manifest, index=False)
        print(f"Wrote manifest {manifest}")

    if use_sql:
        print(f"Wrote SQLite embeddings to {out_path}")
    else:
        print(f"Wrote embeddings to {out_path} ({len(sequences)} sequences; keys are sequence strings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
