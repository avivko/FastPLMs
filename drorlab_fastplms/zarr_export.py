"""Zarr export utilities for drorlab embedding CLI (no fastplms core changes)."""

from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

from fastplms.embedding_mixin import Pooler, build_collator
from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

try:
    import zarr
except Exception:  # pragma: no cover - runtime dependency check
    zarr = None


class ProteinDataset(TorchDataset):
    def __init__(self, sequences: List[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def _ensure_zarr() -> None:
    if zarr is None:
        raise RuntimeError(
            "Zarr output requested, but `zarr` is not installed in this environment. "
            "Install zarr in the runtime image/environment and retry."
        )


def _load_manifest_sequences(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    seen: set[str] = set()
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row.get("sequence")
            if seq:
                seen.add(seq)
    return seen


def _open_manifest(path: str, full_embeddings: bool) -> Tuple[Any, Any]:
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if not exists:
        if full_embeddings:
            writer.writerow(["row_index", "sequence", "id", "residue_start", "residue_length"])
        else:
            writer.writerow(["row_index", "sequence", "id"])
    return f, writer


def _ensure_array(group: Any, name: str, shape: Tuple[int, ...], chunks: Tuple[int, ...], dtype: str):
    if name in group:
        return group[name]
    # zarr v3 uses create_array (create_dataset may not exist).
    return group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)


def _maybe_get_array(group: Any, name: str):
    return group[name] if name in group else None


def default_manifest_path_for_zarr(zarr_path: str) -> str:
    return os.path.splitext(zarr_path.rstrip("/"))[0] + "_manifest.csv"


def default_db_path_for_zarr(zarr_path: str) -> str:
    return os.path.splitext(zarr_path.rstrip("/"))[0] + ".db"


def infer_staged_db_candidate(db_path: str) -> Optional[str]:
    """Infer stable staged DB path used by sbatch_embed.sbatch."""
    key = hashlib.sha1(db_path.encode("utf-8")).hexdigest()
    base = os.path.basename(db_path)
    rel = os.path.join("embed_stage", "by_output", key, base)
    candidates: List[str] = []
    scratch = os.environ.get("SCRATCH")
    if scratch:
        candidates.append(os.path.join(scratch, "fastplms_workspace", rel))
    candidates.append(os.path.join("/workspace", rel))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def choose_resume_db_path(db_path: str, prefer_staged: bool = True) -> Optional[str]:
    """Pick DB source for conversion; optionally prefer newer staged copy."""
    explicit = db_path if os.path.isfile(db_path) else None
    if not prefer_staged:
        return explicit
    staged = infer_staged_db_candidate(db_path)
    if staged is None:
        return explicit
    if explicit is None:
        return staged
    return staged if os.path.getmtime(staged) > os.path.getmtime(explicit) else explicit


def _iter_sqlite_embeddings(path: str, batch_size: int = 2048) -> Iterator[Tuple[str, torch.Tensor]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with sqlite3.connect(path, timeout=30) as conn:
        cur = conn.cursor()
        cur.execute("SELECT sequence, embedding FROM embeddings")
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for seq, blob in rows:
                yield str(seq), embedding_blob_to_tensor(blob)


def convert_db_to_zarr(
    *,
    db_path: str,
    zarr_path: str,
    manifest_path: Optional[str] = None,
    batch_size: int = 2048,
    timing: bool = False,
) -> int:
    """Convert SQLite `.db` embeddings into drorlab `.zarr` format (append/resume safe)."""
    _ensure_zarr()
    if manifest_path is None:
        manifest_path = default_manifest_path_for_zarr(zarr_path)

    seen = _load_manifest_sequences(manifest_path)
    root = zarr.open_group(zarr_path, mode="a")
    manifest_f = None
    manifest_writer = None
    row_count_written = 0
    residue_cursor = 0
    layout = root.attrs.get("layout")
    pooled_arr = None
    residues = None
    starts = None
    lengths = None

    t0 = torch.cuda.Event(enable_timing=False) if torch.cuda.is_available() else None
    _ = t0  # silence lint for conditional local use patterns

    for seq, emb in _iter_sqlite_embeddings(db_path, batch_size=batch_size):
        if seq in seen:
            continue

        if manifest_writer is None:
            if emb.ndim == 2:
                inferred_full = True
                hidden = int(emb.shape[1])
            elif emb.ndim == 1:
                inferred_full = False
                hidden = int(emb.shape[0])
            else:
                raise ValueError(f"Unsupported embedding rank in DB for {seq!r}: {tuple(emb.shape)}")

            if layout is None:
                root.attrs["layout"] = "full_embeddings" if inferred_full else "pooled_embeddings"
                layout = root.attrs["layout"]
            if layout not in {"full_embeddings", "pooled_embeddings"}:
                raise ValueError(f"Unsupported Zarr layout {layout!r} in {zarr_path}")
            if (layout == "full_embeddings") != inferred_full:
                raise ValueError(
                    f"DB tensor rank/layout mismatch for first new row {seq!r}: "
                    f"layout={layout}, tensor_shape={tuple(emb.shape)}"
                )

            manifest_f, manifest_writer = _open_manifest(manifest_path, full_embeddings=(layout == "full_embeddings"))
            if layout == "full_embeddings":
                residues = _ensure_array(root, "residues", shape=(0, hidden), chunks=(2048, hidden), dtype="f4")
                starts = _ensure_array(root, "row_start", shape=(0,), chunks=(16384,), dtype="i8")
                lengths = _ensure_array(root, "row_length", shape=(0,), chunks=(16384,), dtype="i4")
                residue_cursor = int(residues.shape[0])
                row_index = int(starts.shape[0])
            else:
                pooled_arr = _maybe_get_array(root, "pooled")
                if pooled_arr is None:
                    pooled_arr = _ensure_array(root, "pooled", shape=(0, hidden), chunks=(2048, hidden), dtype="f4")
                row_index = int(pooled_arr.shape[0])
        else:
            if layout == "full_embeddings":
                row_index = int(starts.shape[0])
            else:
                row_index = int(pooled_arr.shape[0])

        if layout == "full_embeddings":
            assert residues is not None and starts is not None and lengths is not None and manifest_writer is not None
            arr = emb.to(torch.float32).cpu().numpy()
            old_n = int(residues.shape[0])
            residues.resize((old_n + arr.shape[0], arr.shape[1]))
            residues[old_n:old_n + arr.shape[0], :] = arr
            old_rows = int(starts.shape[0])
            starts.resize((old_rows + 1,))
            lengths.resize((old_rows + 1,))
            starts[old_rows] = residue_cursor
            lengths[old_rows] = int(arr.shape[0])
            manifest_writer.writerow([row_index, seq, "", residue_cursor, int(arr.shape[0])])
            residue_cursor += int(arr.shape[0])
        else:
            assert pooled_arr is not None and manifest_writer is not None
            vec = emb.to(torch.float32).cpu().numpy()
            if vec.ndim != 1:
                raise ValueError(f"Expected pooled vector for layout=pooled_embeddings, got {tuple(vec.shape)}")
            old_n = int(pooled_arr.shape[0])
            pooled_arr.resize((old_n + 1, pooled_arr.shape[1]))
            pooled_arr[old_n, :] = vec
            manifest_writer.writerow([row_index, seq, ""])

        row_count_written += 1
        seen.add(seq)
        if manifest_f is not None and row_count_written % 1000 == 0:
            manifest_f.flush()

    if manifest_f is not None:
        manifest_f.flush()
        manifest_f.close()

    if timing:
        print(f"[timing:zarr-convert] wrote_rows={row_count_written}")
    print(f"Converted DB -> Zarr: {db_path} -> {zarr_path} (added {row_count_written} rows)")
    return row_count_written


def export_embeddings_to_zarr(
    *,
    model: torch.nn.Module,
    sequences: Sequence[str],
    sequence_to_id: Optional[Dict[str, str]],
    tokenizer: Optional[Any],
    save_path: str,
    manifest_path: str,
    batch_size: int,
    max_len: int,
    truncate: bool,
    full_embeddings: bool,
    embed_dtype: torch.dtype,
    pooling_types: List[str],
    num_workers: int = 0,
    timing: bool = False,
) -> None:
    _ensure_zarr()
    assert len(sequences) > 0, "No sequences provided."

    # Match core behavior: dedupe + length sort.
    seqs = list(set([seq[:max_len] if truncate else seq for seq in sequences]))
    seqs = sorted(seqs, key=len, reverse=True)

    seen = _load_manifest_sequences(manifest_path)
    to_embed = [s for s in seqs if s not in seen]
    print(f"Found {len(seen)} already embedded sequences in {manifest_path}")
    print(f"Embedding {len(to_embed)} new sequences")
    if len(to_embed) == 0:
        return

    root = zarr.open_group(save_path, mode="a")
    root.attrs["layout"] = "full_embeddings" if full_embeddings else "pooled_embeddings"
    root.attrs["embed_dtype"] = str(embed_dtype)
    if not full_embeddings:
        root.attrs["pooling_types"] = list(pooling_types)

    hidden_size = int(model.config.hidden_size)
    pooler = Pooler(pooling_types) if not full_embeddings else None
    tokenizer_mode = tokenizer is not None
    device = next(model.parameters()).device

    manifest_f, manifest_writer = _open_manifest(manifest_path, full_embeddings=full_embeddings)
    try:
        if full_embeddings:
            residues = _ensure_array(root, "residues", shape=(0, hidden_size), chunks=(2048, hidden_size), dtype="f4")
            starts = _ensure_array(root, "row_start", shape=(0,), chunks=(16384,), dtype="i8")
            lengths = _ensure_array(root, "row_length", shape=(0,), chunks=(16384,), dtype="i4")
            row_count = int(starts.shape[0])
            residue_cursor = int(residues.shape[0])
        else:
            pooled = _maybe_get_array(root, "pooled")
            row_count = int(pooled.shape[0]) if pooled is not None else 0

        def write_manifest_row(idx: int, seq: str, residue_start: Optional[int], residue_length: Optional[int]) -> None:
            sid = sequence_to_id.get(seq, "") if sequence_to_id is not None else ""
            if full_embeddings:
                assert residue_start is not None and residue_length is not None
                manifest_writer.writerow([idx, seq, sid, residue_start, residue_length])
            else:
                manifest_writer.writerow([idx, seq, sid])

        def handle_batch(batch_seqs: List[str], residue_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> None:
            nonlocal row_count
            nonlocal residue_cursor

            if full_embeddings:
                emb_cpu = residue_embeddings.to(embed_dtype).cpu()
                mask_cpu = attention_mask.bool().cpu()
                per_seq: List[np.ndarray] = []
                starts_batch: List[int] = []
                lens_batch: List[int] = []
                for seq, emb, mask in zip(batch_seqs, emb_cpu, mask_cpu):
                    trimmed = emb[mask].to(torch.float32).numpy()
                    per_seq.append(trimmed)
                    starts_batch.append(residue_cursor)
                    l = int(trimmed.shape[0])
                    lens_batch.append(l)
                    residue_cursor += l

                if per_seq:
                    flat = np.concatenate(per_seq, axis=0) if len(per_seq) > 1 else per_seq[0]
                    old_n = residues.shape[0]
                    residues.resize((old_n + flat.shape[0], hidden_size))
                    residues[old_n:old_n + flat.shape[0], :] = flat

                old_rows = starts.shape[0]
                n_new = len(batch_seqs)
                starts.resize((old_rows + n_new,))
                lengths.resize((old_rows + n_new,))
                starts[old_rows:old_rows + n_new] = np.asarray(starts_batch, dtype=np.int64)
                lengths[old_rows:old_rows + n_new] = np.asarray(lens_batch, dtype=np.int32)

                for seq, s, l in zip(batch_seqs, starts_batch, lens_batch):
                    write_manifest_row(row_count, seq, s, l)
                    row_count += 1
            else:
                assert pooler is not None
                pooled_t = pooler(residue_embeddings, attention_mask).to(embed_dtype).to(torch.float32).cpu().numpy()
                pooled_arr = _maybe_get_array(root, "pooled")
                if pooled_arr is None:
                    out_dim = int(pooled_t.shape[1])
                    pooled_arr = _ensure_array(root, "pooled", shape=(0, out_dim), chunks=(2048, out_dim), dtype="f4")
                old_n = pooled_arr.shape[0]
                n_new = int(pooled_t.shape[0])
                pooled_arr.resize((old_n + n_new, pooled_arr.shape[1]))
                pooled_arr[old_n:old_n + n_new, :] = pooled_t
                for seq in batch_seqs:
                    write_manifest_row(row_count, seq, None, None)
                    row_count += 1

            manifest_f.flush()

        if tokenizer_mode:
            collate_fn = build_collator(tokenizer, padding="longest", max_length=max_len, truncate=truncate)
            dataloader = DataLoader(
                ProteinDataset(to_embed),
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch_factor=2 if num_workers > 0 else None,
                collate_fn=collate_fn,
                shuffle=False,
                pin_memory=True,
            )
            for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Embedding batches (zarr)"):
                batch_seqs = to_embed[i * batch_size:(i + 1) * batch_size]
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                residue_embeddings = model._embed(input_ids, attention_mask)
                handle_batch(batch_seqs, residue_embeddings, attention_mask)
        else:
            for batch_start in tqdm(range(0, len(to_embed), batch_size), desc="Embedding batches (zarr)"):
                batch_seqs = to_embed[batch_start:batch_start + batch_size]
                out = model._embed(batch_seqs, return_attention_mask=True)
                assert isinstance(out, tuple) and len(out) == 2
                residue_embeddings, attention_mask = out
                handle_batch(batch_seqs, residue_embeddings, attention_mask)

        if timing:
            print(f"[timing:zarr] wrote_rows={row_count}")
        print(f"Wrote Zarr embeddings to {save_path} with manifest {manifest_path}")
    finally:
        manifest_f.close()
