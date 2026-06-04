"""Zarr export utilities for drorlab embedding CLI (no fastplms core changes)."""

from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

from fastplms.embedding_mixin import Pooler, _trim_full_embedding, build_collator
from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

try:
    import zarr
except Exception:  # pragma: no cover - runtime dependency check
    zarr = None

# Default ~96 MiB per ``residues`` chunk (float32 row = 4 * hidden_size bytes).
DEFAULT_ZARR_TARGET_CHUNK_BYTES = 96 * 1024 * 1024
DEFAULT_ZARR_INDEX_CHUNK_ROWS = 262_144
MIN_ZARR_RESIDUE_CHUNK_ROWS = 2048
MAX_ZARR_RESIDUE_CHUNK_ROWS = 65_536
# Legacy defaults (pre-2026 chunk tuning); used only if attrs missing on resume.
LEGACY_ZARR_RESIDUE_CHUNK_ROWS = 2048
LEGACY_ZARR_INDEX_CHUNK_ROWS = 16_384


@dataclass(frozen=True)
class ZarrChunkConfig:
    """Chunk shapes for new Zarr arrays. Existing arrays keep their creation-time chunks."""

    target_chunk_bytes: int = DEFAULT_ZARR_TARGET_CHUNK_BYTES
    index_chunk_rows: int = DEFAULT_ZARR_INDEX_CHUNK_ROWS

    def residue_chunk_rows(self, hidden_size: int) -> int:
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        bytes_per_row = hidden_size * 4  # f4
        raw_rows = max(1, self.target_chunk_bytes // bytes_per_row)
        rows = _round_up_power_of_two(raw_rows)
        return int(min(MAX_ZARR_RESIDUE_CHUNK_ROWS, max(MIN_ZARR_RESIDUE_CHUNK_ROWS, rows)))

    def residues_chunks(self, hidden_size: int) -> Tuple[int, int]:
        return (self.residue_chunk_rows(hidden_size), hidden_size)

    def index_chunks(self) -> Tuple[int]:
        return (self.index_chunk_rows,)


def _round_up_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def zarr_chunk_config_from_target_mb(
    target_mb: float,
    index_chunk_rows: int = DEFAULT_ZARR_INDEX_CHUNK_ROWS,
) -> ZarrChunkConfig:
    if target_mb <= 0:
        raise ValueError(f"--zarr-chunk-target-mb must be positive, got {target_mb}")
    return ZarrChunkConfig(
        target_chunk_bytes=int(target_mb * 1024 * 1024),
        index_chunk_rows=index_chunk_rows,
    )


def _full_trimmed_zarr_flat(trimmed: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Flatten a trimmed full embedding for Zarr ``residues`` storage.

    Returns ``(flat_2d [N, H], flat_row_count N, num_hidden_states)``.
    For a single layer, ``num_hidden_states`` is 1 and ``flat_2d`` is ``(T, H)``.
    For all layers, input is ``(L, T, H)`` stored row-major as ``(L*T, H)``.
    """
    if trimmed.ndim == 2:
        return trimmed, int(trimmed.shape[0]), 1
    if trimmed.ndim == 3:
        n_layers, n_tok, hidden = trimmed.shape
        flat = np.ascontiguousarray(trimmed.reshape(n_layers * n_tok, hidden))
        return flat, int(n_layers * n_tok), int(n_layers)
    raise ValueError(f"Expected trimmed full embedding with 2 or 3 dims, got {trimmed.ndim}")


def zarr_flat_to_full_embedding(flat_2d: torch.Tensor, flat_rows: int, num_hidden_states: int) -> torch.Tensor:
    """Inverse of ``_full_trimmed_zarr_flat`` for ``EmbeddingZarrReader``."""
    if num_hidden_states <= 1:
        return flat_2d
    if flat_rows % num_hidden_states != 0:
        raise ValueError(
            f"Cannot reshape Zarr flat rows {flat_rows} into "
            f"{num_hidden_states} layers (not divisible)"
        )
    n_tok = flat_rows // num_hidden_states
    return flat_2d.reshape(num_hidden_states, n_tok, flat_2d.shape[-1])


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
    from drorlab_fastplms.embed_stage import infer_staged_db_candidate as _infer

    return _infer(db_path)


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
    zarr_chunks: Optional[ZarrChunkConfig] = None,
) -> int:
    """Convert SQLite `.db` embeddings into drorlab `.zarr` format (append/resume safe)."""
    _ensure_zarr()
    chunk_cfg = zarr_chunks or ZarrChunkConfig()
    if manifest_path is None:
        manifest_path = default_manifest_path_for_zarr(zarr_path)

    seen = _load_manifest_sequences(manifest_path)
    root = zarr.open_group(zarr_path, mode="a")
    if "zarr_chunk_target_bytes" not in root.attrs:
        root.attrs["zarr_chunk_target_bytes"] = int(chunk_cfg.target_chunk_bytes)
        root.attrs["zarr_index_chunk_rows"] = int(chunk_cfg.index_chunk_rows)
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
            if emb.ndim == 3:
                inferred_full = True
                hidden = int(emb.shape[2])
                root.attrs["store_all_hidden_states"] = True
                root.attrs["num_hidden_states"] = int(emb.shape[0])
            elif emb.ndim == 2:
                inferred_full = True
                hidden = int(emb.shape[1])
                root.attrs.setdefault("store_all_hidden_states", False)
                root.attrs.setdefault("num_hidden_states", 1)
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
                residues = _ensure_array(
                    root,
                    "residues",
                    shape=(0, hidden),
                    chunks=chunk_cfg.residues_chunks(hidden),
                    dtype="f4",
                )
                starts = _ensure_array(
                    root, "row_start", shape=(0,), chunks=chunk_cfg.index_chunks(), dtype="i8"
                )
                lengths = _ensure_array(
                    root, "row_length", shape=(0,), chunks=chunk_cfg.index_chunks(), dtype="i4"
                )
                residue_cursor = int(residues.shape[0])
                row_index = int(starts.shape[0])
            else:
                pooled_arr = _maybe_get_array(root, "pooled")
                if pooled_arr is None:
                    pooled_arr = _ensure_array(
                        root,
                        "pooled",
                        shape=(0, hidden),
                        chunks=chunk_cfg.residues_chunks(hidden),
                        dtype="f4",
                    )
                row_index = int(pooled_arr.shape[0])
        else:
            if layout == "full_embeddings":
                row_index = int(starts.shape[0])
            else:
                row_index = int(pooled_arr.shape[0])

        if layout == "full_embeddings":
            assert residues is not None and starts is not None and lengths is not None and manifest_writer is not None
            arr_np = emb.to(torch.float32).cpu().numpy()
            if arr_np.ndim == 3:
                n_layers, n_tok, hidden = arr_np.shape
                arr_np = np.ascontiguousarray(arr_np.reshape(n_layers * n_tok, hidden))
            elif arr_np.ndim != 2:
                raise ValueError(f"Expected full embedding rank 2 or 3 in DB, got {arr_np.ndim}")
            old_n = int(residues.shape[0])
            residues.resize((old_n + arr_np.shape[0], arr_np.shape[1]))
            residues[old_n:old_n + arr_np.shape[0], :] = arr_np
            old_rows = int(starts.shape[0])
            starts.resize((old_rows + 1,))
            lengths.resize((old_rows + 1,))
            starts[old_rows] = residue_cursor
            lengths[old_rows] = int(arr_np.shape[0])
            manifest_writer.writerow([row_index, seq, "", residue_cursor, int(arr_np.shape[0])])
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
    hidden_state_index: int = -1,
    store_all_hidden_states: bool = False,
    num_workers: int = 0,
    timing: bool = False,
    zarr_chunks: Optional[ZarrChunkConfig] = None,
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

    chunk_cfg = zarr_chunks or ZarrChunkConfig()
    root = zarr.open_group(save_path, mode="a")
    root.attrs["layout"] = "full_embeddings" if full_embeddings else "pooled_embeddings"
    root.attrs["embed_dtype"] = str(embed_dtype)
    root.attrs["store_all_hidden_states"] = bool(store_all_hidden_states)
    root.attrs["hidden_state_index"] = int(hidden_state_index)
    if "zarr_chunk_target_bytes" not in root.attrs:
        root.attrs["zarr_chunk_target_bytes"] = int(chunk_cfg.target_chunk_bytes)
        root.attrs["zarr_index_chunk_rows"] = int(chunk_cfg.index_chunk_rows)
    if not full_embeddings:
        root.attrs["pooling_types"] = list(pooling_types)

    hidden_size = int(model.config.hidden_size)
    if timing or len(to_embed) > 0:
        r_rows = chunk_cfg.residue_chunk_rows(hidden_size)
        print(
            f"[zarr] chunk target ~{chunk_cfg.target_chunk_bytes // (1024 * 1024)} MiB; "
            f"residues chunks=({r_rows}, {hidden_size}), "
            f"index chunks=({chunk_cfg.index_chunk_rows},) "
            f"(existing arrays unchanged on resume)"
        )
    pooler = Pooler(pooling_types) if not full_embeddings else None
    if tokenizer is None and getattr(model.config, "model_type", None) != "E1":
        tokenizer = getattr(model, "tokenizer", None)
    tokenizer_mode = tokenizer is not None
    device = next(model.parameters()).device

    manifest_f, manifest_writer = _open_manifest(manifest_path, full_embeddings=full_embeddings)
    try:
        if full_embeddings:
            residues = _ensure_array(
                root,
                "residues",
                shape=(0, hidden_size),
                chunks=chunk_cfg.residues_chunks(hidden_size),
                dtype="f4",
            )
            starts = _ensure_array(
                root, "row_start", shape=(0,), chunks=chunk_cfg.index_chunks(), dtype="i8"
            )
            lengths = _ensure_array(
                root, "row_length", shape=(0,), chunks=chunk_cfg.index_chunks(), dtype="i4"
            )
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
                    trimmed = _trim_full_embedding(emb, mask).to(torch.float32).numpy()
                    flat, flat_rows, n_layers = _full_trimmed_zarr_flat(trimmed)
                    if n_layers > 1:
                        root.attrs["store_all_hidden_states"] = True
                        root.attrs["num_hidden_states"] = n_layers
                    elif "num_hidden_states" not in root.attrs:
                        root.attrs["num_hidden_states"] = 1
                    per_seq.append(flat)
                    starts_batch.append(residue_cursor)
                    lens_batch.append(flat_rows)
                    residue_cursor += flat_rows

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
                    pooled_arr = _ensure_array(
                        root,
                        "pooled",
                        shape=(0, out_dim),
                        chunks=chunk_cfg.residues_chunks(out_dim),
                        dtype="f4",
                    )
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
                residue_embeddings = model._embed(
                    input_ids,
                    attention_mask,
                    hidden_state_index=hidden_state_index,
                    store_all_hidden_states=store_all_hidden_states,
                )
                handle_batch(batch_seqs, residue_embeddings, attention_mask)
        else:
            for batch_start in tqdm(range(0, len(to_embed), batch_size), desc="Embedding batches (zarr)"):
                batch_seqs = to_embed[batch_start:batch_start + batch_size]
                out = model._embed(
                    batch_seqs,
                    return_attention_mask=True,
                    hidden_state_index=hidden_state_index,
                    store_all_hidden_states=store_all_hidden_states,
                )
                assert isinstance(out, tuple) and len(out) == 2
                residue_embeddings, attention_mask = out
                handle_batch(batch_seqs, residue_embeddings, attention_mask)

        if timing:
            print(f"[timing:zarr] wrote_rows={row_count}")
        print(f"Wrote Zarr embeddings to {save_path} with manifest {manifest_path}")
    finally:
        manifest_f.close()
