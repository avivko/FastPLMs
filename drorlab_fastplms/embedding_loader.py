#!/usr/bin/env python3
"""Load FastPLMs embedding artifacts from `.pth` or SQLite `.db` and map rows to per-residue indices.

Verified layout for `--full-embeddings` outputs from `embed.py` (EmbeddingMixin):

- **ESM2 / ESMC (tokenizer models):** stored length is ``len(sequence) + 2`` (cls/bos + residues + eos).
  Per-residue rows: ``full[1:-1]`` so index ``0`` is residue #1.
- **E1:** stored length is ``len(sequence) + 4`` (``<bos>``, ``1``, residues, ``2``, ``<eos>``).
  Per-residue rows: ``full[2:-2]`` so index ``0`` is residue #1.

SQLite rows use the compact blob format from ``fastplms.embedding_mixin`` (header + raw bytes).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastplms.embedding_mixin import embedding_blob_to_tensor

ModelFamily = Literal["auto", "esm_tokenizer", "e1", "ankh_single_special"]
DEFAULT_DB_BATCH_SIZE = 2048


def load_embeddings_pth(path: str) -> Dict[str, torch.Tensor]:
    """Load ``torch.save`` dict: sequence string -> tensor (vector or per-token matrix)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {path}, got {type(payload)}")
    for k, v in payload.items():
        if not isinstance(k, str) or not isinstance(v, torch.Tensor):
            raise TypeError(f"Expected str->Tensor entries in {path}")
    return payload


def load_embeddings_db(path: str, sequences: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Load SQLite table ``embeddings (sequence, embedding)``; blobs via ``embedding_blob_to_tensor``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out: Dict[str, torch.Tensor] = {}
    with sqlite3.connect(path, timeout=30) as conn:
        cur = conn.cursor()
        if sequences is None:
            cur.execute("SELECT sequence, embedding FROM embeddings")
            rows = cur.fetchall()
        else:
            if len(sequences) == 0:
                return {}
            ph = ",".join("?" * len(sequences))
            cur.execute(f"SELECT sequence, embedding FROM embeddings WHERE sequence IN ({ph})", tuple(sequences))
            rows = cur.fetchall()
        for seq, blob in rows:
            out[str(seq)] = embedding_blob_to_tensor(blob)
    return out


def load_embeddings(path: str, sequences: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    """Dispatch on file extension: ``.db`` -> SQLite; else treat as ``.pth``."""
    lower = path.lower()
    if lower.endswith(".db"):
        return load_embeddings_db(path, sequences=sequences)
    return load_embeddings_pth(path)


class EmbeddingDBReader:
    """Read embeddings from SQLite efficiently with one persistent connection."""

    def __init__(self, path: str, timeout: int = 30) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.path = path
        # Read-only URI avoids accidental writes/locks.
        uri = f"file:{os.path.abspath(path)}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, timeout=timeout)
        self._conn.execute("PRAGMA query_only=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EmbeddingDBReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])

    def get_full_embedding(self, sequence: str) -> torch.Tensor:
        row = self._conn.execute(
            "SELECT embedding FROM embeddings WHERE sequence=?",
            (sequence,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Sequence not found in {self.path}")
        return embedding_blob_to_tensor(row[0])

    def list_sequences(self, limit: int = 10) -> List[str]:
        rows = self._conn.execute(
            "SELECT sequence FROM embeddings ORDER BY sequence LIMIT ?",
            (limit,),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def iter_full_embeddings(
        self,
        sequences: List[str],
        batch_size: int = DEFAULT_DB_BATCH_SIZE,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield (sequence, full_embedding) using batched IN queries."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            placeholders = ",".join(["?"] * len(batch))
            rows = self._conn.execute(
                f"SELECT sequence, embedding FROM embeddings WHERE sequence IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            for seq, blob in rows:
                yield str(seq), embedding_blob_to_tensor(blob)

    def iter_all_full_embeddings(
        self,
        batch_size: int = DEFAULT_DB_BATCH_SIZE,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield all rows using one SELECT and fetchmany batches."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        cur = self._conn.cursor()
        cur.execute("SELECT sequence, embedding FROM embeddings")
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for seq, blob in rows:
                yield str(seq), embedding_blob_to_tensor(blob)


def infer_family(full_emb: torch.Tensor, sequence: str) -> ModelFamily:
    """Infer tokenizer vs E1 from stored token count (full-embeddings only)."""
    if full_emb.ndim != 2:
        raise ValueError(f"Expected 2D per-sequence embedding (T, H), got shape {tuple(full_emb.shape)}")
    L = len(sequence)
    t = full_emb.shape[0]
    if t == L + 4:
        return "e1"
    if t == L + 2:
        return "esm_tokenizer"
    if t == L + 1:
        return "ankh_single_special"
    raise ValueError(
        f"Cannot infer layout: seq_len={L}, stored_tokens={t}. "
        "Expected L+2 (ESM2/ESMC/DPLM/DPLM2), L+4 (E1), L+1 (ANKH), or L (already residue-only)."
    )


def _full_embeddings_to_residue_view(
    full_emb: torch.Tensor,
    sequence: str,
    family: ModelFamily = "auto",
) -> torch.Tensor:
    """Return a residue-aligned tensor view (row i -> residue sequence[i], 0-based)."""
    if full_emb.ndim != 2:
        raise ValueError(f"Expected 2D tensor (T, H), got {tuple(full_emb.shape)}")

    L = len(sequence)
    t = full_emb.shape[0]

    if family == "auto":
        if t == L:
            return full_emb
        family = infer_family(full_emb, sequence)

    if family == "e1":
        if t != L + 4:
            raise ValueError(f"E1 full-embeddings expected T=L+4={L+4}, got T={t}")
        return full_emb[2:-2]

    if family == "esm_tokenizer":
        if t != L + 2:
            raise ValueError(f"ESM tokenizer full-embeddings expected T=L+2={L+2}, got T={t}")
        return full_emb[1:-1]

    if family == "ankh_single_special":
        if t != L + 1:
            raise ValueError(f"ANKH full-embeddings expected T=L+1={L+1}, got T={t}")
        return full_emb[:-1]

    raise ValueError(f"Unknown family: {family}")


def _apply_residue_range_view(
    residue_emb: torch.Tensor,
    residue_range_1b: Optional[Tuple[int, int]],
) -> torch.Tensor:
    """Apply optional 1-based inclusive residue range and return a view."""
    if residue_range_1b is None:
        return residue_emb
    start_1b, end_1b = residue_range_1b
    if start_1b < 1 or end_1b < start_1b or end_1b > residue_emb.shape[0]:
        raise ValueError(
            f"Invalid residue range [{start_1b}, {end_1b}] for length {residue_emb.shape[0]}"
        )
    return residue_emb[start_1b - 1 : end_1b]


def _parse_residue_number_cli(spec: str) -> Union[int, Tuple[int, int], List[int]]:
    """Parse CLI residue number spec: '5', '10-25', or '3,8,21' (all 1-based)."""
    s = spec.strip()
    if "," in s:
        vals = [int(x.strip()) for x in s.split(",") if x.strip()]
        if len(vals) == 0:
            raise ValueError("Empty residue index list.")
        return vals
    if "-" in s:
        a_s, b_s = s.split("-", 1)
        return (int(a_s.strip()), int(b_s.strip()))
    return int(s)


def _apply_residue_number(
    residue_emb: torch.Tensor,
    residue_number_1b: Optional[Union[int, Tuple[int, int], List[int]]],
) -> torch.Tensor:
    """Apply optional 1-based residue numbering on residue-aligned embeddings.

    Supported forms:
    - int: single residue position (returns shape (H,))
    - tuple(start, end): inclusive range (returns view shape (R, H))
    - list[int]: specific residue positions (returns gathered tensor)
    """
    if residue_number_1b is None:
        return residue_emb

    L = residue_emb.shape[0]
    if isinstance(residue_number_1b, int):
        idx = residue_number_1b
        if idx < 1 or idx > L:
            raise ValueError(f"Invalid residue index {idx} for length {L}")
        return residue_emb[idx - 1]

    if isinstance(residue_number_1b, tuple):
        if len(residue_number_1b) != 2:
            raise ValueError("Range tuple must be (start_1b, end_1b).")
        return _apply_residue_range_view(residue_emb, residue_number_1b)

    if isinstance(residue_number_1b, list):
        if len(residue_number_1b) == 0:
            raise ValueError("Residue number list cannot be empty.")
        zero_based = []
        for idx in residue_number_1b:
            if idx < 1 or idx > L:
                raise ValueError(f"Invalid residue number {idx} for length {L}")
            zero_based.append(idx - 1)
        return residue_emb[torch.tensor(zero_based, dtype=torch.long)]

    raise ValueError(f"Unsupported residue number type: {type(residue_number_1b)}")


def get_per_residue_embs(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    sequence: str,
    family: ModelFamily = "auto",
    residue_number_1b: Optional[Union[int, Tuple[int, int], List[int]]] = None,
) -> torch.Tensor:
    """Get per-residue embeddings (or indexed subset) from `.db` / `.pth`.

    - `source`: either a path to `.db`/`.pth` or an already-loaded dict.
    - `sequence`: sequence key.
    - `family`: special-token layout (`auto` infers from stored length).
    - `residue_number_1b`: optional 1-based residue number spec:
      `int` (single residue number), `(start, end)` inclusive range, or `list[int]`.
    """
    if isinstance(source, EmbeddingDBReader):
        full_emb = source.get_full_embedding(sequence)
    elif isinstance(source, str):
        if source.lower().endswith(".db"):
            with EmbeddingDBReader(source) as db:
                full_emb = db.get_full_embedding(sequence)
        else:
            full_emb = load_embeddings_pth(source)[sequence]
    else:
        full_emb = source[sequence]

    residue_emb = _full_embeddings_to_residue_view(full_emb, sequence, family=family)
    return _apply_residue_number(residue_emb, residue_number_1b)


def iter_per_residue_embs(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    sequences: List[str],
    family: ModelFamily = "auto",
    residue_number_1b: Optional[Union[int, Tuple[int, int], List[int]]] = None,
    db_batch_size: int = DEFAULT_DB_BATCH_SIZE,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield per-residue embeddings for many sequences without loading everything into memory."""
    if isinstance(source, EmbeddingDBReader):
        for seq, full_emb in source.iter_full_embeddings(sequences, batch_size=db_batch_size):
            yield seq, _apply_residue_number(
                _full_embeddings_to_residue_view(full_emb, seq, family=family),
                residue_number_1b,
            )
        return

    if isinstance(source, str):
        if source.lower().endswith(".db"):
            with EmbeddingDBReader(source) as db:
                for seq, full_emb in db.iter_full_embeddings(sequences, batch_size=db_batch_size):
                    yield seq, _apply_residue_number(
                        _full_embeddings_to_residue_view(full_emb, seq, family=family),
                        residue_number_1b,
                    )
            return
        pth = load_embeddings_pth(source)
        for seq in sequences:
            full_emb = pth[seq]
            yield seq, _apply_residue_number(
                _full_embeddings_to_residue_view(full_emb, seq, family=family),
                residue_number_1b,
            )
        return

    for seq in sequences:
        full_emb = source[seq]
        yield seq, _apply_residue_number(
            _full_embeddings_to_residue_view(full_emb, seq, family=family),
            residue_number_1b,
        )


def load_per_residue_embs(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    sequences: Optional[List[str]] = None,
    family: ModelFamily = "auto",
    residue_number_1b: Optional[Union[int, Tuple[int, int], List[int]]] = None,
    batch_size: Optional[int] = None,
) -> Union[Dict[str, torch.Tensor], Iterator[Tuple[str, torch.Tensor]]]:
    """Unified loader for per-residue embeddings.

    Behavior by `batch_size`:
    - `None`: load selected entries fully in memory and return `Dict[str, Tensor]`.
    - `1`: return iterator yielding one sequence at a time.
    - `>1`: return iterator yielding sequence embeddings from batched fetch/decode.
    """
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive or None")

    # In-memory mode
    if batch_size is None:
        return load_all_per_residue_embs_in_memory(
            source=source,
            family=family,
            residue_number_1b=residue_number_1b,
            db_batch_size=DEFAULT_DB_BATCH_SIZE,
        ) if sequences is None else {
            seq: emb for seq, emb in iter_per_residue_embs(
                source=source,
                sequences=sequences,
                family=family,
                residue_number_1b=residue_number_1b,
                db_batch_size=DEFAULT_DB_BATCH_SIZE,
            )
        }

    # Iterator mode (batch_size >= 1)
    if sequences is not None:
        return iter_per_residue_embs(
            source=source,
            sequences=sequences,
            family=family,
            residue_number_1b=residue_number_1b,
            db_batch_size=batch_size,
        )

    # No explicit sequence list -> iterate all rows
    if isinstance(source, EmbeddingDBReader):
        def _iter_db_all(reader: EmbeddingDBReader) -> Iterator[Tuple[str, torch.Tensor]]:
            for seq, full_emb in reader.iter_all_full_embeddings(batch_size=batch_size):
                yield seq, _apply_residue_number(
                    _full_embeddings_to_residue_view(full_emb, seq, family=family),
                    residue_number_1b,
                )
        return _iter_db_all(source)

    if isinstance(source, str) and source.lower().endswith(".db"):
        def _iter_db_all_from_path(path: str) -> Iterator[Tuple[str, torch.Tensor]]:
            with EmbeddingDBReader(path) as db:
                for seq, full_emb in db.iter_all_full_embeddings(batch_size=batch_size):
                    yield seq, _apply_residue_number(
                        _full_embeddings_to_residue_view(full_emb, seq, family=family),
                        residue_number_1b,
                    )
        return _iter_db_all_from_path(source)

    # For .pth / dict, "all" iteration still reads in-memory source once.
    full = load_embeddings_pth(source) if isinstance(source, str) else source

    def _iter_map_all(m: Dict[str, torch.Tensor]) -> Iterator[Tuple[str, torch.Tensor]]:
        for seq, full_emb in m.items():
            yield seq, _apply_residue_number(
                _full_embeddings_to_residue_view(full_emb, seq, family=family),
                residue_number_1b,
            )
    return _iter_map_all(full)


def get_per_residue_embs_single(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    sequence: str,
    family: ModelFamily = "auto",
    residue_number_1b: Optional[Union[int, Tuple[int, int], List[int]]] = None,
) -> torch.Tensor:
    """Explicit single-sequence mode (batch size 1)."""
    return get_per_residue_embs(
        source=source,
        sequence=sequence,
        family=family,
        residue_number_1b=residue_number_1b,
    )


def load_all_per_residue_embs_in_memory(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    family: ModelFamily = "auto",
    residue_number_1b: Optional[Union[int, Tuple[int, int], List[int]]] = None,
    db_batch_size: int = DEFAULT_DB_BATCH_SIZE,
) -> Dict[str, torch.Tensor]:
    """Load all entries into memory as per-residue tensors.

    Use with care on very large datasets.
    """
    if isinstance(source, EmbeddingDBReader):
        seqs = source.list_sequences(limit=source.count())
        return {seq: emb for seq, emb in iter_per_residue_embs(
            source=source,
            sequences=seqs,
            family=family,
            residue_number_1b=residue_number_1b,
            db_batch_size=db_batch_size,
        )}

    if isinstance(source, str) and source.lower().endswith(".db"):
        with EmbeddingDBReader(source) as db:
            seqs = db.list_sequences(limit=db.count())
            return {seq: emb for seq, emb in iter_per_residue_embs(
                source=db,
                sequences=seqs,
                family=family,
                residue_number_1b=residue_number_1b,
                db_batch_size=db_batch_size,
            )}

    if isinstance(source, str):
        full = load_embeddings_pth(source)
    else:
        full = source
    return {
        seq: _apply_residue_number(
            _full_embeddings_to_residue_view(full_emb, seq, family=family),
            residue_number_1b,
        )
        for seq, full_emb in full.items()
    }


def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inspect / load FastPLMs embeddings (.pth or .db).")
    p.add_argument("--path", required=True, help="Path to .pth or .db")
    p.add_argument("--sequence", default=None, help="Sequence key to decode")
    p.add_argument(
        "--family",
        choices=("auto", "esm_tokenizer", "e1", "ankh_single_special"),
        default="auto",
        help="How to strip specials (default: infer from lengths)",
    )
    p.add_argument("--strip", action="store_true", help="Print per-residue shape")
    p.add_argument(
        "--residue-number",
        default=None,
        help="1-based residue number spec: '5' or '3-10' or '2,7,20'",
    )
    p.add_argument("--list-limit", type=int, default=10, help="How many keys to preview when --sequence is omitted")
    args = p.parse_args(argv)

    if args.sequence:
        try:
            t = get_per_residue_embs(args.path, args.sequence, family=args.family)
        except KeyError:
            print(f"Sequence not found in {args.path}", file=sys.stderr)
            return 2
        print("key:", args.sequence)
        print("per_residue_shape:", tuple(t.shape), "expected_L:", len(args.sequence))
        if args.strip:
            print("strip_mode: enabled")
        if args.residue_number:
            idx = _parse_residue_number_cli(args.residue_number)
            sl = _apply_residue_number(t, idx)
            print(f"residue_number_{args.residue_number}_shape:", tuple(sl.shape))
    else:
        if args.path.lower().endswith(".db"):
            with EmbeddingDBReader(args.path) as db:
                total = db.count()
                print(f"Embedding DB entries: {total} in {args.path}")
                if args.list_limit > 0:
                    keys = db.list_sequences(limit=args.list_limit)
                    for i, k in enumerate(keys):
                        print(i, k[:60] + ("..." if len(k) > 60 else ""))
                    if total > len(keys):
                        print("...")
        else:
            data = load_embeddings(args.path)
            print(f"Loaded {len(data)} entries from {args.path}")
            for i, k in enumerate(sorted(data.keys())):
                print(i, k[:60] + ("..." if len(k) > 60 else ""), tuple(data[k].shape))
                if i >= args.list_limit - 1:
                    if len(data) > args.list_limit:
                        print("...")
                    break
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
