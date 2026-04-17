#!/usr/bin/env python3
"""Load FastPLMs embedding artifacts from `.pth` or SQLite `.db` and map rows to per-residue indices.

Verified layout for `--full-embeddings` outputs from `embed.py` (EmbeddingMixin):

- **ESM2 / ESMC (tokenizer models):** stored length is ``len(sequence) + 2`` (cls/bos + residues + eos).
  Per-residue rows: ``full[1:-1]`` so index ``0`` is residue #1.
- **E1:** stored length is ``len(sequence) + 4`` (``<bos>``, ``1``, residues, ``2``, ``<eos>``).
  Per-residue rows: ``full[2:-2]`` so index ``0`` is residue #1.

SQLite rows use the compact blob format decoded by ``drorlab_fastplms.embedding_blob`` (header + raw bytes; same wire format as ``fastplms.embedding_mixin``).
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

from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

ModelFamily = Literal["auto", "esm_tokenizer", "e1", "ankh_single_special"]
DEFAULT_DB_BATCH_SIZE = 2048

# One residue selection: 1-based int, inclusive (start,end), gather list, slice on residue rows (0-based), or None = all.
ResidueSpecAtom = Union[int, Tuple[int, int], List[int], slice]
ResidueSpec = Optional[ResidueSpecAtom]


def _is_uniform_range_pair(obj: object) -> bool:
    """True for ``(start_1b, end_1b)`` inclusive range (two ints), not a parallel spec container."""
    return isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(x, int) for x in obj)


def _is_parallel_spec_list(sequences: Optional[List[str]], residue_number_1b: object) -> bool:
    """True when ``residue_number_1b`` is a list/tuple aligned 1:1 with ``sequences``.

    A length-2 tuple of ints is always treated as one inclusive range for uniform mode, never as two
    parallel specs — use ``[a, b]`` if two sequences need single residues ``a`` and ``b``.
    """
    if sequences is None or len(sequences) == 0:
        return False
    n = len(sequences)
    if isinstance(residue_number_1b, list):
        return len(residue_number_1b) == n
    if isinstance(residue_number_1b, tuple):
        if _is_uniform_range_pair(residue_number_1b):
            return False
        return len(residue_number_1b) == n
    return False


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
    residue_number_1b: ResidueSpec,
) -> torch.Tensor:
    """Apply optional residue selection on residue-aligned embeddings.

    Supported forms:
    - int: single residue position, 1-based (returns shape (H,))
    - tuple(start, end): inclusive 1-based range (returns view shape (R, H))
    - list[int]: specific 1-based residue positions (returns gathered tensor)
    - slice: row slice on the residue matrix (0-based; e.g. ``slice(None, 3)`` = first three residues)
    """
    if residue_number_1b is None:
        return residue_emb

    L = residue_emb.shape[0]
    if isinstance(residue_number_1b, slice):
        return residue_emb[residue_number_1b]
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
    residue_number_1b: ResidueSpec = None,
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
    residue_number_1b: ResidueSpec = None,
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


def _iter_per_residue_embs_ordered_pairs(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    pairs: List[Tuple[str, ResidueSpec]],
    family: ModelFamily = "auto",
    db_batch_size: int = DEFAULT_DB_BATCH_SIZE,
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield (sequence, tensor) in ``pairs`` order; supports duplicate sequence strings with different specs."""
    if isinstance(source, EmbeddingDBReader):
        seqs_unique = list(dict.fromkeys(s for s, _ in pairs))
        full_map: Dict[str, torch.Tensor] = {}
        for seq, full_emb in source.iter_full_embeddings(seqs_unique, batch_size=db_batch_size):
            full_map[seq] = full_emb
        for seq, spec in pairs:
            if seq not in full_map:
                raise KeyError(seq)
            yield seq, _apply_residue_number(
                _full_embeddings_to_residue_view(full_map[seq], seq, family=family),
                spec,
            )
        return

    if isinstance(source, str):
        if source.lower().endswith(".db"):
            with EmbeddingDBReader(source) as db:
                yield from _iter_per_residue_embs_ordered_pairs(
                    db, pairs, family=family, db_batch_size=db_batch_size
                )
            return

        full = load_embeddings_pth(source)
        for seq, spec in pairs:
            full_emb = full[seq]
            yield seq, _apply_residue_number(
                _full_embeddings_to_residue_view(full_emb, seq, family=family),
                spec,
            )
        return

    for seq, spec in pairs:
        full_emb = source[seq]
        yield seq, _apply_residue_number(
            _full_embeddings_to_residue_view(full_emb, seq, family=family),
            spec,
        )


def load_per_residue_embs(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    sequence: Optional[str] = None,
    sequences: Optional[List[str]] = None,
    family: ModelFamily = "auto",
    residue_number_1b: Optional[Union[ResidueSpecAtom, List[ResidueSpec], Tuple[ResidueSpec, ...]]] = None,
    residue_number_by_sequence: Optional[Dict[str, ResidueSpec]] = None,
    batch_size: Optional[int] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor], Iterator[Tuple[str, torch.Tensor]]]:
    """Load per-residue embeddings from `.db` / `.pth` (single unified entry point).

    **Single sequence:** pass ``sequence="ACDEF..."`` and optional ``residue_number_1b`` or a
    one-entry ``residue_number_by_sequence``. Returns a ``Tensor``.

    **Many sequences — uniform residue selection:** pass ``sequences=[...]`` (or ``None`` to load
    all rows) and ``residue_number_1b`` (same for every sequence). A length-2 tuple of ints is
    always one inclusive 1-based range, not two parallel specs.

    **Many sequences — different residue per row:** either

    - ``residue_number_by_sequence={seq: spec, ...}`` (optional ``sequences`` filters keys), or
    - ``sequences=[s0, s1, ...]`` with ``residue_number_1b=[spec0, spec1, ...]`` the same length
      (list or tuple). Each ``spec`` is an int, ``(start, end)`` inclusive range, ``list[int]``
      gather, ``slice`` on residue rows (0-based), or ``None`` for full per-residue tensor.

    **batch_size:**
    - ``None`` (default): return ``Dict[str, Tensor]`` with all requested sequences in memory.
    - ``1`` or ``>1``: return an iterator ``(sequence, tensor)``. For SQLite, ``batch_size`` is the
      DB fetch batch size (batched ``IN`` queries or ``fetchmany`` when iterating all rows).
    """
    if residue_number_1b is not None and residue_number_by_sequence is not None:
        raise ValueError("Use only one of residue_number_1b or residue_number_by_sequence.")

    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive or None")

    # --- Single-sequence -> Tensor ---
    if sequence is not None:
        if sequences is not None:
            raise ValueError("Pass either sequence=... or sequences=[...], not both.")
        if residue_number_by_sequence is not None:
            if sequence not in residue_number_by_sequence:
                raise KeyError(f"sequence {sequence!r} not in residue_number_by_sequence")
            rn = residue_number_by_sequence[sequence]
        else:
            rn = residue_number_1b
        return get_per_residue_embs(source, sequence, family=family, residue_number_1b=rn)

    # --- Per-sequence residue specs (dict or aligned list/tuple with sequences) ---
    eff_by_seq: Optional[Dict[str, ResidueSpec]] = None
    parallel_list = _is_parallel_spec_list(sequences, residue_number_1b)
    if residue_number_by_sequence is not None:
        if len(residue_number_by_sequence) == 0:
            raise ValueError("residue_number_by_sequence cannot be empty.")
        eff_by_seq = dict(residue_number_by_sequence)
    elif parallel_list:
        assert sequences is not None and isinstance(residue_number_1b, (list, tuple))

    if eff_by_seq is not None or parallel_list:
        if residue_number_by_sequence is not None:
            assert eff_by_seq is not None
            if sequences is None:
                pairs = list(eff_by_seq.items())
            else:
                missing = set(sequences) - set(eff_by_seq.keys())
                if missing:
                    raise KeyError(
                        f"sequences not found in residue_number_by_sequence: {sorted(missing)[:5]}..."
                    )
                pairs = [(s, eff_by_seq[s]) for s in sequences]
        else:
            assert parallel_list
            assert sequences is not None and isinstance(residue_number_1b, (list, tuple))
            pairs = list(zip(sequences, residue_number_1b))
            if batch_size is None and len(sequences) != len(set(sequences)):
                raise ValueError(
                    "Duplicate entries in sequences with batch_size=None (dict result would drop "
                    "rows): use batch_size>=1 to stream ordered (sequence, tensor) pairs, or use "
                    "unique sequence keys."
                )

        if batch_size is None:
            return {
                seq: emb
                for seq, emb in _iter_per_residue_embs_ordered_pairs(
                    source=source,
                    pairs=pairs,
                    family=family,
                    db_batch_size=DEFAULT_DB_BATCH_SIZE,
                )
            }
        return _iter_per_residue_embs_ordered_pairs(
            source=source,
            pairs=pairs,
            family=family,
            db_batch_size=batch_size,
        )

    # --- Uniform residue_number_1b for many sequences ---
    if batch_size is None:
        return load_all_per_residue_embs_in_memory(
            source=source,
            family=family,
            residue_number_1b=residue_number_1b,
            db_batch_size=DEFAULT_DB_BATCH_SIZE,
        ) if sequences is None else {
            seq: emb
            for seq, emb in iter_per_residue_embs(
                source=source,
                sequences=sequences,
                family=family,
                residue_number_1b=residue_number_1b,
                db_batch_size=DEFAULT_DB_BATCH_SIZE,
            )
        }

    if sequences is not None:
        return iter_per_residue_embs(
            source=source,
            sequences=sequences,
            family=family,
            residue_number_1b=residue_number_1b,
            db_batch_size=batch_size,
        )

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

    full = load_embeddings_pth(source) if isinstance(source, str) else source

    def _iter_map_all(m: Dict[str, torch.Tensor]) -> Iterator[Tuple[str, torch.Tensor]]:
        for seq, full_emb in m.items():
            yield seq, _apply_residue_number(
                _full_embeddings_to_residue_view(full_emb, seq, family=family),
                residue_number_1b,
            )

    return _iter_map_all(full)


def load_all_per_residue_embs_in_memory(
    source: Union[str, Dict[str, torch.Tensor], EmbeddingDBReader],
    family: ModelFamily = "auto",
    residue_number_1b: ResidueSpec = None,
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
            t = load_per_residue_embs(args.path, sequence=args.sequence, family=args.family)
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
