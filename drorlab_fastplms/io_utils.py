"""Load sequences (and optional ids) from CSV or FASTA for Drorlab CLIs."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import pandas as pd

from fastplms.embedding_mixin import parse_fasta


def load_sequences_fasta(path: str) -> Tuple[List[str], None]:
    """Return (sequences, None) — FASTA has no row ids unless headers are handled elsewhere."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"FASTA not found: {path}")
    seqs = parse_fasta(path)
    if not seqs:
        raise ValueError(f"No sequences parsed from {path}")
    return seqs, None


def load_sequences_csv(
    path: str,
    seq_col: str,
    id_col: Optional[str] = None,
) -> Tuple[List[str], Optional[List[str]]]:
    """Load ``seq_col`` from CSV; optional ``id_col`` for manifest output."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if seq_col not in df.columns:
        raise ValueError(f"Column {seq_col!r} not in CSV. Available: {list(df.columns)}")
    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"Column {id_col!r} not in CSV. Available: {list(df.columns)}")
    sequences: List[str] = []
    ids: List[str] = []
    for i, row in df.iterrows():
        raw = row[seq_col]
        if pd.isna(raw):
            continue
        s = str(raw).strip()
        if not s:
            continue
        sequences.append(s)
        if id_col is not None:
            rid = row[id_col]
            ids.append("" if pd.isna(rid) else str(rid))
    if not sequences:
        raise ValueError(f"No non-empty sequences in {path} column {seq_col!r}")
    if id_col is None:
        return sequences, None
    return sequences, ids


def load_csv_as_records(path: str) -> List[dict]:
    """Return list of row dicts (column -> scalar) for per-row E1 assembly."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
