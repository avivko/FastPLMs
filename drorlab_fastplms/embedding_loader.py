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
from typing import Dict, List, Literal, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastplms.embedding_mixin import embedding_blob_to_tensor

ModelFamily = Literal["auto", "esm_tokenizer", "e1", "ankh_single_special"]


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


def full_embeddings_to_residues(
    full_emb: torch.Tensor,
    sequence: str,
    family: ModelFamily = "auto",
) -> torch.Tensor:
    """Strip special/boundary tokens so row ``i`` aligns with residue ``sequence[i]`` (0-based)."""
    if full_emb.ndim != 2:
        raise ValueError(f"Expected 2D tensor (T, H), got {tuple(full_emb.shape)}")

    L = len(sequence)
    t = full_emb.shape[0]

    if family == "auto":
        if t == L:
            return full_emb.clone()
        family = infer_family(full_emb, sequence)

    if family == "e1":
        if t != L + 4:
            raise ValueError(f"E1 full-embeddings expected T=L+4={L+4}, got T={t}")
        return full_emb[2:-2].contiguous()

    if family == "esm_tokenizer":
        if t != L + 2:
            raise ValueError(f"ESM tokenizer full-embeddings expected T=L+2={L+2}, got T={t}")
        return full_emb[1:-1].contiguous()

    if family == "ankh_single_special":
        if t != L + 1:
            raise ValueError(f"ANKH full-embeddings expected T=L+1={L+1}, got T={t}")
        return full_emb[:-1].contiguous()

    raise ValueError(f"Unknown family: {family}")


def mean_residue_range(
    residue_emb: torch.Tensor,
    start_1b: int,
    end_1b: int,
) -> torch.Tensor:
    """Mean over residues ``start_1b``..``end_1b`` inclusive (1-based sequence positions)."""
    if residue_emb.ndim != 2:
        raise ValueError(f"Expected (L, H), got {tuple(residue_emb.shape)}")
    if start_1b < 1 or end_1b < start_1b:
        raise ValueError(f"Invalid range [{start_1b}, {end_1b}]")
    sl = residue_emb[start_1b - 1 : end_1b]
    return sl.mean(dim=0)


def _main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Inspect / load FastPLMs embeddings (.pth or .db).")
    p.add_argument("--path", required=True, help="Path to .pth or .db")
    p.add_argument("--sequence", default=None, help="Sequence key to decode (required for --strip)")
    p.add_argument(
        "--family",
        choices=("auto", "esm_tokenizer", "e1", "ankh_single_special"),
        default="auto",
        help="How to strip specials (default: infer from lengths)",
    )
    p.add_argument("--strip", action="store_true", help="Print residue-only shape after stripping")
    p.add_argument("--mean-range", default=None, help="1-based inclusive range like 3-10 for mean pooling")
    args = p.parse_args(argv)

    data = load_embeddings(args.path, sequences=[args.sequence] if args.sequence else None)
    if args.sequence:
        if args.sequence not in data:
            print(f"Sequence not found in {args.path}", file=sys.stderr)
            return 2
        t = data[args.sequence]
        print("key:", args.sequence)
        print("stored_shape:", tuple(t.shape))
        if args.strip:
            r = full_embeddings_to_residues(t, args.sequence, family=args.family)  # type: ignore[arg-type]
            print("residue_shape:", tuple(r.shape), "expected_L:", len(args.sequence))
            if args.mean_range:
                a, b = (int(x) for x in args.mean_range.split("-", 1))
                m = mean_residue_range(r, a, b)
                print(f"mean_{a}_{b}_shape:", tuple(m.shape))
    else:
        print(f"Loaded {len(data)} entries from {args.path}")
        for i, k in enumerate(sorted(data.keys())):
            print(i, k[:60] + ("..." if len(k) > 60 else ""), tuple(data[k].shape))
            if i >= 9:
                print("...")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
