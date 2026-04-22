#!/usr/bin/env python3
"""Benchmark loader time / memory for a subset of sequences."""

from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import time
from typing import List, Optional

from drorlab_fastplms.embedding_loader import load_per_residue_embs


def list_sequences(db_path: str, n: int) -> List[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT sequence FROM embeddings ORDER BY rowid LIMIT ?",
            (n,),
        ).fetchall()
    return [str(r[0]) for r in rows]


def run_once(
    db_path: str,
    n: int,
    batch_size: Optional[int],
    family: str,
    residue_number: Optional[int],
) -> dict:
    seqs = list_sequences(db_path, n)
    if len(seqs) != n:
        raise ValueError(f"Requested n={n} but DB only has {len(seqs)} rows")

    t0 = time.perf_counter()
    if batch_size is None:
        out = load_per_residue_embs(
            db_path,
            sequences=seqs,
            family=family,
            residue_number_1b=residue_number,
            batch_size=None,
        )
        assert isinstance(out, dict)
        rows = 0
        elems = 0
        for emb in out.values():
            rows += int(emb.shape[0]) if emb.ndim == 2 else 1
            elems += int(emb.numel())
    else:
        it = load_per_residue_embs(
            db_path,
            sequences=seqs,
            family=family,
            residue_number_1b=residue_number,
            batch_size=batch_size,
        )
        rows = 0
        elems = 0
        count = 0
        for _seq, emb in it:
            count += 1
            rows += int(emb.shape[0]) if emb.ndim == 2 else 1
            elems += int(emb.numel())
        if count != n:
            raise RuntimeError(f"Iterator returned {count} items, expected {n}")
    elapsed_s = time.perf_counter() - t0
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "n_sequences": n,
        "batch_size": batch_size,
        "family": family,
        "residue_number_1b": residue_number,
        "elapsed_s": elapsed_s,
        "throughput_seq_per_s": n / elapsed_s if elapsed_s > 0 else None,
        "decoded_rows": rows,
        "decoded_elements": elems,
        "peak_rss_kb": int(peak_rss_kb),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--batch-size", default="none", help="'none' or positive int")
    p.add_argument("--family", default="auto")
    p.add_argument("--residue-number-1b", type=int, default=None)
    args = p.parse_args()

    bs = None if str(args.batch_size).lower() == "none" else int(args.batch_size)
    if bs is not None and bs <= 0:
        raise ValueError("batch-size must be > 0 or 'none'")

    res = run_once(
        db_path=args.db,
        n=args.n,
        batch_size=bs,
        family=args.family,
        residue_number=args.residue_number_1b,
    )
    print(json.dumps(res))


if __name__ == "__main__":
    main()
