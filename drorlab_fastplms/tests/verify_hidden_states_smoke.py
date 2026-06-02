#!/usr/bin/env python3
"""Verify embed.py hidden-state outputs (run after manual embed smoke in Docker)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from embed_test_utils import assert_store_all_hidden_states_artifacts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default="ACDEFGHIKLMNPQRSTVWY")
    p.add_argument("--hidden0-db", required=True)
    p.add_argument("--all-db", required=True)
    p.add_argument("--all-zarr", required=True)
    p.add_argument("--all-pth", required=True)
    args = p.parse_args()

    assert_store_all_hidden_states_artifacts(
        args.seq,
        hidden0_db=args.hidden0_db,
        all_db=args.all_db,
        all_zarr=args.all_zarr,
        all_pth=args.all_pth,
    )
    print(f"OK seq={args.seq!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
