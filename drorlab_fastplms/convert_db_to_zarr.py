#!/usr/bin/env python3
"""Convert drorlab SQLite embedding DB to drorlab Zarr embedding store."""

from __future__ import annotations

import argparse

from drorlab_fastplms.zarr_export import convert_db_to_zarr, default_manifest_path_for_zarr


def main() -> int:
    p = argparse.ArgumentParser(description="Convert embedding .db to .zarr (append/resume-safe).")
    p.add_argument("--db", required=True, help="Input SQLite embedding DB path")
    p.add_argument("--zarr", required=True, help="Output Zarr store path")
    p.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest CSV path (default: <zarr_stem>_manifest.csv)",
    )
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--timing", action="store_true")
    args = p.parse_args()

    manifest = args.manifest or default_manifest_path_for_zarr(args.zarr)
    convert_db_to_zarr(
        db_path=args.db,
        zarr_path=args.zarr,
        manifest_path=manifest,
        batch_size=args.batch_size,
        timing=args.timing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
