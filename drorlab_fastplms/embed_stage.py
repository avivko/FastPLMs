"""Stable scratch staging paths for ``sbatch_embed.sbatch`` (``embed_stage/by_output/...``)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import List, Optional, Tuple

STAGE_OUTPUT_METADATA = "oak_output_path.txt"


def output_stage_hash(output_path: str) -> str:
    return hashlib.sha1(output_path.encode("utf-8")).hexdigest()


def output_basename(output_path: str) -> str:
    return os.path.basename(output_path.rstrip("/"))


def output_stage_label(output_path: str) -> str:
    """Human-readable label for stage dirs: parent folder name, else output basename."""
    trimmed = output_path.rstrip("/")
    parent = os.path.dirname(trimmed)
    if parent and parent not in (".", "/"):
        label = os.path.basename(parent)
        if label:
            return label
    return output_basename(trimmed)


def legacy_stage_dir_name(output_path: str) -> str:
    return output_stage_hash(output_path)


def hash_basename_stage_dir_name(output_path: str) -> str:
    """Previous naming scheme (``hash_basename``); kept for resume compatibility."""
    return f"{output_stage_hash(output_path)}_{output_basename(output_path)}"


def basename_hash_stage_dir_name(output_path: str) -> str:
    """Previous naming scheme (``basename_hash``); kept for resume compatibility."""
    return f"{output_basename(output_path)}_{output_stage_hash(output_path)}"


def stage_dir_name(output_path: str) -> str:
    """Directory name under ``embed_stage/by_output`` (parent folder label + hash)."""
    return f"{output_stage_label(output_path)}_{output_stage_hash(output_path)}"


def stage_rel_dir(output_path: str, *, scheme: str = "named") -> str:
    if scheme == "legacy":
        name = legacy_stage_dir_name(output_path)
    elif scheme == "hash_basename":
        name = hash_basename_stage_dir_name(output_path)
    elif scheme == "basename_hash":
        name = basename_hash_stage_dir_name(output_path)
    elif scheme == "named":
        name = stage_dir_name(output_path)
    else:
        raise ValueError(f"Unknown stage dir scheme {scheme!r}")
    return os.path.join("embed_stage", "by_output", name)


def staged_artifact_relpath(output_path: str, *, scheme: str = "named") -> str:
    return os.path.join(stage_rel_dir(output_path, scheme=scheme), output_basename(output_path))


def iter_stage_dir_candidates(output_path: str) -> List[Tuple[str, str]]:
    """Return ``(rel_dir, scheme)`` with preferred naming first, then older schemes."""
    return [
        (stage_rel_dir(output_path, scheme="named"), "named"),
        (stage_rel_dir(output_path, scheme="basename_hash"), "basename_hash"),
        (stage_rel_dir(output_path, scheme="hash_basename"), "hash_basename"),
        (stage_rel_dir(output_path, scheme="legacy"), "legacy"),
    ]


def stage_metadata_host_path(stage_dir_host: str) -> str:
    return os.path.join(stage_dir_host.rstrip("/"), STAGE_OUTPUT_METADATA)


def write_stage_metadata(stage_dir_host: str, output_path: str) -> str:
    """Write ``oak_output_path.txt`` under the stage directory; return its host path."""
    os.makedirs(stage_dir_host, exist_ok=True)
    meta_path = stage_metadata_host_path(stage_dir_host)
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(output_path.rstrip("/"))
        f.write("\n")
    return meta_path


def read_stage_metadata(stage_dir_host: str) -> Optional[str]:
    meta_path = stage_metadata_host_path(stage_dir_host)
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as f:
        line = f.readline().strip()
    return line or None


def _artifact_exists(path: str, *, is_zarr: bool) -> bool:
    if is_zarr:
        return os.path.isdir(path)
    return os.path.isfile(path)


def resolve_embed_stage(output_path: str, workspace_host: str) -> dict[str, object]:
    """Pick staged artifact path; prefer existing stage, else create under current naming."""
    base = output_basename(output_path)
    is_zarr = output_path.lower().endswith(".zarr")
    ws = workspace_host.rstrip("/")

    for rel_dir, scheme in iter_stage_dir_candidates(output_path):
        artifact_host = os.path.join(ws, rel_dir, base)
        if _artifact_exists(artifact_host, is_zarr=is_zarr):
            rel_artifact = os.path.join(rel_dir, base)
            return {
                "stage_host": artifact_host,
                "run_output": f"/workspace/{rel_artifact}",
                "stage_rel_dir": rel_dir,
                "scheme": scheme,
                "legacy": scheme != "named",
            }

    rel_dir = stage_rel_dir(output_path, scheme="named")
    artifact_host = os.path.join(ws, rel_dir, base)
    rel_artifact = os.path.join(rel_dir, base)
    return {
        "stage_host": artifact_host,
        "run_output": f"/workspace/{rel_artifact}",
        "stage_rel_dir": rel_dir,
        "scheme": "named",
        "legacy": False,
    }


def workspace_roots() -> List[str]:
    roots: List[str] = []
    scratch = os.environ.get("SCRATCH")
    if scratch:
        roots.append(os.path.join(scratch, "fastplms_workspace"))
    roots.append("/workspace")
    return roots


def infer_staged_db_candidate(db_path: str) -> Optional[str]:
    """Infer staged DB path (current naming, then older schemes)."""
    base = output_basename(db_path)
    for root in workspace_roots():
        for rel_dir, _ in iter_stage_dir_candidates(db_path):
            candidate = os.path.join(root, rel_dir, base)
            if os.path.isfile(candidate):
                return candidate
    return None


def _dir_matches_stage_hash(name: str, stage_hash: str) -> bool:
    if name == stage_hash:
        return True
    if name.startswith(f"{stage_hash}_"):
        return True
    if name.endswith(f"_{stage_hash}"):
        return True
    return False


def lookup_stage_hash(stage_hash: str, workspace_host: str) -> List[str]:
    """List staged artifacts whose directory matches a legacy or named stage hash."""
    base_dir = os.path.join(workspace_host.rstrip("/"), "embed_stage", "by_output")
    if not os.path.isdir(base_dir):
        return []

    matches: List[str] = []
    for name in sorted(os.listdir(base_dir)):
        if not _dir_matches_stage_hash(name, stage_hash):
            continue
        inner = os.path.join(base_dir, name)
        if not os.path.isdir(inner):
            continue
        oak_path = read_stage_metadata(inner)
        prefix = f"{name}: {oak_path}" if oak_path else name
        for item in sorted(os.listdir(inner)):
            if item == STAGE_OUTPUT_METADATA:
                continue
            matches.append(f"{prefix}/{item}")
    return matches


def _cmd_resolve(args: argparse.Namespace) -> int:
    info = resolve_embed_stage(args.output, args.workspace)
    print(json.dumps(info, indent=2))
    return 0


def _cmd_lookup(args: argparse.Namespace) -> int:
    matches = lookup_stage_hash(args.hash, args.workspace)
    if not matches:
        print(f"No staged artifacts found for hash {args.hash!r} under {args.workspace}")
        return 1
    for item in matches:
        print(item)
    return 0


def _cmd_hash(args: argparse.Namespace) -> int:
    print(output_stage_hash(args.output))
    print(stage_dir_name(args.output))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Embed scratch stage path helpers.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_resolve = sub.add_parser("resolve", help="Resolve stage paths for an OUTPUT path.")
    p_resolve.add_argument("output", help="Final OUTPUT path passed to sbatch_embed.")
    p_resolve.add_argument(
        "workspace",
        nargs="?",
        default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "fastplms_workspace"),
        help="Host workspace root (default: $SCRATCH/fastplms_workspace).",
    )
    p_resolve.set_defaults(func=_cmd_resolve)

    p_lookup = sub.add_parser("lookup", help="List staged files for a stage hash directory name.")
    p_lookup.add_argument("hash", help="40-char sha1 (legacy dir, hash_basename, or basename_hash).")
    p_lookup.add_argument(
        "workspace",
        nargs="?",
        default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "fastplms_workspace"),
    )
    p_lookup.set_defaults(func=_cmd_lookup)

    p_hash = sub.add_parser("hash", help="Print sha1 hash and stage directory name for OUTPUT.")
    p_hash.add_argument("output")
    p_hash.set_defaults(func=_cmd_hash)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
