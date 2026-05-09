#!/bin/bash
#SBATCH --job-name=fp_embed
#SBATCH --partition=owners,rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=120:00:00
#SBATCH --output=/oak/stanford/groups/rondror/software/plms/slurm_outs/fp_embed_%j.out
#SBATCH --error=/oak/stanford/groups/rondror/software/plms/slurm_outs/fp_embed_%j.err
#
# Usage (from repo root; paths are inside the container, usually under /workspace):
#   sbatch drorlab_fastplms/sbatch_embed.sbatch <MODEL> <INPUT> <OUTPUT> [extra embed.py args...]
#
# Example:
#   sbatch drorlab_fastplms/sbatch_embed.sbatch \
#     Synthyra/ESM2-8M /workspace/data/in.csv /workspace/runs/emb.pth \
#     --seq-col sequence --batch-size 16
#
# Optional env: SIF, REPO_ROOT, HF_TOKEN, HUGGING_FACE_HUB_TOKEN
#
# Zarr scratch→Oak sync (rsync): defaults tuned for many small chunk files.
#   FASTPLMS_RSYNC_PARALLEL=N   Rsync immediate children of the store with up to N parallel jobs
#                               (default 8). Set to 1 for a single full-tree rsync (also enables
#                               FASTPLMS_RSYNC_PROGRESS below).
#   FASTPLMS_RSYNC_PROGRESS=1   Pass --info=progress2 (only when PARALLEL is 1).
#
# Stageable outputs (*.db, *.zarr): by default stage on $SCRATCH for fast local I/O, then sync back
# to final OUTPUT after the job. Stage path is stable per OUTPUT so a new job can resume from the
# latest staged artifact after interrupted runs. Disable with FASTPLMS_NO_EMBED_STAGE=1
# (legacy alias: FASTPLMS_NO_EMBED_DB_STAGE=1).

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "sbatch_embed: need at least 3 arguments: MODEL INPUT OUTPUT [embed.py args...]" >&2
  echo "Example: sbatch $0 Synthyra/ESM2-8M /workspace/in.csv /workspace/out.pth --seq-col sequence" >&2
  exit 1
fi

MODEL=$1
INPUT=$2
OUTPUT=$3
shift 3

SIF="${SIF:-/oak/stanford/groups/rondror/software/plms/singularity/fastplms.sif}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

mkdir -p "${SCRATCH}/fastplms_workspace"

_EMBED_STAGE_HOST=""
_EMBED_STAGE_KIND=""
RUN_OUTPUT="$OUTPUT"

_sqlite_maybe_checkpoint() {
  local db="$1"
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
  fi
}

_sync_tree() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  if command -v rsync >/dev/null 2>&1; then
    # --whole-file: skip rolling checksums (often much faster scratch→NFS than default delta mode).
    local jobs="${FASTPLMS_RSYNC_PARALLEL:-8}"
    if ! [[ "$jobs" =~ ^[0-9]+$ ]]; then
      jobs=8
    fi
    if [[ "$jobs" -gt 1 ]]; then
      # One rsync per top-level member (array dirs + root metadata files).
      find "${src%/}" -mindepth 1 -maxdepth 1 -print0 2>/dev/null \
        | xargs -0 -r -P "$jobs" -n1 bash -c '
          dstroot="$1"
          item="$2"
          bn=$(basename "$item")
          if [[ -d "$item" ]]; then
            mkdir -p "${dstroot%/}/$bn"
            rsync -a --whole-file --delete "${item%/}/" "${dstroot%/}/$bn/"
          else
            rsync -a --whole-file "$item" "${dstroot%/}/$bn"
          fi
        ' bash "$dst"
    else
      local rsync_opts=( -a --whole-file --delete )
      [[ "${FASTPLMS_RSYNC_PROGRESS:-0}" == "1" ]] && rsync_opts+=( --info=progress2 )
      rsync "${rsync_opts[@]}" "${src%/}/" "${dst%/}/"
    fi
  else
    rm -rf "${dst%/}"/* "${dst%/}"/.[!.]* "${dst%/}"/..?* 2>/dev/null || true
    cp -a "${src%/}/." "${dst%/}/"
  fi
}

_NO_STAGE="${FASTPLMS_NO_EMBED_STAGE:-${FASTPLMS_NO_EMBED_DB_STAGE:-0}}"
if [[ "$_NO_STAGE" != "1" ]] && [[ "${OUTPUT,,}" == *.db || "${OUTPUT,,}" == *.zarr ]]; then
  _OUT_KEY="$(printf '%s' "$OUTPUT" | sha1sum | awk '{print $1}')"
  _OUT_BASE="$(basename "$OUTPUT")"
  _STAGE_REL="embed_stage/by_output/${_OUT_KEY}"
  mkdir -p "${SCRATCH}/fastplms_workspace/${_STAGE_REL}"
  _EMBED_STAGE_HOST="${SCRATCH}/fastplms_workspace/${_STAGE_REL}/${_OUT_BASE}"
  RUN_OUTPUT="/workspace/${_STAGE_REL}/${_OUT_BASE}"
  if [[ "${OUTPUT,,}" == *.db ]]; then
    _EMBED_STAGE_KIND="db"
    echo "sbatch_embed: staging SQLite OUTPUT on SCRATCH (${_EMBED_STAGE_HOST}) — final destination: ${OUTPUT}" >&2
    if [[ -f "${_EMBED_STAGE_HOST}" ]]; then
      if [[ -f "$OUTPUT" ]] && [[ "$OUTPUT" -nt "${_EMBED_STAGE_HOST}" ]]; then
        echo "sbatch_embed: output DB is newer than staged DB; refreshing stage from OUTPUT." >&2
        cp -f "$OUTPUT" "${_EMBED_STAGE_HOST}"
        [[ -f "${OUTPUT}-wal" ]] && cp -f "${OUTPUT}-wal" "${_EMBED_STAGE_HOST}-wal" || rm -f "${_EMBED_STAGE_HOST}-wal" 2>/dev/null || true
        [[ -f "${OUTPUT}-shm" ]] && cp -f "${OUTPUT}-shm" "${_EMBED_STAGE_HOST}-shm" || rm -f "${_EMBED_STAGE_HOST}-shm" 2>/dev/null || true
      else
        echo "sbatch_embed: using existing staged DB for resume." >&2
      fi
    elif [[ -f "$OUTPUT" ]]; then
      echo "sbatch_embed: seeding stage from existing OUTPUT DB." >&2
      cp -f "$OUTPUT" "${_EMBED_STAGE_HOST}"
      [[ -f "${OUTPUT}-wal" ]] && cp -f "${OUTPUT}-wal" "${_EMBED_STAGE_HOST}-wal" || true
      [[ -f "${OUTPUT}-shm" ]] && cp -f "${OUTPUT}-shm" "${_EMBED_STAGE_HOST}-shm" || true
    fi
  else
    _EMBED_STAGE_KIND="zarr"
    echo "sbatch_embed: staging Zarr OUTPUT on SCRATCH (${_EMBED_STAGE_HOST}) — final destination: ${OUTPUT}" >&2
    if [[ -d "${_EMBED_STAGE_HOST}" ]]; then
      if [[ -d "$OUTPUT" ]] && [[ "$OUTPUT" -nt "${_EMBED_STAGE_HOST}" ]]; then
        echo "sbatch_embed: output Zarr is newer than staged Zarr; refreshing stage from OUTPUT." >&2
        _sync_tree "$OUTPUT" "${_EMBED_STAGE_HOST}"
      else
        echo "sbatch_embed: using existing staged Zarr for resume." >&2
      fi
    elif [[ -d "$OUTPUT" ]]; then
      echo "sbatch_embed: seeding stage from existing OUTPUT Zarr." >&2
      _sync_tree "$OUTPUT" "${_EMBED_STAGE_HOST}"
    fi
  fi
fi

# Host Conda often sets CC/CXX to conda's cross-compiler; Singularity inherits the login env.
# Triton JIT then fails compiling launchers (missing x86_64-linux-gnu/python3.12/pyconfig.h).
unset CC CXX CPPFLAGS CFLAGS LDFLAGS

set +e
singularity exec --nv \
  -B "${SCRATCH}/fastplms_workspace:/workspace" \
  -B "${REPO_ROOT}:/workspace/repo" \
  "$SIF" \
  env PYTHONPATH="/app:/workspace/repo" HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" FASTPLMS_FINAL_OUTPUT="${OUTPUT}" \
  python /workspace/repo/drorlab_fastplms/embed.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$RUN_OUTPUT" \
    "$@"
_PY_EXIT=$?
set -e

if [[ "${_EMBED_STAGE_KIND}" == "db" ]] && [[ -f "${_EMBED_STAGE_HOST}" ]]; then
  _sqlite_maybe_checkpoint "${_EMBED_STAGE_HOST}"
  mkdir -p "$(dirname "$OUTPUT")"
  rm -f "${OUTPUT}-wal" "${OUTPUT}-shm" 2>/dev/null || true
  cp -f "${_EMBED_STAGE_HOST}" "$OUTPUT"
  [[ -f "${_EMBED_STAGE_HOST}-wal" ]] && cp -f "${_EMBED_STAGE_HOST}-wal" "${OUTPUT}-wal"
  [[ -f "${_EMBED_STAGE_HOST}-shm" ]] && cp -f "${_EMBED_STAGE_HOST}-shm" "${OUTPUT}-shm"
  echo "sbatch_embed: copied staged DB back to ${OUTPUT}" >&2
fi

if [[ "${_EMBED_STAGE_KIND}" == "zarr" ]] && [[ -d "${_EMBED_STAGE_HOST}" ]]; then
  mkdir -p "$OUTPUT"
  _sync_tree "${_EMBED_STAGE_HOST}" "$OUTPUT"
  echo "sbatch_embed: copied staged Zarr back to ${OUTPUT}" >&2
fi

exit "${_PY_EXIT}"
