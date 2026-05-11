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
# Zarr scratch→Oak sync prefers `dsync`/`dcp` from `mpifileutils`; falls back to rsync/cp if unavailable.
#   FASTPLMS_MPI_SYNC_NP=N      mpifileutils ranks for dsync/dcp (default SLURM_CPUS_PER_TASK or 1).
#   FASTPLMS_DSYNC_PROGRESS=N   dsync only: print progress about every N seconds (default off).
#   FASTPLMS_RSYNC_PARALLEL=N   Fallback rsync mode: sync immediate children of the store with up to
#                               N parallel jobs (default 8). Set to 1 for a single full-tree rsync
#                               (also enables FASTPLMS_RSYNC_PROGRESS below).
#   FASTPLMS_RSYNC_PROGRESS=1   Fallback rsync mode: pass --info=progress2 (only when PARALLEL is 1).
#
# Slurm writes stdout -> .out and stderr -> .err. tqdm/warnings go to stderr (.err); embed prints -> .out.
# Batch script milestones print to BOTH streams so either file shows staging + copy/sync. PYTHONUNBUFFERED=1
# keeps Python/tqdm line-buffered into Slurm logs instead of one blob at exit.
#
# Stageable outputs (*.db, *.zarr): by default stage on $SCRATCH for fast local I/O, then sync back
# to final OUTPUT after the job. Stage path is stable per OUTPUT so a new job can resume from the
# latest staged artifact after interrupted runs. Existing staged data is trusted over OUTPUT by
# default so interrupted publishes do not poison resume state. Set FASTPLMS_REFRESH_STAGE_FROM_OUTPUT=1
# to explicitly replace an existing stage from OUTPUT before running. Disable staging entirely with
# FASTPLMS_NO_EMBED_STAGE=1
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

# Print to stdout and stderr so Slurm .out and .err both show wrapper milestones (copy/sync, staging).
_sbatch_embed_msg() {
  printf '%s\n' "$*"
  printf '%s\n' "$*" >&2
}

_maybe_load_mpifileutils() {
  if command -v dsync >/dev/null 2>&1 || command -v dcp >/dev/null 2>&1; then
    return 0
  fi

  if command -v ml >/dev/null 2>&1; then
    ml system mpifileutils >/dev/null 2>&1 || true
  elif command -v module >/dev/null 2>&1; then
    module load system mpifileutils >/dev/null 2>&1 || true
  fi

  command -v dsync >/dev/null 2>&1 || command -v dcp >/dev/null 2>&1
}

_run_mpifileutils() {
  local tool="$1"
  shift

  local np="${FASTPLMS_MPI_SYNC_NP:-${SLURM_CPUS_PER_TASK:-1}}"
  if ! [[ "$np" =~ ^[0-9]+$ ]] || [[ "$np" -lt 1 ]]; then
    np=1
  fi

  if [[ "$np" -gt 1 ]] && command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun --ntasks="$np" --cpus-per-task=1 "$tool" "$@"
  elif [[ "$np" -gt 1 ]] && command -v mpirun >/dev/null 2>&1; then
    mpirun -np "$np" "$tool" "$@"
  else
    "$tool" "$@"
  fi
}

_sqlite_maybe_checkpoint() {
  local db="$1"
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$db" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
  fi
}

_zarr_manifest_path() {
  local zarr_path="${1%/}"
  local zarr_dir
  local zarr_base
  zarr_dir="$(dirname "$zarr_path")"
  zarr_base="$(basename "$zarr_path")"
  printf '%s/%s_manifest.csv\n' "$zarr_dir" "${zarr_base%.zarr}"
}

_sync_zarr_sidecars() {
  local src_zarr="$1"
  local dst_zarr="$2"
  local src_manifest
  local dst_manifest

  src_manifest="$(_zarr_manifest_path "$src_zarr")"
  dst_manifest="$(_zarr_manifest_path "$dst_zarr")"

  if [[ -f "$src_manifest" ]]; then
    mkdir -p "$(dirname "$dst_manifest")"
    cp -f "$src_manifest" "$dst_manifest"
  else
    rm -f "$dst_manifest" 2>/dev/null || true
  fi
}

_sync_tree() {
  local src="$1"
  local dst="$2"
  local src_trim="${src%/}"
  local dst_trim="${dst%/}"
  local dst_parent

  dst_parent="$(dirname "$dst_trim")"
  mkdir -p "$dst_parent"

  if _maybe_load_mpifileutils; then
    if command -v dsync >/dev/null 2>&1; then
      local dsync_opts=( -D )
      if [[ -n "${FASTPLMS_DSYNC_PROGRESS:-}" ]]; then
        dsync_opts+=( --progress "${FASTPLMS_DSYNC_PROGRESS}" )
      fi
      mkdir -p "$dst_trim"
      _sbatch_embed_msg "sbatch_embed: syncing tree with dsync (${src_trim} -> ${dst_trim})"
      _run_mpifileutils dsync "${dsync_opts[@]}" "$src_trim" "$dst_trim"
      return
    fi

    if command -v dcp >/dev/null 2>&1; then
      rm -rf "$dst_trim"
      _sbatch_embed_msg "sbatch_embed: copying tree with dcp (${src_trim} -> ${dst_trim})"
      _run_mpifileutils dcp "$src_trim" "$dst_parent/"
      return
    fi
  elif command -v rsync >/dev/null 2>&1; then
    mkdir -p "$dst_trim"
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
        ' bash "$dst_trim"
    else
      local rsync_opts=( -a --whole-file --delete )
      [[ "${FASTPLMS_RSYNC_PROGRESS:-0}" == "1" ]] && rsync_opts+=( --info=progress2 )
      rsync "${rsync_opts[@]}" "${src_trim}/" "${dst_trim}/"
    fi
  else
    rm -rf "$dst_trim"
    mkdir -p "$dst_trim"
    cp -a "${src_trim}/." "${dst_trim}/"
  fi
}

_NO_STAGE="${FASTPLMS_NO_EMBED_STAGE:-${FASTPLMS_NO_EMBED_DB_STAGE:-0}}"
_REFRESH_STAGE_FROM_OUTPUT="${FASTPLMS_REFRESH_STAGE_FROM_OUTPUT:-0}"
if [[ "$_NO_STAGE" != "1" ]] && [[ "${OUTPUT,,}" == *.db || "${OUTPUT,,}" == *.zarr ]]; then
  _OUT_KEY="$(printf '%s' "$OUTPUT" | sha1sum | awk '{print $1}')"
  _OUT_BASE="$(basename "$OUTPUT")"
  _STAGE_REL="embed_stage/by_output/${_OUT_KEY}"
  mkdir -p "${SCRATCH}/fastplms_workspace/${_STAGE_REL}"
  _EMBED_STAGE_HOST="${SCRATCH}/fastplms_workspace/${_STAGE_REL}/${_OUT_BASE}"
  RUN_OUTPUT="/workspace/${_STAGE_REL}/${_OUT_BASE}"
  if [[ "${OUTPUT,,}" == *.db ]]; then
    _EMBED_STAGE_KIND="db"
    _sbatch_embed_msg "sbatch_embed: staging SQLite OUTPUT on SCRATCH (${_EMBED_STAGE_HOST}) — final destination: ${OUTPUT}"
    if [[ -f "${_EMBED_STAGE_HOST}" ]]; then
      if [[ "$_REFRESH_STAGE_FROM_OUTPUT" == "1" ]] && [[ -f "$OUTPUT" ]]; then
        _sbatch_embed_msg "sbatch_embed: FASTPLMS_REFRESH_STAGE_FROM_OUTPUT=1, refreshing staged DB from OUTPUT."
        cp -f "$OUTPUT" "${_EMBED_STAGE_HOST}"
        [[ -f "${OUTPUT}-wal" ]] && cp -f "${OUTPUT}-wal" "${_EMBED_STAGE_HOST}-wal" || rm -f "${_EMBED_STAGE_HOST}-wal" 2>/dev/null || true
        [[ -f "${OUTPUT}-shm" ]] && cp -f "${OUTPUT}-shm" "${_EMBED_STAGE_HOST}-shm" || rm -f "${_EMBED_STAGE_HOST}-shm" 2>/dev/null || true
      else
        _sbatch_embed_msg "sbatch_embed: using existing staged DB for resume; ignoring OUTPUT copy because scratch stage already exists (set FASTPLMS_REFRESH_STAGE_FROM_OUTPUT=1 to override)."
      fi
    elif [[ -f "$OUTPUT" ]]; then
      _sbatch_embed_msg "sbatch_embed: seeding stage from existing OUTPUT DB."
      cp -f "$OUTPUT" "${_EMBED_STAGE_HOST}"
      [[ -f "${OUTPUT}-wal" ]] && cp -f "${OUTPUT}-wal" "${_EMBED_STAGE_HOST}-wal" || true
      [[ -f "${OUTPUT}-shm" ]] && cp -f "${OUTPUT}-shm" "${_EMBED_STAGE_HOST}-shm" || true
    fi
  else
    _EMBED_STAGE_KIND="zarr"
    _sbatch_embed_msg "sbatch_embed: staging Zarr OUTPUT on SCRATCH (${_EMBED_STAGE_HOST}) — final destination: ${OUTPUT}"
    if [[ -d "${_EMBED_STAGE_HOST}" ]]; then
      if [[ "$_REFRESH_STAGE_FROM_OUTPUT" == "1" ]] && [[ -d "$OUTPUT" ]]; then
        _sbatch_embed_msg "sbatch_embed: FASTPLMS_REFRESH_STAGE_FROM_OUTPUT=1, refreshing staged Zarr from OUTPUT."
        _sync_tree "$OUTPUT" "${_EMBED_STAGE_HOST}"
        _sync_zarr_sidecars "$OUTPUT" "${_EMBED_STAGE_HOST}"
      else
        _sbatch_embed_msg "sbatch_embed: using existing staged Zarr for resume; ignoring OUTPUT copy because scratch stage already exists (set FASTPLMS_REFRESH_STAGE_FROM_OUTPUT=1 to override)."
      fi
    elif [[ -d "$OUTPUT" ]]; then
      _sbatch_embed_msg "sbatch_embed: seeding stage from existing OUTPUT Zarr."
      _sync_tree "$OUTPUT" "${_EMBED_STAGE_HOST}"
      _sync_zarr_sidecars "$OUTPUT" "${_EMBED_STAGE_HOST}"
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
  env PYTHONUNBUFFERED=1 PYTHONPATH="/app:/workspace/repo" HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" FASTPLMS_FINAL_OUTPUT="${OUTPUT}" \
  python /workspace/repo/drorlab_fastplms/embed.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$RUN_OUTPUT" \
    "$@"
_PY_EXIT=$?
set -e

if [[ "${_EMBED_STAGE_KIND}" == "db" ]] && [[ -f "${_EMBED_STAGE_HOST}" ]]; then
  _sbatch_embed_msg "sbatch_embed: copying staged SQLite -> ${OUTPUT}"
  _sqlite_maybe_checkpoint "${_EMBED_STAGE_HOST}"
  mkdir -p "$(dirname "$OUTPUT")"
  rm -f "${OUTPUT}-wal" "${OUTPUT}-shm" 2>/dev/null || true
  _t0=$(date +%s)
  cp -f "${_EMBED_STAGE_HOST}" "$OUTPUT"
  [[ -f "${_EMBED_STAGE_HOST}-wal" ]] && cp -f "${_EMBED_STAGE_HOST}-wal" "${OUTPUT}-wal"
  [[ -f "${_EMBED_STAGE_HOST}-shm" ]] && cp -f "${_EMBED_STAGE_HOST}-shm" "${OUTPUT}-shm"
  _sbatch_embed_msg "sbatch_embed: SQLite copy finished in $(($(date +%s) - _t0))s -> ${OUTPUT}"
fi

if [[ "${_EMBED_STAGE_KIND}" == "zarr" ]] && [[ -d "${_EMBED_STAGE_HOST}" ]]; then
  mkdir -p "$OUTPUT"
  _sbatch_embed_msg "sbatch_embed: syncing staged Zarr -> ${OUTPUT} (no per-file progress here; large stores may take many minutes)"
  _t0=$(date +%s)
  _sync_tree "${_EMBED_STAGE_HOST}" "$OUTPUT"
  _sync_zarr_sidecars "${_EMBED_STAGE_HOST}" "$OUTPUT"
  _sbatch_embed_msg "sbatch_embed: Zarr sync finished in $(($(date +%s) - _t0))s -> ${OUTPUT}"
fi

exit "${_PY_EXIT}"
