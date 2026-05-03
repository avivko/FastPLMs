#!/bin/bash
#SBATCH --job-name=fp_embed
#SBATCH --partition=owners,rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
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

# Host Conda often sets CC/CXX to conda's cross-compiler; Singularity inherits the login env.
# Triton JIT then fails compiling launchers (missing x86_64-linux-gnu/python3.12/pyconfig.h).
unset CC CXX CPPFLAGS CFLAGS LDFLAGS

singularity exec --nv \
  -B "${SCRATCH}/fastplms_workspace:/workspace" \
  -B "${REPO_ROOT}:/workspace/repo" \
  "$SIF" \
  env PYTHONPATH="/app:/workspace/repo" HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
  python /workspace/repo/drorlab_fastplms/embed.py \
    --model "$MODEL" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    "$@"
