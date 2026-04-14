#!/bin/bash
#SBATCH --job-name=drorlab_smoke
#SBATCH --partition=rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/oak/stanford/groups/rondror/software/plms/slurm_outs/drorlab_smoke_%j.out
#SBATCH --error=/oak/stanford/groups/rondror/software/plms/slurm_outs/drorlab_smoke_%j.err
#
# Drorlab embed.py / finetune.py smoke only (no pytest).
# Covers: ESM2, ESMC (ESM3-class), E1 single-seq, E1 multi-seq — embed + finetune each.
# Submit from the FastPLMs repo root, or set REPO_ROOT.
#
# Optional env: SIF, REPO_ROOT, HF_TOKEN, HUGGING_FACE_HUB_TOKEN, DRORLAB_SMOKE_OUT

set -euo pipefail

SIF="${SIF:-/oak/stanford/groups/rondror/software/plms/singularity/fastplms.sif}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

mkdir -p "${SCRATCH}/fastplms_workspace/drorlab_smoke"
mkdir -p "${SCRATCH}/fastplms_workspace/.cache/huggingface"

singularity exec --nv \
  -B "${SCRATCH}/fastplms_workspace:/workspace" \
  -B "${REPO_ROOT}:/workspace/repo" \
  "$SIF" \
  env PYTHONPATH="/app:/workspace/repo" \
    HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}" \
    USE_TF=0 USE_TORCH=1 \
    HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
    REPO_ROOT="/workspace/repo" \
    DRORLAB_SMOKE_OUT="${DRORLAB_SMOKE_OUT:-/workspace/drorlab_smoke}" \
  bash /workspace/repo/drorlab_fastplms/tests/run_drorlab_smoke.sh
