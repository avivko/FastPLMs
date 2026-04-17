#!/bin/bash
#SBATCH --job-name=fp_pytest_full
#SBATCH --partition=rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/oak/stanford/groups/rondror/software/plms/slurm_outs/fastplms_pytest_full_%j.out
#SBATCH --error=/oak/stanford/groups/rondror/software/plms/slurm_outs/fastplms_pytest_full_%j.err
#
# FastPLMs library pytest — **broader** slice than ``sbatch_pytest_fast.sbatch``.
# Still skips ``large`` and ``structure`` by default; allows ``slow`` tests.
#
# Submit from the FastPLMs repo root, or set REPO_ROOT.
#
# Optional env: SIF, REPO_ROOT, HF_TOKEN, HUGGING_FACE_HUB_TOKEN

set -euo pipefail

SIF="${SIF:-/oak/stanford/groups/rondror/software/plms/singularity/fastplms.sif}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

mkdir -p "${SCRATCH}/fastplms_workspace/.cache/huggingface"

singularity exec --nv \
  -B "${SCRATCH}/fastplms_workspace:/workspace" \
  -B "${REPO_ROOT}:/workspace/repo" \
  "$SIF" \
  bash -c "set -euo pipefail
export PYTHONPATH=/app:/workspace/repo
export HF_HOME=\${HF_HOME:-/workspace/.cache/huggingface}
export XDG_CACHE_HOME=\${XDG_CACHE_HOME:-/workspace/.cache}
export USE_TF=0 USE_TORCH=1
python -m pytest /app/testing/ -m 'gpu and not large and not structure' -v
echo FastPLMs pytest full slice finished OK.
"
