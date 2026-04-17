#!/bin/bash
#SBATCH --job-name=fp_pytest_fast
#SBATCH --partition=rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oak/stanford/groups/rondror/software/plms/slurm_outs/fastplms_pytest_%j.out
#SBATCH --error=/oak/stanford/groups/rondror/software/plms/slurm_outs/fastplms_pytest_%j.err
#
# FastPLMs GPU pytest — **fast** slice (``not slow`` and ``not large``).
# Related: ``sbatch_pytest_full.sbatch`` (broader FastPLMs slice), ``sbatch_drorlab_pytest.sbatch`` (Drorlab-only).
# Submit from the FastPLMs repo root, or set REPO_ROOT to the checkout.
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
  bash -c '
set -euo pipefail
export PYTHONPATH="/app:/workspace/repo"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export USE_TF=0 USE_TORCH=1

python -m pytest /app/testing/ \
  -m "gpu and not slow and not large and not structure" -v

echo "FastPLMs pytest fast slice finished OK."
'
