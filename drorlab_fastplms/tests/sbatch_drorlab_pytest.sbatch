#!/bin/bash
#SBATCH --job-name=fp_drorlab_pytest
#SBATCH --partition=rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/oak/stanford/groups/rondror/software/plms/slurm_outs/drorlab_pytest_%j.out
#SBATCH --error=/oak/stanford/groups/rondror/software/plms/slurm_outs/drorlab_pytest_%j.err
#
# Drorlab pytest only (``drorlab_fastplms/tests``: embedding blob + embedding loader).
# Tests are CPU-only; GPU Slurm settings match other Drorlab jobs for partition compatibility.
#
# Usage (from FastPLMs repo root, or set REPO_ROOT):
#   sbatch drorlab_fastplms/tests/sbatch_drorlab_pytest.sbatch [MODE]
#
# MODE:
#   embedding_blob    — ``pytest -m embedding_blob``
#   embedding_loader  — ``pytest -m embedding_loader``
#   all               — full ``drorlab_fastplms/tests`` (default)
#
# Optional env: SIF, REPO_ROOT

set -euo pipefail

MODE="${1:-all}"
SIF="${SIF:-/oak/stanford/groups/rondror/software/plms/singularity/fastplms.sif}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

mkdir -p "${SCRATCH}/fastplms_workspace/.cache/huggingface"

case "$MODE" in
  embedding_blob)
    ARGS_STR="drorlab_fastplms/tests -m embedding_blob"
    ;;
  embedding_loader)
    ARGS_STR="drorlab_fastplms/tests -m embedding_loader"
    ;;
  all)
    ARGS_STR="drorlab_fastplms/tests"
    ;;
  *)
    echo "sbatch_drorlab_pytest: unknown MODE=${MODE}" >&2
    echo "Usage: sbatch $0 [embedding_blob|embedding_loader|all]" >&2
    exit 1
    ;;
esac

singularity exec --nv \
  -B "${SCRATCH}/fastplms_workspace:/workspace" \
  -B "${REPO_ROOT}:/workspace/repo" \
  "$SIF" \
  bash -c "set -euo pipefail
cd /workspace/repo
export PYTHONPATH=/app:/workspace/repo
export HF_HOME=\${HF_HOME:-/workspace/.cache/huggingface}
export XDG_CACHE_HOME=\${XDG_CACHE_HOME:-/workspace/.cache}
export USE_TF=0 USE_TORCH=1
python -m pytest ${ARGS_STR} -v
echo \"Drorlab pytest (${MODE}) finished OK.\"
"
