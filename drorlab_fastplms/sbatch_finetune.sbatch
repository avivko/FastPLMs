#!/bin/bash
#SBATCH --job-name=fp_ft
#SBATCH --partition=owners,rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/oak/stanford/groups/rondror/software/plms/slurm_outs/fp_ft_%j.out
#SBATCH --error=/oak/stanford/groups/rondror/software/plms/slurm_outs/fp_ft_%j.err
#
# Usage:
#   sbatch drorlab_fastplms/sbatch_finetune.sbatch <MODEL> <TRAIN_CSV> <OUTPUT_DIR> <TASK> [extra finetune.py args...]
#
# TASK is regression or classification. For classification, pass e.g. --num-labels 2 in the extra args.
# Include --seq-col (and optional --label-col, --val-csv, etc.) in the extra args.
#
# Example (regression):
#   sbatch drorlab_fastplms/sbatch_finetune.sbatch \
#     Synthyra/ESM2-150M /workspace/data/train.csv /workspace/runs/ft1 regression \
#     --seq-col sequence --epochs 5
#
# Example (classification):
#   sbatch drorlab_fastplms/sbatch_finetune.sbatch \
#     Synthyra/ESM2-150M /workspace/data/train.csv /workspace/runs/ft2 classification \
#     --seq-col sequence --num-labels 2
#
# Optional env: SIF, REPO_ROOT, HF_TOKEN, HUGGING_FACE_HUB_TOKEN

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "sbatch_finetune: need at least 4 arguments: MODEL TRAIN_CSV OUTPUT_DIR TASK [finetune.py args...]" >&2
  echo "Example: sbatch $0 Synthyra/ESM2-150M /workspace/train.csv /workspace/ft regression --seq-col sequence" >&2
  exit 1
fi

MODEL=$1
TRAIN_CSV=$2
OUTPUT_DIR=$3
TASK=$4
shift 4

SIF="${SIF:-/oak/stanford/groups/rondror/software/plms/singularity/fastplms.sif}"
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"

mkdir -p "${SCRATCH}/fastplms_workspace"
mkdir -p "${OUTPUT_DIR}"

singularity exec --nv \
  -B "${SCRATCH}/fastplms_workspace:/workspace" \
  -B "${REPO_ROOT}:/workspace/repo" \
  "$SIF" \
  env PYTHONPATH="/app:/workspace/repo" USE_TF=0 USE_TORCH=1 HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}" \
  python /workspace/repo/drorlab_fastplms/finetune.py \
    --model "$MODEL" \
    --train-csv "$TRAIN_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --task "$TASK" \
    "$@"
