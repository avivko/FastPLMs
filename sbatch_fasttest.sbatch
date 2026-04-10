#!/bin/bash
#SBATCH --job-name=fastplms_test
#SBATCH --partition=rondror
#SBATCH --gres=gpu:1
#SBATCH --constraint=GPU_GEN:AMP|GPU_GEN:HPR
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/oak/stanford/groups/rondror/software/plms/slurm_outs/fastplms_%j.out
#SBATCH --error=/oak/stanford/groups/rondror/software/plms/slurm_outs/fastplms_%j.err

set -euo pipefail

SIF=//oak/stanford/groups/rondror/software/plms/singularity/fastplms.sif
mkdir -p "$SCRATCH/fastplms_workspace"

singularity exec --nv \
  -B "$SCRATCH/fastplms_workspace:/workspace" \
  "$SIF" \
  python -m pytest /app/testing/ \
    -m "gpu and not slow and not large and not structure" -v