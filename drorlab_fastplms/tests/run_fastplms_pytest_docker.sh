#!/usr/bin/env bash
# Run FastPLMs library pytest against /app/testing/ inside the fastplms image (Docker or Singularity).
#
# Usage inside container:
#   bash /workspace/repo/drorlab_fastplms/tests/run_fastplms_pytest_docker.sh [fast|full]
#
#   fast — same marker slice as sbatch_pytest_fast.sbatch
#   full — same marker slice as sbatch_pytest_full.sbatch

set -euo pipefail

SLICE="${1:-fast}"
export PYTHONPATH="${PYTHONPATH:-/app:/workspace/repo}"
export USE_TF="${USE_TF:-0}"
export USE_TORCH="${USE_TORCH:-1}"

case "$SLICE" in
  fast)
    MARK='gpu and not slow and not large and not structure'
    ;;
  full)
    MARK='gpu and not large and not structure'
    ;;
  *)
    echo "run_fastplms_pytest_docker.sh: unknown SLICE=${SLICE}" >&2
    echo "Usage: $0 [fast|full]" >&2
    exit 1
    ;;
esac

python -m pytest /app/testing/ -m "${MARK}" -v
echo "FastPLMs pytest (${SLICE}) finished OK."
