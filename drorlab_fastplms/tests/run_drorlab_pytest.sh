#!/usr/bin/env bash
# Run Drorlab pytests (embedding_blob / embedding_loader / all). Intended for Docker or Singularity
# with PYTHONPATH=/app:${REPO_ROOT} and repo at REPO_ROOT (default /workspace/repo).
#
# Usage inside container:
#   bash /workspace/repo/drorlab_fastplms/tests/run_drorlab_pytest.sh [embedding_blob|embedding_loader|all]
#
# See tests/README.md for docker run one-liners.

set -euo pipefail

REPO="${REPO_ROOT:-/workspace/repo}"
cd "$REPO"

MODE="${1:-all}"
case "$MODE" in
  embedding_blob)
    ARGS=(drorlab_fastplms/tests -m embedding_blob)
    ;;
  embedding_loader)
    ARGS=(drorlab_fastplms/tests -m embedding_loader)
    ;;
  all)
    ARGS=(drorlab_fastplms/tests)
    ;;
  *)
    echo "run_drorlab_pytest.sh: unknown MODE=${MODE}" >&2
    echo "Usage: $0 [embedding_blob|embedding_loader|all]" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${PYTHONPATH:-/app:${REPO}}"
export USE_TF="${USE_TF:-0}"
export USE_TORCH="${USE_TORCH:-1}"

python -m pytest "${ARGS[@]}" -v
echo "Drorlab pytest (${MODE}) finished OK."
