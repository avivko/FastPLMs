"""Pytest configuration for ``drorlab_fastplms`` tests."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def pytest_configure(config: object) -> None:
    config.addinivalue_line(
        "markers",
        "embedding_blob: compact blob codec (embedding_blob.py).",
    )
    config.addinivalue_line(
        "markers",
        "embedding_loader: embedding_loader.py SQLite/.pth and residue APIs.",
    )
    config.addinivalue_line("markers", "gpu: requires CUDA GPU (E1 embed smoke tests).")
    config.addinivalue_line(
        "markers",
        "embed: embed.py CLI output modes and hidden-state export (GPU tests need CUDA).",
    )
