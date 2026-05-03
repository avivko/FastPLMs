"""Pytest configuration for ``drorlab_fastplms`` tests."""

from __future__ import annotations


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
