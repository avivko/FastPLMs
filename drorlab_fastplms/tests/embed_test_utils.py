"""Shared helpers for ``embed.py`` pytest and smoke verification."""

from __future__ import annotations

from pathlib import Path

import torch

from drorlab_fastplms.embedding_loader import (
    EmbeddingZarrReader,
    load_embeddings,
    load_per_residue_embs,
)

_TESTS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = _TESTS_DIR / "examples"
SEQ3_ONLY_CSV = EXAMPLES_DIR / "seq3_only.csv"
DEFAULT_SEQ3 = "ACDEFGHIKLMNPQRSTVWY"
ESM2_8M = "Synthyra/ESM2-8M"


def assert_store_all_hidden_states_artifacts(
    seq: str,
    *,
    hidden0_db: str,
    all_db: str,
    all_zarr: str,
    all_pth: str,
) -> None:
    """Check embed outputs for ``--hidden-state-index 0`` and ``--store-all-hidden-states``."""
    h0 = load_embeddings(hidden0_db)[seq]
    assert h0.ndim == 2, f"hidden0 db expected 2D, got {tuple(h0.shape)}"
    assert torch.isfinite(h0).all(), "hidden0 db has non-finite values"

    all_db_t = load_embeddings(all_db)[seq]
    assert all_db_t.ndim == 3, f"all-states db expected 3D, got {tuple(all_db_t.shape)}"
    n_layers, n_tok, hidden = all_db_t.shape
    assert n_tok == h0.shape[0], f"token dim mismatch: all T={n_tok} vs hidden0 T={h0.shape[0]}"
    assert torch.isfinite(all_db_t).all(), "all-states db has non-finite values"
    assert not torch.allclose(all_db_t[-1], all_db_t[0]), "first and last layer should differ"

    all_pth_t = load_embeddings(all_pth)[seq]
    assert all_pth_t.shape == all_db_t.shape, f"pth {all_pth_t.shape} vs db {all_db_t.shape}"

    with EmbeddingZarrReader(all_zarr) as zr:
        assert zr._num_hidden_states == n_layers
        all_z = zr.get_full_embedding(seq)
    assert all_z.shape == (n_layers, n_tok, hidden), f"zarr full shape {tuple(all_z.shape)}"
    assert torch.allclose(all_z, all_db_t, rtol=1e-5, atol=1e-5), "zarr vs db mismatch"

    r0 = load_per_residue_embs(
        all_db, sequence=seq, family="esm_tokenizer", hidden_state_index=0
    )
    assert r0.shape == (len(seq), hidden)
    assert torch.allclose(r0, all_db_t[0, 1:-1])

    r_last = load_per_residue_embs(
        all_db, sequence=seq, family="esm_tokenizer", hidden_state_index=-1
    )
    assert torch.allclose(r_last, all_db_t[-1, 1:-1])
