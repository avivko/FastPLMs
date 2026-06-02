"""``embed.py`` CLI: output-mode defaults, validation, and GPU integration."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForMaskedLM

from drorlab_fastplms.cli_common import apply_attn_backend_after_load, model_config_with_attn
from drorlab_fastplms.embed import main as embed_main
from embed_test_utils import (
    DEFAULT_SEQ3,
    ESM2_8M,
    SEQ3_ONLY_CSV,
    assert_store_all_hidden_states_artifacts,
)
from drorlab_fastplms.embedding_loader import load_embeddings, load_per_residue_embs


def _embed_argv(
    output: Path,
    *,
    pooling: str | None = None,
    hidden_state_index: int | None = None,
    store_all_hidden_states: bool = False,
) -> list[str]:
    argv = [
        "--model",
        ESM2_8M,
        "--input",
        str(SEQ3_ONLY_CSV),
        "--seq-col",
        "sequence",
        "--output",
        str(output),
        "--batch-size",
        "4",
        "--dtype",
        "bfloat16",
        "--attn-backend",
        "auto",
        "--no-entrypoint-setup",
    ]
    if pooling is not None:
        argv.extend(["--pooling", pooling])
    if hidden_state_index is not None:
        argv.extend(["--hidden-state-index", str(hidden_state_index)])
    if store_all_hidden_states:
        argv.append("--store-all-hidden-states")
    return argv


@pytest.mark.embed
def test_cli_rejects_store_all_hidden_states_with_pooling() -> None:
    rc = embed_main(
        _embed_argv(
            Path("/tmp/should_not_be_written.db"),
            pooling="mean",
            store_all_hidden_states=True,
        )
    )
    assert rc == 2


@pytest.mark.gpu
@pytest.mark.embed
def test_embed_cli_default_writes_per_residue_full(tmp_path: Path) -> None:
    """Default ``embed.py`` (no ``--pooling``) writes per-token matrices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    os.environ.setdefault("TQDM_DISABLE", "1")
    out_db = tmp_path / "default.db"
    rc = embed_main(_embed_argv(out_db))
    assert rc == 0

    seq = DEFAULT_SEQ3
    emb = load_embeddings(str(out_db))[seq]
    assert emb.ndim == 2
    assert emb.shape[0] == len(seq) + 2
    assert emb.shape[1] > 0
    assert torch.isfinite(emb).all()

    residues = load_per_residue_embs(str(out_db), sequence=seq, family="esm_tokenizer")
    assert residues.shape == (len(seq), emb.shape[1])
    assert torch.allclose(residues, emb[1:-1])


@pytest.mark.gpu
@pytest.mark.embed
def test_embed_cli_pooling_writes_sequence_vector(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    os.environ.setdefault("TQDM_DISABLE", "1")
    out_db = tmp_path / "pooled.db"
    rc = embed_main(_embed_argv(out_db, pooling="mean"))
    assert rc == 0

    emb = load_embeddings(str(out_db))[DEFAULT_SEQ3]
    assert emb.ndim == 1
    assert torch.isfinite(emb).all()


@pytest.mark.gpu
@pytest.mark.embed
def test_embed_cli_store_all_hidden_states_db_pth_zarr(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    os.environ.setdefault("TQDM_DISABLE", "1")
    seq = DEFAULT_SEQ3

    hidden0_db = tmp_path / "hidden0.db"
    assert embed_main(_embed_argv(hidden0_db, hidden_state_index=0)) == 0

    all_db = tmp_path / "all.db"
    all_pth = tmp_path / "all.pth"
    all_zarr = tmp_path / "all.zarr"
    assert embed_main(_embed_argv(all_db, store_all_hidden_states=True)) == 0
    assert embed_main(_embed_argv(all_pth, store_all_hidden_states=True)) == 0
    assert embed_main(_embed_argv(all_zarr, store_all_hidden_states=True)) == 0

    assert_store_all_hidden_states_artifacts(
        seq,
        hidden0_db=str(hidden0_db),
        all_db=str(all_db),
        all_zarr=str(all_zarr),
        all_pth=str(all_pth),
    )


@pytest.mark.gpu
@pytest.mark.embed
def test_embed_dataset_store_all_hidden_states_matches_cli(tmp_path: Path) -> None:
    """``embed_dataset(store_all_hidden_states=True)`` matches default full-embed path."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    os.environ.setdefault("TQDM_DISABLE", "1")
    seq = DEFAULT_SEQ3
    cfg = model_config_with_attn(ESM2_8M, "auto")
    model = AutoModelForMaskedLM.from_pretrained(
        ESM2_8M,
        config=cfg,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    apply_attn_backend_after_load(model, "auto", cfg)
    model.eval().cuda()

    with torch.inference_mode():
        embs = model.embed_dataset(
            sequences=[seq],
            batch_size=4,
            full_embeddings=True,
            store_all_hidden_states=True,
            embed_dtype=torch.float32,
            save=False,
        )
    assert embs is not None
    t = embs[seq]
    assert t.ndim == 3
    assert t.shape[0] > 1
    assert t.shape[1] == len(seq) + 2
    assert torch.isfinite(t).all()

    del model
    torch.cuda.empty_cache()
