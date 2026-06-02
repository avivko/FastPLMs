"""Tests for embed.py output mode resolution (full vs pooled)."""

from __future__ import annotations

import pytest

from drorlab_fastplms.embed import resolve_embed_output_mode


pytestmark = pytest.mark.embed


def test_default_is_full_embeddings():
    full, pooling, err = resolve_embed_output_mode(None, False)
    assert err is None
    assert full is True
    assert pooling == ["mean"]


def test_pooling_disables_full_embeddings():
    full, pooling, err = resolve_embed_output_mode("mean,max", False)
    assert err is None
    assert full is False
    assert pooling == ["mean", "max"]


def test_store_all_hidden_states_rejects_pooling():
    full, _, err = resolve_embed_output_mode("mean", True)
    assert full is False
    assert err is not None
    assert "pooling" in err.lower()


def test_store_all_hidden_states_allowed_without_pooling():
    full, pooling, err = resolve_embed_output_mode(None, True)
    assert err is None
    assert full is True
    assert pooling == ["mean"]


@pytest.mark.parametrize("pooling", ["", "  "])
def test_empty_pooling_string_treated_as_default_full(pooling: str):
    full, _, err = resolve_embed_output_mode(pooling, False)
    assert err is None
    assert full is True
