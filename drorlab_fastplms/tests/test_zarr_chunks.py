"""Unit tests for Zarr chunk sizing (inode-friendly defaults)."""

from __future__ import annotations

import pytest

from drorlab_fastplms.zarr_export import (
    DEFAULT_ZARR_TARGET_CHUNK_BYTES,
    ZarrChunkConfig,
    zarr_chunk_config_from_target_mb,
)


@pytest.mark.embed
def test_default_residue_chunk_rows_esmc_hidden():
    cfg = ZarrChunkConfig()
    # H=2560, f4 -> 10 KiB/row; 96 MiB target -> ~9830 rows -> power-of-two 16384
    assert cfg.residue_chunk_rows(2560) == 16384


@pytest.mark.embed
def test_default_residue_chunk_rows_ankh3_xl():
    cfg = ZarrChunkConfig()
    assert cfg.residue_chunk_rows(2560) == 16384


@pytest.mark.embed
def test_larger_target_mb_increases_chunk_rows_up_to_cap():
    small = zarr_chunk_config_from_target_mb(96.0)
    large = zarr_chunk_config_from_target_mb(256.0)
    assert small.residue_chunk_rows(2560) == 16384
    assert large.residue_chunk_rows(2560) == 32768


@pytest.mark.embed
def test_small_hidden_uses_min_rows():
    cfg = ZarrChunkConfig(target_chunk_bytes=96 * 1024 * 1024)
    assert cfg.residue_chunk_rows(128) == 65536  # capped at MAX


@pytest.mark.embed
def test_chunk_config_from_target_mb_matches_bytes():
    cfg = zarr_chunk_config_from_target_mb(96.0)
    assert cfg.target_chunk_bytes == DEFAULT_ZARR_TARGET_CHUNK_BYTES


@pytest.mark.embed
def test_invalid_target_mb_raises():
    with pytest.raises(ValueError, match="positive"):
        zarr_chunk_config_from_target_mb(0)
