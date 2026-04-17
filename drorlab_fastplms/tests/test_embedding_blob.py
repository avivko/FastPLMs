"""Pytests for standalone embedding blob decode."""

from __future__ import annotations

import io
import math
import struct

import pytest
import torch


def _pack_compact_float32(shape: tuple[int, ...], flat_floats: list[float]) -> bytes:
    assert len(flat_floats) == int(math.prod(shape))
    ndim = len(shape)
    hdr = struct.pack("<BBi" + "i" * ndim, 0x01, 2, ndim, *shape)
    body = struct.pack("<" + "f" * len(flat_floats), *flat_floats)
    return hdr + body


@pytest.mark.embedding_blob
def test_compact_float32_roundtrip():
    from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

    want = torch.tensor([[0.0, 1.5, -2.25], [3.125, 4.0, 5.5]], dtype=torch.float32)
    flat = want.reshape(-1).tolist()
    blob = _pack_compact_float32(tuple(want.shape), flat)
    got = embedding_blob_to_tensor(blob)
    assert got.dtype == torch.float32
    assert torch.equal(got, want)


@pytest.mark.embedding_blob
def test_compact_float16_roundtrip():
    from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

    vals = [1.0, -2.0, 0.5]
    ndim = 1
    shape = (3,)
    hdr = struct.pack("<BBi" + "i" * ndim, 0x01, 0, ndim, *shape)
    body = struct.pack("<" + "e" * len(vals), *vals)
    got = embedding_blob_to_tensor(hdr + body)
    want = torch.tensor(vals, dtype=torch.float16).reshape(shape)
    assert got.dtype == torch.float16
    assert torch.allclose(got.float(), want.float())


@pytest.mark.embedding_blob
def test_torch_save_fallback():
    from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

    t = torch.randn(2, 5, dtype=torch.float64)
    buf = io.BytesIO()
    torch.save(t, buf)
    try:
        got = embedding_blob_to_tensor(buf.getvalue())
    except ValueError:
        pytest.skip("torch.load fallback not available in this environment (e.g. numpy/pickle stack)")
    assert torch.equal(got, t)


@pytest.mark.embedding_blob
def test_legacy_raw_float32_with_fallback_shape():
    from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

    t = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    blob = struct.pack("<" + "f" * 12, *t.reshape(-1).tolist())
    got = embedding_blob_to_tensor(blob, fallback_shape=(3, 4))
    assert torch.equal(got, t)


@pytest.mark.embedding_blob
def test_bad_dtype_code():
    from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor

    hdr = struct.pack("<BBii", 0x01, 99, 1, 4)
    with pytest.raises(ValueError, match="Unknown compact dtype"):
        embedding_blob_to_tensor(hdr + (b"\x00" * 16))


@pytest.mark.embedding_blob
def test_parity_with_fastplms_mixin_fp16_bf16_fp32():
    pytest.importorskip("fastplms.embedding_mixin")
    from fastplms.embedding_mixin import embedding_blob_to_tensor as mixin_dec
    from fastplms.embedding_mixin import tensor_to_embedding_blob as mixin_enc

    from drorlab_fastplms.embedding_blob import embedding_blob_to_tensor as local_dec

    torch.manual_seed(42)
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.randn(5, 13, dtype=dtype)
        blob = mixin_enc(x)
        a = mixin_dec(blob)
        b = local_dec(blob)
        assert a.dtype == dtype
        assert b.dtype == dtype
        assert torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-2)
