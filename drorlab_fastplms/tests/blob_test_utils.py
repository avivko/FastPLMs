"""Build compact embedding blobs for tests (torch + struct only; matches wire format)."""

from __future__ import annotations

import struct
from typing import Tuple

import torch

_COMPACT_VERSION = 0x01


def pack_compact_tensor(t: torch.Tensor) -> bytes:
    """Serialize tensor to compact **float32** blob (same header layout as ``fastplms.embedding_mixin``)."""
    t = t.cpu().contiguous()
    if t.dtype != torch.float32:
        raise TypeError("blob_test_utils.pack_compact_tensor only supports float32 in tests")
    dtype_code = 2
    shape = tuple(t.shape)
    ndim = len(shape)
    hdr = struct.pack("<BBi" + "i" * ndim, _COMPACT_VERSION, dtype_code, ndim, *shape)
    flat = t.reshape(-1).tolist()
    body = struct.pack("<" + "f" * len(flat), *flat)
    return hdr + body


def e1_full_shape(seq_len: int, hidden: int = 8) -> Tuple[int, int]:
    """Shape (T, H) for E1 full-embedding layout (T = L+4)."""
    return (seq_len + 4, hidden)
