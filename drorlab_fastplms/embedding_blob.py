"""Compact SQLite embedding blob decode (torch-only).

Mirrors ``fastplms.embedding_mixin.embedding_blob_to_tensor`` for the formats written by
``tensor_to_embedding_blob`` / ``embed.py`` full-embedding outputs. Keep in sync with
``fastplms/embedding_mixin.py`` if the wire format changes.

Dependencies: Python 3.8+, PyTorch (``torch.frombuffer``).
"""
from __future__ import annotations

import io
import math
import struct
from typing import Optional, Tuple

import torch

_COMPACT_VERSION = 0x01
_CODE_TO_DTYPE = {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}


def embedding_blob_to_tensor(blob: bytes, fallback_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """Deserialize a blob back to a tensor. Auto-detects compact vs legacy formats."""
    if len(blob) >= 6 and blob[0] == _COMPACT_VERSION:
        dtype_code = blob[1]
        if dtype_code not in _CODE_TO_DTYPE:
            raise ValueError(f"Unknown compact dtype_code: {dtype_code}")
        ndim = struct.unpack_from("<i", blob, 2)[0]
        if ndim < 0:
            raise ValueError(f"Invalid ndim in blob: {ndim}")
        shape = struct.unpack_from(f"<{ndim}i", blob, 6)
        data_offset = 6 + 4 * ndim
        target_dtype = _CODE_TO_DTYPE[dtype_code]
        numel = int(math.prod(shape)) if shape else 0
        if dtype_code in (0, 1):
            el_bytes = 2
        else:
            el_bytes = 4
        need = data_offset + numel * el_bytes
        if len(blob) < need:
            raise ValueError(
                f"Blob too short: need {need} bytes for shape {shape} dtype_code {dtype_code}, got {len(blob)}"
            )
        raw = blob[data_offset:need]
        # float16 storage for codes 0 and 1; code 1 is bfloat16 logical dtype after cast
        load_dtype = torch.float16 if dtype_code in (0, 1) else torch.float32
        t = torch.frombuffer(bytearray(raw), dtype=load_dtype).reshape(shape).clone()
        if target_dtype != t.dtype:
            t = t.to(target_dtype)
        return t

    for weights_only in (True, False):
        try:
            buffer = io.BytesIO(blob)
            return torch.load(buffer, map_location="cpu", weights_only=weights_only)
        except Exception:
            continue

    if fallback_shape is None:
        raise ValueError(
            "Cannot deserialize blob: unknown format and no fallback_shape provided."
        )
    t = torch.frombuffer(bytearray(blob), dtype=torch.float32).reshape(fallback_shape).clone()
    return t
