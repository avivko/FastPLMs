"""Shared CLI helpers for Drorlab FastPLMs scripts (container: PYTHONPATH=/app)."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import torch
from transformers import AutoConfig, PretrainedConfig


def try_entrypoint_setup() -> None:
    """Optional TF32 / cuDNN / inductor tuning (see ``entrypoint_setup.py``)."""
    try:
        import entrypoint_setup  # noqa: F401
    except ImportError:
        pass


def configure_hf_token() -> None:
    """Pass HF_TOKEN from environment if set (for gated models)."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)


def model_config_with_attn(model_id: str, attn_backend: str, trust_remote_code: bool = True) -> PretrainedConfig:
    """Load config and set ``attn_backend`` when the config supports it (ESM2, ESMC, E1, ANKH)."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if hasattr(config, "attn_backend"):
        config.attn_backend = attn_backend
    return config


def apply_attn_backend_after_load(model: Any, attn_backend: str, config: PretrainedConfig) -> None:
    """DPLM/DPLM2: mutable ``model.attn_backend`` after load."""
    mt = getattr(config, "model_type", "") or ""
    if mt.lower() in ("dplm", "dplm2") and hasattr(model, "attn_backend"):
        model.attn_backend = attn_backend


def is_e1_config(config: PretrainedConfig) -> bool:
    return (getattr(config, "model_type", "") or "").lower() == "e1"


def is_esmc_config(config: PretrainedConfig) -> bool:
    return (getattr(config, "model_type", "") or "").lower() in ("esmplusplus", "esmc")


def resolve_torch_dtype(name: str) -> torch.dtype:
    n = name.lower().strip()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype {name!r}; use bfloat16, float16, or float32")


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
