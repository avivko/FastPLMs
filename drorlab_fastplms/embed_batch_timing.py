"""Per-batch embedding timing via instance patch (drorlab only; keeps fastplms untouched).

``EmbeddingMixin.embed_dataset`` calls ``compiled_model._embed(...)`` once per batch.
Wrapping ``model._embed`` yields:
  - ``forward_s``: wall time around _embed with optional CUDA synchronize (approx. GPU batch work)
  - ``since_prev_forward_s``: wall time since the previous _embed returned (pooling, blob/SQL, Python, etc.)

When ``padding='longest'``, ``maybe_compile`` skips torch.compile, so patching ``model._embed`` matches
the path used inside ``embed_dataset``. If the model is compiled, the wrapped method may not run; use
``padding='max_length'`` only if you need timing and see missing lines.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from typing import Any, Callable, List, Optional

import torch


def _p95(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    ys = sorted(vals)
    i = max(0, min(len(ys) - 1, math.ceil(0.95 * len(ys)) - 1))
    return ys[i]


def patch_embed_timing(
    model: torch.nn.Module,
    *,
    cuda_sync: bool = True,
    log_every: int = 50,
    file=sys.stdout,
) -> Callable[[], None]:
    """Replace ``model._embed`` with a timed wrapper; return a no-arg restore callable."""
    if not hasattr(model, "_embed"):
        print(
            "[timing-batches] model has no _embed; skipping batch-level timing.",
            file=file,
            flush=True,
        )
        return lambda: None

    orig = model._embed
    batch_idx = [0]
    prev_forward_end_t: List[Optional[float]] = [None]
    forward_times: List[float] = []
    gap_times: List[float] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        t_enter = time.perf_counter()
        gap_s = None
        if prev_forward_end_t[0] is not None:
            gap_s = t_enter - prev_forward_end_t[0]
            gap_times.append(gap_s)

        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        try:
            out = orig(*args, **kwargs)
        finally:
            if cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_s = time.perf_counter() - t0
            batch_idx[0] += 1
            forward_times.append(forward_s)
            prev_forward_end_t[0] = time.perf_counter()

            if log_every > 0 and batch_idx[0] % log_every == 0:
                g = gap_s if gap_s is not None else float("nan")
                print(
                    f"[timing-batches] batch={batch_idx[0]} "
                    f"since_prev_forward_s={g:.4f} forward_s={forward_s:.4f}",
                    file=file,
                    flush=True,
                )

        return out

    model._embed = wrapped  # type: ignore[assignment]

    def restore() -> None:
        model._embed = orig  # type: ignore[assignment]
        n = len(forward_times)
        if n == 0:
            return
        mean_fw = statistics.mean(forward_times)
        mean_gap = statistics.mean(gap_times) if gap_times else float("nan")
        p95_gap = _p95(gap_times)
        print(
            f"[timing-batches] summary batches={n} "
            f"mean_forward_s={mean_fw:.4f} mean_since_prev_forward_s={mean_gap:.4f} p95_gap_s={p95_gap:.4f}",
            file=file,
            flush=True,
        )

    return restore
