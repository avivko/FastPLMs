"""Build E1 multi-sequence strings: comma-separated segments; last segment is the query."""

from __future__ import annotations

from typing import List, Sequence


def normalize_e1_multiseq_string(s: str) -> str:
    """
    Strip each comma-separated segment and drop empties (CSV ``,,`` / stray commas).

    E1 ``prepare_multiseq`` splits on commas only; callers should run this on any
    user-provided multiseq string before the model sees it.
    """
    parts = [p.strip() for p in (s or "").split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(
            "E1 multi-sequence string has no non-empty segments after comma split "
            f"(e.g. stray commas only). Original: {s!r}"
        )
    return ",".join(parts)


def join_e1_multiseq(context_segments: Sequence[str], query: str) -> str:
    """Concatenate context sequence(s) and query for E1 ``prepare_multiseq`` (comma-separated)."""
    parts: List[str] = []
    for seg in context_segments:
        s = (seg or "").strip()
        if s:
            parts.append(s)
    q = (query or "").strip()
    if not q:
        raise ValueError("E1 multi-sequence row requires a non-empty query sequence.")
    parts.append(q)
    return ",".join(parts)


def build_e1_row_strings(
    *,
    combined_col: str | None,
    context_cols: Sequence[str] | None,
    query_col: str,
    row: dict,
) -> str:
    """
    Build one E1 input string from a CSV row dict (column name -> value).

    If ``combined_col`` is set, that column's value is used as the full multi-seq string (may already contain commas).

    Otherwise ``context_cols`` (in order) + ``query_col`` are joined with commas.
    """
    if combined_col is not None:
        raw = row.get(combined_col)
        if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
            raise ValueError(f"Missing or empty combined column {combined_col!r}")
        s = str(raw).strip()
        if not s:
            raise ValueError(f"Empty combined column {combined_col!r}")
        return s

    q = row.get(query_col)
    if q is None or (isinstance(q, float) and str(q) == "nan"):
        raise ValueError(f"Missing query column {query_col!r}")
    query = str(q).strip()
    ctx: List[str] = []
    if context_cols:
        for c in context_cols:
            v = row.get(c)
            if v is None or (isinstance(v, float) and str(v) == "nan"):
                continue
            t = str(v).strip()
            if t:
                ctx.append(t)
    return join_e1_multiseq(ctx, query)
