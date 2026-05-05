"""Build E1 multi-sequence strings: comma-separated segments; last segment is the query."""

from __future__ import annotations

from typing import List, Sequence

# Must match ``E1PreTrainedModel.validate_sequence`` / ``prepare_singleseq`` in ``fastplms/e1/modeling_e1.py``.
_E1_MASK = "?"


def _invalid_e1_chars(segment: str, *, mask_token: str = _E1_MASK) -> List[str]:
    """Characters that violate E1 ``prepare_singleseq`` rules for one segment."""
    bad: List[str] = []
    for ch in segment:
        if ch == mask_token:
            continue
        if "A" <= ch <= "Z":
            continue
        if ch not in bad:
            bad.append(ch)
    return bad


def _nonempty_multiseq_parts(s: str) -> List[str]:
    """Split a comma-separated multiseq-like string and drop empty/whitespace-only parts."""
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def e1_segment_valid_for_model(segment: str, *, mask_token: str = _E1_MASK) -> bool:
    """Return True if ``segment`` is valid for E1 ``prepare_singleseq`` (uppercase A–Z and mask only)."""
    if not isinstance(segment, str):
        return False
    core = segment.replace(mask_token, "")
    return bool(core) and core.isalpha() and core.isupper()


def validate_e1_embed_inputs(
    sequences: List[str],
    *,
    row_labels: Sequence[str] | None = None,
) -> None:
    """
    Ensure every comma-separated segment in each string is valid before loading the model.

    Raises ``ValueError`` on the first invalid segment with row context.
    """
    for i, multiseq in enumerate(sequences):
        label = row_labels[i] if row_labels is not None and i < len(row_labels) else str(i)
        parts = multiseq.split(",")
        for j, seg in enumerate(parts):
            if not e1_segment_valid_for_model(seg):
                bad_chars = _invalid_e1_chars(seg)
                if bad_chars:
                    mismatch_msg = "mismatched characters: " + ", ".join(repr(c) for c in bad_chars)
                else:
                    mismatch_msg = "mismatched characters: none (segment is empty or mask-only)"
                raise ValueError(
                    f"E1 invalid sequence (row key/id={label!r}, segment {j + 1}/{len(parts)}): {seg!r}; "
                    f"{mismatch_msg}; "
                    "allowed characters are uppercase A–Z and '?' (mask), and each segment must contain "
                    "at least one amino acid letter."
                )


def prepare_e1_inputs_for_runtime(
    sequences: Sequence[str],
    *,
    truncate: bool,
    max_len: int,
) -> List[str]:
    """
    Prepare E1 strings for runtime exactly as embed/finetune intend.

    When truncating raw multiseq strings by character count, a cut can land on a comma
    and create an empty segment at runtime. To prevent ``prepare_multiseq`` failures,
    we drop empty comma-separated segments after truncation.
    """
    out: List[str] = []
    for s in sequences:
        t = s[:max_len] if truncate else s
        parts = _nonempty_multiseq_parts(t)
        if not parts:
            raise ValueError(
                "After --max-len truncation, an E1 multi-sequence string has no non-empty segments. "
                "Use a larger --max-len or disable truncation."
            )
        out.append(",".join(parts))
    return out


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
    """
    Concatenate context sequence(s) and query for E1 ``prepare_multiseq``.

    If the final non-empty context segment (after comma split) already matches ``query``,
    do not append ``query`` again.
    """
    parts: List[str] = []
    for seg in context_segments:
        parts.extend(_nonempty_multiseq_parts(seg))
    q = (query or "").strip()
    if not q:
        raise ValueError("E1 multi-sequence row requires a non-empty query sequence.")
    if not parts or parts[-1] != q:
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
