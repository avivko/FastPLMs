"""E1 CSV conventions: rhodb-style combined column, query position, normalization."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from drorlab_fastplms.e1_context import (
    build_e1_row_strings,
    join_e1_multiseq,
    normalize_e1_multiseq_string,
)

_EXAMPLES = Path(__file__).resolve().parent / "examples"
_RHODB_LIKE = _EXAMPLES / "rhodb_minimal_like.csv"


def test_combined_col_uses_only_rag_column_query_col_ignored_for_body() -> None:
    """``--e1-combined-col`` path does not append ``contexted_domain_seq``; pipeline must put query in rag."""
    row = pd.read_csv(_RHODB_LIKE).iloc[0].to_dict()
    a = build_e1_row_strings(
        combined_col="rag_homolog_context_multiseq",
        context_cols=[],
        query_col="contexted_domain_seq",
        row=row,
    )
    row_changed = {**row, "contexted_domain_seq": "ZZZZZZZZZZZZZZZZZZZZZZ"}
    b = build_e1_row_strings(
        combined_col="rag_homolog_context_multiseq",
        context_cols=[],
        query_col="contexted_domain_seq",
        row=row_changed,
    )
    assert a == b
    assert str(row["rag_homolog_context_multiseq"]) == a


def test_rhodb_like_row_query_is_last_comma_segment() -> None:
    """Dataset convention: last homolog segment equals ``contexted_domain_seq`` (E1 query = last segment)."""
    row = pd.read_csv(_RHODB_LIKE).iloc[0].to_dict()
    raw = build_e1_row_strings(
        combined_col="rag_homolog_context_multiseq",
        context_cols=[],
        query_col="contexted_domain_seq",
        row=row,
    )
    norm = normalize_e1_multiseq_string(raw)
    parts = norm.split(",")
    query = str(row["contexted_domain_seq"]).strip()
    assert parts[-1] == query
    assert len(parts) == 3


def test_normalize_messy_commas_rhodb_style() -> None:
    row = {
        "contexted_domain_seq": "MKLL",
        "rag_homolog_context_multiseq": " ,ACDEFG,,GHIKL,  ,MKLL  , ",
    }
    norm = normalize_e1_multiseq_string(
        build_e1_row_strings(
            combined_col="rag_homolog_context_multiseq",
            context_cols=[],
            query_col="contexted_domain_seq",
            row=row,
        )
    )
    assert norm == "ACDEFG,GHIKL,MKLL"


def test_without_combined_join_puts_query_last() -> None:
    row = {
        "contexted_domain_seq": "QUERYONLY",
        "context_a": "CTX1",
        "context_b": "CTX2",
    }
    s = build_e1_row_strings(
        combined_col=None,
        context_cols=["context_a", "context_b"],
        query_col="contexted_domain_seq",
        row=row,
    )
    assert s == join_e1_multiseq(["CTX1", "CTX2"], "QUERYONLY")
    assert s.split(",")[-1] == "QUERYONLY"


def test_pa4_tail_matches_contexted_domain_from_public_header_sample() -> None:
    """Sanity check on pasted Abs_Max_Dataset… row pa4: last rag segment equals ``contexted_domain_seq``."""
    # Truncated from user paste: same tail as contexted_domain_seq column for pa4.
    tail = (
        "MKNQVEKITPLSLWANTPSLLTKLLLSMAILLFPTAVYAAANLQPNDFVGISFWLISMALMASTVFFLWETQCVTAKWKTSLTVSALVTLIAAVHYFYMRDVWVATGETPTVYRYIDWLLTVPLLMIEFYLILRAIGAASAGIFWRLLIGTLVMLIAGFMGEVGYISVTVGFVIGMLGWFYILYEIFLGEAGKAAQQQASDSVKFAYNLMRWIVTVGWAIYPIGYVLGYMMGAVDDASLNLVYNLADVINKIAFGLLIWYAATSESQDA"
    )
    contexted = tail
    # Two short fake homologs + real query tail (commas only as segment boundaries).
    rag = "IGMDKILAADDLVGVS,MRVKTLSKVLAGGLAMSALVPT," + tail
    norm = normalize_e1_multiseq_string(rag)
    assert norm.split(",")[-1] == contexted
    assert norm.endswith("SESQDA")


def test_e1_batch_preparer_last_segment_is_query_mask_region() -> None:
    """``prepare_multiseq``: tokens before last segment are context_len (labels masked there by default)."""
    from fastplms.e1.modeling_e1 import E1BatchPreparer

    prep = E1BatchPreparer()
    s = "ACDEFGHIK,MNPQRSTVW,MKTAYIAKQRQ"
    enc = prep.prepare_multiseq(s)
    # Third segment is shortest query-like tail; context is first two segments' tokens.
    assert enc["context_len"] > 0
    n1 = len(prep.prepare_singleseq("ACDEFGHIK")["input_ids"])
    n2 = len(prep.prepare_singleseq("MNPQRSTVW")["input_ids"])
    assert enc["context_len"] == n1 + n2


@pytest.mark.gpu
def test_rhodb_minimal_like_full_embeddings_shape_and_query_aa_slice() -> None:
    """Mirror ``embed.py`` E1 path: normalize multiseq, ``embed_dataset`` full, rows match tokens; AA slice = len(query)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    from fastplms.e1.modeling_e1 import E1BatchPreparer, E1ForMaskedLM

    from testing.conftest import MODEL_REGISTRY

    os.environ.setdefault("TQDM_DISABLE", "1")

    df = pd.read_csv(_RHODB_LIKE)
    sequences: list[str] = []
    queries: list[str] = []
    for _, row in df.iterrows():
        raw = build_e1_row_strings(
            combined_col="rag_homolog_context_multiseq",
            context_cols=[],
            query_col="contexted_domain_seq",
            row=row.to_dict(),
        )
        sequences.append(normalize_e1_multiseq_string(raw))
        queries.append(str(row["contexted_domain_seq"]).strip())

    cfg = MODEL_REGISTRY["e1"]
    device = torch.device("cuda:0")
    model = E1ForMaskedLM.from_pretrained(
        cfg["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    prep = model.model.prep_tokens

    with torch.inference_mode():
        embs = model.embed_dataset(
            sequences=sequences,
            tokenizer=None,
            batch_size=1,
            max_len=8192,
            truncate=False,
            full_embeddings=True,
            embed_dtype=torch.float32,
            pooling_types=["mean"],
            save=False,
        )
    assert embs is not None

    cpu_prep = E1BatchPreparer()
    for seq, query in zip(sequences, queries):
        assert seq.split(",")[-1] == query
        emb = embs[seq]
        assert torch.isfinite(emb).all(), seq[:80]
        assert emb.ndim == 2
        assert emb.shape[1] == model.config.hidden_size

        batch = prep.get_batch_kwargs([seq], device=device)
        n_tokens = int((batch["sequence_ids"][0] != -1).sum().item())
        assert emb.shape[0] == n_tokens, (
            f"full-embedding rows {emb.shape[0]} != non-padding token count {n_tokens}"
        )

        enc_u = cpu_prep.prepare_multiseq(seq)
        assert n_tokens == len(enc_u["input_ids"])

        parts = seq.split(",")
        n_aa = len(query)
        last_seg_tokens = cpu_prep.prepare_singleseq(parts[-1])["input_ids"]
        n_last = len(last_seg_tokens)
        assert n_last == n_aa + 4
        total_u = len(enc_u["input_ids"])
        start = total_u - n_last
        emb_aa_only = emb[start + 2 : start + 2 + n_aa]
        assert emb_aa_only.shape == (n_aa, model.config.hidden_size)

    del model
    torch.cuda.empty_cache()
