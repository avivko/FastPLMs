"""E1 comma-separated multiseq: drorlab normalize_e1_multiseq_string + embed on GPU (cuda:0)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from drorlab_fastplms.e1_context import build_e1_row_strings, normalize_e1_multiseq_string
from fastplms.e1.modeling_e1 import E1BatchPreparer, E1ForMaskedLM
from testing.conftest import MODEL_REGISTRY

_DATA_CSV = Path(__file__).resolve().parent / "examples" / "e1_multiseq_edge_cases.csv"


def test_normalize_e1_multiseq_string_variants() -> None:
    canonical = "MAVL,GCHK,MWY"
    for m in (
        ",MAVL,,GCHK,MWY,",
        "  MAVL  ,  GCHK  ,  MWY  ",
        ",,,MAVL,GCHK,,,MWY,,",
    ):
        assert normalize_e1_multiseq_string(m) == canonical


def test_normalize_e1_multiseq_string_raises_when_empty() -> None:
    with pytest.raises(ValueError, match="no non-empty segments"):
        normalize_e1_multiseq_string(",,,")
    with pytest.raises(ValueError, match="no non-empty segments"):
        normalize_e1_multiseq_string("   ,  ,  ")
    with pytest.raises(ValueError, match="no non-empty segments"):
        normalize_e1_multiseq_string("")


def test_normalize_then_prepare_multiseq_matches_canonical() -> None:
    prep = E1BatchPreparer()
    canonical = "MAVL,GCHK,MWY"
    messy = ",MAVL,,GCHK,MWY,"
    ref = prep.prepare_multiseq(canonical)
    got = prep.prepare_multiseq(normalize_e1_multiseq_string(messy))
    assert torch.equal(ref["input_ids"], got["input_ids"])
    assert ref["context_len"] == got["context_len"]


def test_prepare_multiseq_single_segment_no_commas() -> None:
    prep = E1BatchPreparer()
    enc = prep.prepare_multiseq("MKLLVVAA")
    enc_single = prep.prepare_singleseq("MKLLVVAA")
    assert torch.equal(enc["input_ids"], enc_single["input_ids"])


@pytest.mark.gpu
def test_e1_embed_multiseq_edges_cuda0() -> None:
    """Load E1 on cuda:0; normalized CSV strings embed; pooled vectors finite (mirrors embed.py preprocessing)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    os.environ.setdefault("TQDM_DISABLE", "1")

    cfg = MODEL_REGISTRY["e1"]
    model = E1ForMaskedLM.from_pretrained(
        cfg["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()

    canonical = "MAVL,GCHK,MWY"
    messy = ",MAVL,,GCHK,MWY,"
    assert normalize_e1_multiseq_string(messy) == canonical

    multiseq_short = "MK,LK"
    singles = ["MKLLVVMM", "ACDEFGHIKLMNPQRSTVWY"]
    # Same as drorlab embed.py: normalize every E1 string before embed_dataset (dedupe keeps one MAVL,GCHK,MWY).
    sequences = [normalize_e1_multiseq_string(s) for s in [canonical, messy, multiseq_short] + singles]
    sequences = list(dict.fromkeys(sequences))

    with torch.inference_mode():
        embs = model.embed_dataset(
            sequences=sequences,
            tokenizer=None,
            batch_size=2,
            max_len=1024,
            truncate=True,
            full_embeddings=False,
            embed_dtype=torch.float32,
            pooling_types=["mean"],
            save=False,
        )
    assert embs is not None
    for seq in sequences:
        assert seq in embs
        assert torch.isfinite(embs[seq]).all(), f"non-finite embedding for key {seq!r}"

    df = pd.read_csv(_DATA_CSV)
    csv_strings = [
        normalize_e1_multiseq_string(
            build_e1_row_strings(
                combined_col="rag_homolog_context_multiseq",
                context_cols=[],
                query_col="contexted_domain_seq",
                row=row.to_dict(),
            )
        )
        for _, row in df.iterrows()
    ]
    csv_strings = list(dict.fromkeys(csv_strings))

    with torch.inference_mode():
        embs_csv = model.embed_dataset(
            sequences=csv_strings,
            tokenizer=None,
            batch_size=2,
            max_len=1024,
            truncate=True,
            full_embeddings=False,
            embed_dtype=torch.float32,
            pooling_types=["mean"],
            save=False,
        )
    assert embs_csv is not None
    for s in csv_strings:
        assert torch.isfinite(embs_csv[s]).all(), f"non-finite for string {s!r}"

    assert canonical in embs_csv
    torch.testing.assert_close(embs[canonical], embs_csv[canonical], rtol=2e-3, atol=2e-3)

    del model
    torch.cuda.empty_cache()
