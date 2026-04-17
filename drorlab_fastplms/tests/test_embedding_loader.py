"""Pytests for ``embedding_loader`` (SQLite, .pth, residue selection)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
import torch

from drorlab_fastplms.tests.blob_test_utils import e1_full_shape, pack_compact_tensor


@pytest.fixture()
def tmp_sqlite_e1_db():
    """SQLite DB with one E1-shaped full embedding (float32), sequence ``ACDEFG``."""
    seq = "ACDEFG"
    t, h = e1_full_shape(len(seq), hidden=16)
    full = torch.randn(t, h, dtype=torch.float32)
    blob = pack_compact_tensor(full)
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE embeddings (sequence TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
    conn.execute("INSERT INTO embeddings VALUES (?, ?)", (seq, blob))
    conn.commit()
    conn.close()
    yield path
    os.unlink(path)


@pytest.mark.embedding_loader
def test_load_per_residue_single_sequence_e1(tmp_sqlite_e1_db: str):
    from drorlab_fastplms.embedding_loader import load_per_residue_embs

    seq = "ACDEFG"
    out = load_per_residue_embs(tmp_sqlite_e1_db, sequence=seq, family="e1", residue_number_1b=None)
    assert out.shape == (len(seq), 16)
    assert out.ndim == 2


@pytest.mark.embedding_loader
def test_load_per_residue_residue_number_1b(tmp_sqlite_e1_db: str):
    from drorlab_fastplms.embedding_loader import load_per_residue_embs

    seq = "ACDEFG"
    out = load_per_residue_embs(tmp_sqlite_e1_db, sequence=seq, family="e1", residue_number_1b=(2, 4))
    assert out.shape == (3, 16)


@pytest.mark.embedding_loader
def test_parallel_duplicate_sequences_dict_raises(tmp_sqlite_e1_db: str):
    from drorlab_fastplms.embedding_loader import load_per_residue_embs

    seq = "ACDEFG"
    with pytest.raises(ValueError, match="Duplicate entries"):
        load_per_residue_embs(
            tmp_sqlite_e1_db,
            sequences=[seq, seq],
            residue_number_1b=[1, (2, 3)],
            family="e1",
            batch_size=None,
        )


@pytest.mark.embedding_loader
def test_parallel_two_unique_sequences(tmp_sqlite_e1_db: str):
    from drorlab_fastplms.embedding_loader import load_per_residue_embs

    seq = "ACDEFG"
    seq2 = "MKT"
    t2, h2 = e1_full_shape(len(seq2), hidden=16)
    conn = sqlite3.connect(tmp_sqlite_e1_db)
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?)",
        (seq2, pack_compact_tensor(torch.randn(t2, h2, dtype=torch.float32))),
    )
    conn.commit()
    conn.close()

    out = load_per_residue_embs(
        tmp_sqlite_e1_db,
        sequences=[seq, seq2],
        residue_number_1b=[1, (1, 2)],
        family="e1",
        batch_size=None,
    )
    assert set(out.keys()) == {seq, seq2}
    assert out[seq].shape == (16,)
    assert out[seq2].shape == (2, 16)


@pytest.mark.embedding_loader
def test_embedding_db_reader_iter(tmp_sqlite_e1_db: str):
    from drorlab_fastplms.embedding_loader import EmbeddingDBReader

    with EmbeddingDBReader(tmp_sqlite_e1_db) as db:
        assert db.count() == 1
        keys = list(db.iter_all_full_embeddings(batch_size=1))
        assert len(keys) == 1
        s, emb = keys[0]
        assert s == "ACDEFG"
        assert emb.shape[0] == len("ACDEFG") + 4


@pytest.mark.embedding_loader
def test_load_embeddings_pth_roundtrip(tmp_path: Path):
    from drorlab_fastplms.embedding_loader import load_embeddings_pth, load_per_residue_embs

    seq = "AC"
    d = {seq: torch.randn(len(seq) + 2, 8, dtype=torch.float32)}
    p = tmp_path / "t.pth"
    torch.save(d, p)
    loaded = load_embeddings_pth(str(p))
    assert loaded[seq].shape == d[seq].shape
    out = load_per_residue_embs(str(p), sequence=seq, family="esm_tokenizer", residue_number_1b=None)
    assert out.shape == (len(seq), 8)
