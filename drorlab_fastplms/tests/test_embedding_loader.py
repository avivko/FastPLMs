"""Pytests for ``embedding_loader`` (SQLite, .pth, residue selection)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import csv
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
def test_e1_multiseq_per_residue_auto(tmp_path: Path):
    from drorlab_fastplms.embedding_loader import (
        e1_multiseq_full_token_count,
        load_per_residue_embs,
    )

    # Two short context segments + query (matches rhodb-style key shape).
    key = "ACDEFGHIK,MNPQR,ZZZ"
    parts = [p.strip() for p in key.split(",") if p.strip()]
    t = e1_multiseq_full_token_count(parts)
    h = 8
    full = torch.arange(t * h, dtype=torch.float32).reshape(t, h)
    p = tmp_path / "ms.pth"
    torch.save({key: full}, p)
    out = load_per_residue_embs(str(p), sequence=key, family="auto", residue_number_1b=None)
    assert out.shape == (len("ZZZ"), h)
    # Seg0: 9+4=13 tok; seg1: 5+4=9; query AA rows start at 13+9+2=24 for "ZZZ".
    assert torch.equal(out, full[24:27])


@pytest.mark.embedding_loader
def test_e1_multiseq_explicit_family_and_residue_range(tmp_path: Path):
    from drorlab_fastplms.embedding_loader import e1_multiseq_full_token_count, load_per_residue_embs

    key = "AA,BB,CCCC"
    parts = [p.strip() for p in key.split(",") if p.strip()]
    t = e1_multiseq_full_token_count(parts)
    h = 16
    full = torch.randn(t, h, dtype=torch.float32)
    p = tmp_path / "ms2.pth"
    torch.save({key: full}, p)
    out = load_per_residue_embs(str(p), sequence=key, family="e1_multiseq", residue_number_1b=(2, 3))
    assert out.shape == (2, h)


@pytest.mark.embedding_loader
def test_e1_single_still_works_with_auto(tmp_path: Path):
    from drorlab_fastplms.embedding_loader import load_per_residue_embs

    seq = "MKTAY"
    full = torch.randn(len(seq) + 4, 11, dtype=torch.float32)
    p = tmp_path / "single.pth"
    torch.save({seq: full}, p)
    out = load_per_residue_embs(str(p), sequence=seq, family="auto")
    assert out.shape == (len(seq), 11)


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


def _zarr_or_skip():
    return pytest.importorskip("zarr")


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.mark.embedding_loader
def test_embedding_zarr_reader_full_embeddings(tmp_path: Path):
    zarr = _zarr_or_skip()
    from drorlab_fastplms.embedding_loader import EmbeddingZarrReader

    store = tmp_path / "full.zarr"
    manifest = tmp_path / "full_manifest.csv"

    root = zarr.open_group(str(store), mode="w")
    root.attrs["layout"] = "full_embeddings"
    root.create_array("residues", data=torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4).numpy(), chunks=(6, 4))
    root.create_array("row_start", data=[0, 3], chunks=(2,), dtype="i8")
    root.create_array("row_length", data=[3, 3], chunks=(2,), dtype="i4")
    _write_manifest(
        manifest,
        [
            {"row_index": 0, "sequence": "AAA", "id": "id0", "residue_start": 0, "residue_length": 3},
            {"row_index": 1, "sequence": "BBB", "id": "id1", "residue_start": 3, "residue_length": 3},
        ],
    )

    with EmbeddingZarrReader(str(store)) as zr:
        assert zr.count() == 2
        assert zr.list_sequences(limit=2) == ["AAA", "BBB"]
        emb = zr.get_full_embedding("BBB")
        assert emb.shape == (3, 4)
        assert torch.equal(emb, torch.arange(12, 24, dtype=torch.float32).reshape(3, 4))


@pytest.mark.embedding_loader
def test_load_per_residue_from_zarr_full(tmp_path: Path):
    zarr = _zarr_or_skip()
    from drorlab_fastplms.embedding_loader import load_per_residue_embs

    # E1 single-sequence full layout: L+4 rows.
    seq = "ACDE"
    full = torch.arange((len(seq) + 4) * 5, dtype=torch.float32).reshape(len(seq) + 4, 5)

    store = tmp_path / "e1full.zarr"
    manifest = tmp_path / "e1full_manifest.csv"
    root = zarr.open_group(str(store), mode="w")
    root.attrs["layout"] = "full_embeddings"
    root.create_array("residues", data=full.numpy(), chunks=(len(full), 5))
    root.create_array("row_start", data=[0], chunks=(1,), dtype="i8")
    root.create_array("row_length", data=[len(full)], chunks=(1,), dtype="i4")
    _write_manifest(
        manifest,
        [{"row_index": 0, "sequence": seq, "id": "id0", "residue_start": 0, "residue_length": len(full)}],
    )

    out = load_per_residue_embs(str(store), sequence=seq, family="e1")
    assert out.shape == (len(seq), 5)
    assert torch.equal(out, full[2:-2])


@pytest.mark.embedding_loader
def test_load_embeddings_dispatch_zarr_pooled(tmp_path: Path):
    zarr = _zarr_or_skip()
    from drorlab_fastplms.embedding_loader import load_embeddings

    store = tmp_path / "pooled.zarr"
    manifest = tmp_path / "pooled_manifest.csv"
    pooled = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=torch.float32)

    root = zarr.open_group(str(store), mode="w")
    root.attrs["layout"] = "pooled_embeddings"
    root.create_array("pooled", data=pooled.numpy(), chunks=(2, 3))
    _write_manifest(
        manifest,
        [
            {"row_index": 0, "sequence": "S1", "id": "a"},
            {"row_index": 1, "sequence": "S2", "id": "b"},
        ],
    )

    loaded = load_embeddings(str(store))
    assert set(loaded.keys()) == {"S1", "S2"}
    assert loaded["S1"].shape == (3,)
    assert torch.equal(loaded["S2"], pooled[1])
