import hashlib
import os

import pytest

from drorlab_fastplms.embed_stage import (
    STAGE_OUTPUT_METADATA,
    basename_hash_stage_dir_name,
    clear_all_stage_dirs,
    clear_embed_output_at,
    infer_staged_db_candidate,
    legacy_stage_dir_name,
    lookup_stage_hash,
    output_stage_hash,
    output_stage_label,
    prepare_fresh_embed,
    read_stage_metadata,
    resolve_embed_stage,
    stage_dir_name,
    staged_artifact_relpath,
    write_stage_metadata,
)

ANKH3_OUTPUT = (
    "/oak/stanford/groups/deissero/users/kormanav/fastplms_embs/"
    "ankh3_Dataset_Rhodopsins_rhodb_v2_apr29_2026_v2_with_bitbiome_and_eukaryotes_combined_and_clustered_99id_80cov/"
    "embeddings.zarr"
)


def test_stage_dir_name_uses_parent_folder_label() -> None:
    label = output_stage_label(ANKH3_OUTPUT)
    h = hashlib.sha1(ANKH3_OUTPUT.encode()).hexdigest()
    assert label.startswith("ankh3_Dataset_Rhodopsins")
    assert stage_dir_name(ANKH3_OUTPUT) == f"{label}_{h}"


@pytest.mark.parametrize(
    ("output", "expected_label"),
    [
        (
            "/oak/stanford/groups/deissero/users/kormanav/fastplms_embs/esm2_Dataset/embeddings.db",
            "esm2_Dataset",
        ),
        ("/workspace/runs/foo/embeddings.zarr", "foo"),
    ],
)
def test_output_stage_label(output: str, expected_label: str) -> None:
    assert output_stage_label(output) == expected_label
    h = hashlib.sha1(output.encode()).hexdigest()
    assert stage_dir_name(output) == f"{expected_label}_{h}"
    assert legacy_stage_dir_name(output) == h


def test_resolve_prefers_existing_legacy_stage(tmp_path) -> None:
    output = "/oak/groups/me/runs/job/embeddings.db"
    rel_legacy = os.path.join("embed_stage", "by_output", output_stage_hash(output), "embeddings.db")
    legacy_host = tmp_path / rel_legacy
    legacy_host.parent.mkdir(parents=True)
    legacy_host.write_bytes(b"sqlite")

    info = resolve_embed_stage(output, str(tmp_path))
    assert info["scheme"] == "legacy"
    assert info["stage_host"] == str(legacy_host)


def test_resolve_prefers_named_over_basename_hash_when_both_exist(tmp_path) -> None:
    output = "/oak/groups/me/runs/job/embeddings.db"
    rel_named = staged_artifact_relpath(output, scheme="named")
    rel_old_named = staged_artifact_relpath(output, scheme="basename_hash")
    named_host = tmp_path / rel_named
    old_named_host = tmp_path / rel_old_named
    named_host.parent.mkdir(parents=True)
    old_named_host.parent.mkdir(parents=True)
    named_host.write_bytes(b"named")
    old_named_host.write_bytes(b"old")

    info = resolve_embed_stage(output, str(tmp_path))
    assert info["scheme"] == "named"
    assert info["stage_host"] == str(named_host)


def test_resolve_uses_parent_label_naming_when_no_existing_stage(tmp_path) -> None:
    output = "/oak/groups/me/runs/job/embeddings.zarr"
    info = resolve_embed_stage(output, str(tmp_path))
    assert info["scheme"] == "named"
    assert stage_dir_name(output) in str(info["stage_rel_dir"])


def test_write_and_read_stage_metadata(tmp_path) -> None:
    output = ANKH3_OUTPUT
    stage_dir = tmp_path / "embed_stage" / "by_output" / stage_dir_name(output)
    meta_path = write_stage_metadata(str(stage_dir), output)
    assert os.path.basename(meta_path) == STAGE_OUTPUT_METADATA
    assert read_stage_metadata(str(stage_dir)) == output.rstrip("/")


def test_infer_staged_db_candidate_checks_named_then_legacy(tmp_path, monkeypatch) -> None:
    output = "/oak/groups/me/runs/job/embeddings.db"
    monkeypatch.setenv("SCRATCH", str(tmp_path))

    new_rel = staged_artifact_relpath(output, scheme="named")
    new_host = tmp_path / "fastplms_workspace" / new_rel
    new_host.parent.mkdir(parents=True)
    new_host.write_bytes(b"new")

    assert infer_staged_db_candidate(output) == str(new_host)


def test_lookup_stage_hash_finds_all_naming_schemes(tmp_path) -> None:
    base = tmp_path / "embed_stage" / "by_output"
    oak = "/oak/groups/me/job/embeddings.db"

    legacy = base / "abc123" / "embeddings.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"x")
    write_stage_metadata(str(legacy.parent), oak)

    hash_base = base / "def456_embeddings.zarr" / "embeddings.zarr"
    hash_base.parent.mkdir(parents=True)
    hash_base.mkdir()

    basename_hash = base / "embeddings.zarr_789xyz" / "embeddings.zarr"
    basename_hash.parent.mkdir(parents=True)
    basename_hash.mkdir()

    named = base / f"ankh3_Dataset_Rhodopsins_789xyz" / "embeddings.zarr"
    named.parent.mkdir(parents=True)
    named.mkdir()
    write_stage_metadata(str(named.parent), "/oak/groups/me/job/embeddings.zarr")

    assert lookup_stage_hash("abc123", str(tmp_path)) == [
        f"abc123: {oak}/embeddings.db"
    ]
    assert lookup_stage_hash("def456", str(tmp_path)) == ["def456_embeddings.zarr/embeddings.zarr"]
    assert lookup_stage_hash("789xyz", str(tmp_path)) == [
        "ankh3_Dataset_Rhodopsins_789xyz: /oak/groups/me/job/embeddings.zarr/embeddings.zarr",
        "embeddings.zarr_789xyz/embeddings.zarr",
    ]


def test_basename_hash_scheme_still_resolves(tmp_path) -> None:
    output = "/oak/groups/me/runs/job/embeddings.db"
    rel = staged_artifact_relpath(output, scheme="basename_hash")
    host = tmp_path / rel
    host.parent.mkdir(parents=True)
    host.write_bytes(b"old")

    info = resolve_embed_stage(output, str(tmp_path))
    assert info["scheme"] == "basename_hash"
    assert info["stage_host"] == str(host)
    assert basename_hash_stage_dir_name(output) in str(info["stage_rel_dir"])


def test_clear_embed_output_at_removes_db_zarr_and_sidecars(tmp_path) -> None:
    db = tmp_path / "job" / "embeddings.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"db")
    (tmp_path / "job" / "embeddings.db-wal").write_bytes(b"wal")
    (tmp_path / "job" / "embeddings.db-shm").write_bytes(b"shm")

    removed = clear_embed_output_at(str(db))
    assert not db.exists()
    assert len(removed) == 3

    zarr = tmp_path / "job2" / "embeddings.zarr"
    manifest = tmp_path / "job2" / "embeddings_manifest.csv"
    zarr.mkdir(parents=True)
    (zarr / "meta").write_text("x")
    manifest.write_text("row_index,sequence\n")

    removed = clear_embed_output_at(str(zarr))
    assert not zarr.exists()
    assert not manifest.exists()
    assert str(zarr) in removed


def test_clear_all_stage_dirs_removes_every_naming_scheme(tmp_path) -> None:
    output = "/oak/groups/me/runs/job/embeddings.db"
    dirs = [
        tmp_path / staged_artifact_relpath(output, scheme="named"),
        tmp_path / staged_artifact_relpath(output, scheme="basename_hash"),
        tmp_path / staged_artifact_relpath(output, scheme="hash_basename"),
        tmp_path / staged_artifact_relpath(output, scheme="legacy"),
    ]
    for rel in dirs:
        rel.parent.mkdir(parents=True, exist_ok=True)
        rel.write_bytes(b"x")

    removed = clear_all_stage_dirs(output, str(tmp_path))
    assert len(removed) == 4
    assert not any(rel.exists() for rel in dirs)


def test_prepare_fresh_embed_clears_output_and_stages(tmp_path) -> None:
    oak = tmp_path / "oak" / "job" / "embeddings.zarr"
    oak.mkdir(parents=True)
    manifest = tmp_path / "oak" / "job" / "embeddings_manifest.csv"
    manifest.write_text("row_index,sequence\n")
    output = str(oak)

    stage_rel = staged_artifact_relpath(output, scheme="legacy")
    stage_host = tmp_path / "fastplms_workspace" / stage_rel
    stage_host.parent.mkdir(parents=True)
    stage_host.mkdir()

    info = prepare_fresh_embed(output, str(tmp_path / "fastplms_workspace"))
    assert not oak.exists()
    assert not manifest.exists()
    assert not stage_host.exists()
    assert info["scheme"] == "named"
