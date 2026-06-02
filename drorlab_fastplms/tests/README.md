# Drorlab tests

Pytests live in this directory. Markers:

| Marker | Files | What |
|--------|-------|------|
| `embedding_blob` | `test_embedding_blob.py` | Compact blob decode vs hand-built bytes; optional parity with `fastplms.embedding_mixin` when importable. |
| `embedding_loader` | `test_embedding_loader.py` | `embedding_loader` SQLite / `.pth` / Zarr / residue selection (uses `blob_test_utils` for float32 blobs). |
| `embed` | `test_embed_output_mode.py`, `test_embed_cli.py` | `embed.py` default per-residue output, `--pooling`, `--store-all-hidden-states` (GPU tests need CUDA). |
| `gpu` | `test_embed_cli.py`, `test_e1_multiseq_embed.py`, … | CUDA required. |

## Local (repo root)

```bash
export PYTHONPATH=/path/to/FastPLMs

# One marker
pytest drorlab_fastplms/tests -m embedding_blob -v
pytest drorlab_fastplms/tests -m embedding_loader -v
pytest drorlab_fastplms/tests -m "embed and not gpu" -v
pytest drorlab_fastplms/tests -m embed -v   # includes GPU embed integration

# Everything under this folder
pytest drorlab_fastplms/tests -v
```

## Docker (`fastplms` image)

Same layout as Slurm: **`fastplms`** has **`fastplms` on `/app`**; mount the repo at **`/workspace/repo`** and set **`PYTHONPATH=/app:/workspace/repo`**. Drorlab tests only need CPU; **`--gpus`** is optional.

From the **FastPLMs repo root** on the host (after `docker build -t fastplms .` if needed):

```bash
# Drorlab tests (embedding_blob | embedding_loader | embed | all; embed GPU needs --gpus)
docker run --rm \
  -v "$(pwd)":/workspace/repo -v /tmp/fp_ws:/workspace \
  -e HF_HOME=/workspace/.cache/huggingface -e XDG_CACHE_HOME=/workspace/.cache \
  -e USE_TF=0 -e USE_TORCH=1 \
  fastplms env PYTHONPATH=/app:/workspace/repo REPO_ROOT=/workspace/repo \
  bash /workspace/repo/drorlab_fastplms/tests/run_drorlab_pytest.sh all

# FastPLMs /app/testing — fast or full marker slice (needs GPU in practice)
docker run --rm --gpus '"device=0"' \
  -v "$(pwd)":/workspace/repo -v /tmp/fp_ws:/workspace \
  -e HF_HOME=/workspace/.cache/huggingface -e XDG_CACHE_HOME=/workspace/.cache \
  -e USE_TF=0 -e USE_TORCH=1 \
  fastplms env PYTHONPATH=/app:/workspace/repo \
  bash /workspace/repo/drorlab_fastplms/tests/run_fastplms_pytest_docker.sh fast
```

Scripts: [`run_drorlab_pytest.sh`](run_drorlab_pytest.sh), [`run_fastplms_pytest_docker.sh`](run_fastplms_pytest_docker.sh).

## Slurm (Singularity)

From the FastPLMs repo root:

| Script | Purpose |
|--------|---------|
| [`sbatch_drorlab_pytest.sbatch`](sbatch_drorlab_pytest.sbatch) | Drorlab tests only; optional arg `embedding_blob` \| `embedding_loader` \| `all` (default). |
| [`sbatch_pytest_fast.sbatch`](sbatch_pytest_fast.sbatch) | FastPLMs `/app/testing/` — **fast** slice (`gpu and not slow and not large and not structure`). |
| [`sbatch_pytest_full.sbatch`](sbatch_pytest_full.sbatch) | FastPLMs `/app/testing/` — **full** slice (`gpu and not large and not structure`; allows `slow`). |

Examples:

```bash
sbatch drorlab_fastplms/tests/sbatch_drorlab_pytest.sbatch embedding_blob
sbatch drorlab_fastplms/tests/sbatch_drorlab_pytest.sbatch embedding_loader
sbatch drorlab_fastplms/tests/sbatch_drorlab_pytest.sbatch all

sbatch drorlab_fastplms/tests/sbatch_pytest_fast.sbatch
sbatch drorlab_fastplms/tests/sbatch_pytest_full.sbatch
```

Smoke (embed + finetune) remains [`sbatch_drorlab_smoke.sbatch`](sbatch_drorlab_smoke.sbatch).
