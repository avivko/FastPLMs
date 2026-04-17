# Drorlab FastPLMs (Sherlock + Singularity)

Stanford Dror-lab helpers to run **FastPLMs** on **Sherlock** with the group **Singularity** image, plus **optional local Docker** on a GPU workstation.

**Upstream:** [README.md](../README.md) · [embedding API](../docs/embedding_api.md) · [finetuning](../docs/finetuning.md) · [CLAUDE.md](../CLAUDE.md)

---

## What you get

| Piece | Purpose |
|--------|---------|
| [`embed.py`](embed.py) | Embed CSV or FASTA → `.pth` or SQLite |
| [`embedding_loader.py`](embedding_loader.py) | Load `.pth` / `.db` and map full rows → per-residue indices (ESM2, ESMC, E1, DPLM, DPLM2, ANKH) |
| [`finetune.py`](finetune.py) | Train sequence classification / regression / MLM from CSV |
| [`tests/sbatch_pytest_fast.sbatch`](tests/sbatch_pytest_fast.sbatch) | Slurm: FastPLMs **pytest** GPU slice only |
| [`tests/sbatch_drorlab_smoke.sbatch`](tests/sbatch_drorlab_smoke.sbatch) | Slurm: **Drorlab** embed + finetune smoke (see below) |
| [`tests/run_drorlab_smoke.sh`](tests/run_drorlab_smoke.sh) | Same smoke steps; run inside Docker/Singularity |
| [`sbatch_embed.sbatch`](sbatch_embed.sbatch) | Slurm wrapper for `embed.py` |
| [`sbatch_finetune.sbatch`](sbatch_finetune.sbatch) | Slurm wrapper for `finetune.py` |

The SIF installs **`fastplms` under `/app`**. Your git checkout is **bind-mounted at `/workspace/repo`** so you can change Drorlab scripts **without rebuilding the SIF**. Keep **`fastplms` from the image**—do **not** mount the repo over `/app` unless you mean to override the library.

---

## Quick start (Sherlock)

**Prerequisites:** GPU job (`#SBATCH --gres=gpu:1` or your group’s rule), `$SCRATCH`, network on compute nodes if models are not cached yet. Optional: `HF_TOKEN` for gated models.

**Paths in the examples below are inside the container.** Put data under the scratch bind (e.g. `$SCRATCH/fastplms_workspace/...` on the host maps to `/workspace/...` in the job).

### FastPLMs pytest (library smoke)

From the **FastPLMs repo root**:

```bash
cd /oak/stanford/groups/rondror/software/plms/FastPLMs
sbatch drorlab_fastplms/tests/sbatch_pytest_fast.sbatch
```

Runs: `pytest /app/testing/ -m "gpu and not slow and not large and not structure"`.

### Drorlab CLI smoke (embed + finetune)

Separate job — exercises **`embed.py`** and **`finetune.py`** with tiny CSVs under [`tests/examples/`](tests/examples/):

| Step | Model (HF id) | What it checks |
|------|----------------|----------------|
| Embed | `Synthyra/ESM2-8M` | ESM2 CSV |
| Embed | `Synthyra/ESMplusplus_small` | **ESMC** (EvolutionaryScale / **ESM3-open-class** weights in FastPLMs) |
| Embed | `Synthyra/Profluent-E1-150M` | E1 single-sequence CSV |
| Embed | `Synthyra/Profluent-E1-150M` | E1 multi-sequence CSV (`--e1-context-cols`) |
| Finetune | Same four checkpoints | One epoch regression each (`--no-lora`) |

```bash
sbatch drorlab_fastplms/tests/sbatch_drorlab_smoke.sbatch
```

Outputs default to `/workspace/drorlab_smoke` in the container (under your `$SCRATCH` bind). Override with env **`DRORLAB_SMOKE_OUT`**.

**Docker (workstation):** from repo root, with GPU and scratch mounted:

```bash
docker run --rm --gpus '"device=0"' \
  -v "$(pwd)":/workspace/repo -v /tmp/fp_ws:/workspace \
  -e HF_HOME=/workspace/.cache/huggingface -e XDG_CACHE_HOME=/workspace/.cache \
  -e USE_TF=0 -e USE_TORCH=1 \
  fastplms env PYTHONPATH=/app:/workspace/repo REPO_ROOT=/workspace/repo \
  bash /workspace/repo/drorlab_fastplms/tests/run_drorlab_smoke.sh
```

Optional: `REPO_ROOT=/path/to/FastPLMs sbatch ...` if you submit from elsewhere.

### Embedding job

Pass **model**, **input**, **output**, then any extra `embed.py` flags (no `export` needed):

```bash
sbatch drorlab_fastplms/sbatch_embed.sbatch \
  Synthyra/ESM2-8M \
  /workspace/data/in.csv \
  /workspace/runs/emb.pth \
  --seq-col my_sequence_column \
  --batch-size 16 --pooling mean,max
```

For **CSV**, pass **`--seq-col`** (there is no default). **FASTA** does not use `--seq-col`. **E1** CSV with only **`--e1-combined-col`** may omit **`--seq-col`**; if you use **`--e1-context-cols`**, **`--seq-col`** names the query column.

### Fine-tuning job

Pass **model**, **train CSV**, **output directory**, **task** (`regression`, `classification`, or `mlm`), then any extra `finetune.py` flags:

```bash
# Regression
sbatch drorlab_fastplms/sbatch_finetune.sbatch \
  Synthyra/ESM2-150M \
  /workspace/data/train.csv \
  /workspace/runs/ft_reg \
  regression \
  --seq-col my_sequence_column \
  --epochs 5 --batch-size 4

# Classification (add --num-labels in the extra args)
sbatch drorlab_fastplms/sbatch_finetune.sbatch \
  Synthyra/ESM2-150M \
  /workspace/data/train.csv \
  /workspace/runs/ft_cls \
  classification \
  --seq-col my_sequence_column \
  --num-labels 2 --epochs 10
```

**`--seq-col` is required** by `finetune.py` (no default). Pass it in the trailing arguments after **`TASK`**.

The Slurm templates **require** the full positional list (**3** tokens for embed, **4** for finetune, then optional CLI flags). If you submit with too few arguments, the batch script **exits with an error** before running Singularity.

---

## Slurm: optional environment

| Variable | When to set |
|----------|-------------|
| `REPO_ROOT` | Checkout path if not submitting from the repo root (default: `SLURM_SUBMIT_DIR`) |
| `SIF` | Override `.sif` path (default: group `fastplms.sif` on OAK) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Gated Hugging Face models |
| `HF_HOME` / `XDG_CACHE_HOME` | Cache dirs (Slurm smoke scripts default these under `/workspace/.cache` when unset; the `fastplms` image sets `HF_HOME` in its Dockerfile) |
| `DRORLAB_SMOKE_OUT` | Drorlab smoke artifacts directory inside the container (default `/workspace/drorlab_smoke`) |

Example:

```bash
REPO_ROOT=/oak/stanford/groups/rondror/users/you/FastPLMs \
  sbatch drorlab_fastplms/sbatch_embed.sbatch \
  Synthyra/ESM2-8M /workspace/in.csv /workspace/out.pth --seq-col sequence
```

---

## `embed.py` (reference)

- **Input:** CSV (**`--seq-col` required** for typical CSV; optional **`--id-col`** → manifest next to output for `.pth`) or FASTA (**no `--seq-col`**). E1 CSV with **`--e1-combined-col`** only may omit **`--seq-col`**.
- **Output:** `.db` → SQLite; anything else → `.pth` keyed by sequence string.
- **Attention:** Default **`--attn-backend auto`** (fastest available in the image, often flash → flex → sdpa). Use **`--attn-backend sdpa`** for strict reproducibility vs SDPA; flash is approximate. ANKH may fall back from flash (T5 RPE). See [main README § Attention Backends](../README.md#attention-backends) and [CLAUDE.md](../CLAUDE.md).
- **E1 multi-sequence:** Comma-separated segments: all but the last = context, last = query ([E1 cookbook](../official/e1/cookbook/)). Use `--e1-combined-col` or `--e1-context-cols ... --seq-col query`. Build MSA / sampling **offline**, then CSV.

**Docker one-liner (workstation):**

```bash
cd /path/to/FastPLMs && docker build -t fastplms .
docker run --rm --gpus '"device=0"' \
  -v "$(pwd)":/workspace/repo -v /tmp/fp_ws:/workspace \
  fastplms env PYTHONPATH=/app:/workspace/repo \
  python /workspace/repo/drorlab_fastplms/embed.py --help
```

Use **`--gpus '"device=0"'`** when only the training GPU should be visible (e.g. A100 vs display GPU).

### SQLite output format and residue indexing

When output ends with `.db`, embeddings are written to SQLite table `embeddings`:

- `sequence` (`TEXT PRIMARY KEY`)
- `embedding` (`BLOB NOT NULL`)

Each `embedding` blob is the **compact** format from `fastplms.embedding_mixin` (same for ESM2, ESMC, E1, DPLM, DPLM2, and ANKH):

- Header: `[version:1][dtype_code:1][ndim:4][shape:4*ndim]`
- Then raw tensor bytes
- `dtype_code`: `0=float16`, `1=bfloat16-stored-as-fp16-bytes`, `2=float32`

Decode with `embedding_blob_to_tensor` from `fastplms.embedding_mixin`, or use [`embedding_loader.py`](embedding_loader.py) below.

**`--full-embeddings` row layout (verified on `Synthyra/ESM2-8M`, `Synthyra/ESMplusplus_small`, `Synthyra/Profluent-E1-150M`, `Synthyra/DPLM-150M`, `Synthyra/DPLM2-150M`, `Synthyra/ANKH_base`, `Synthyra/ANKH2_large`, `Synthyra/ANKH3_large`):**

| Model family | Stored length `T` | Align residues (0-based row `i` → residue #`i+1`) |
|--------------|-------------------|-----------------------------------------------------|
| ESM2, ESMC, DPLM, DPLM2 | `len(sequence) + 2` | `residue_emb = full[1:-1]` (drop BOS/cls + EOS) |
| E1 | `len(sequence) + 4` | `residue_emb = full[2:-2]` (drop `<bos>`, `1`, `2`, `<eos>`) |
| ANKH (base/2/3) | `len(sequence) + 1` | `residue_emb = full[:-1]` (drop trailing special token) |

So after stripping, **`residue_emb[0]` is always residue #1**, **`residue_emb[4]` is residue #5**, etc.

**Loader helper + CLI** (works for `.db` and `.pth` dicts):

```bash
docker run --rm --gpus '"device=0"' \
  -v "$(pwd)":/workspace/repo -v /tmp/fp_ws:/workspace \
  fastplms env PYTHONPATH=/app:/workspace/repo \
  python /workspace/repo/drorlab_fastplms/embedding_loader.py \
  --path /workspace/your.db --sequence YOUR_SEQUENCE --residue-number 10-25
# also supported:
#   --residue-number 5
#   --residue-number 3,8,21
```

In Python:

```python
from drorlab_fastplms.embedding_loader import load_per_residue_embs

# Single unified API controlled by batch_size:

# 1) batch_size=None (default): load selected entries fully in memory (dict)
all_or_selected = load_per_residue_embs(
    "embeddings.db",
    sequences=["ACDEFG..."],      # optional; None means all
    family="auto",
    residue_number_1b=(10, 25),   # optional; int | (start,end) | [list]
    batch_size=None,              # default
)
vec = all_or_selected["ACDEFG..."].mean(dim=0)

# 2) batch_size=1: one-at-a-time iterator (minimal memory)
for seq, emb in load_per_residue_embs(
    "embeddings.db",
    sequences=["ACDEFG...", "MKTW...", "GGHQL..."],  # optional; None => iterate all rows
    family="auto",
    batch_size=1,
):
    do_something = emb.mean(dim=0)

# 3) batch_size>1: batched iterator (fewer queries / better throughput)
for seq, emb in load_per_residue_embs(
    "embeddings.db",
    sequences=["ACDEFG...", "MKTW...", "GGHQL..."],
    family="auto",
    batch_size=2048,
):
    do_something = emb.mean(dim=0)
```

For large SQLite datasets (100k+ sequences), use iterator mode (`batch_size>=1`) to avoid loading everything into RAM.

---

## `finetune.py` (reference)

- **CSV columns:** **`--seq-col`** is **required** (no default). **`--label-col`** defaults to `label` (ignored for **`--task mlm`**). Example CSVs under [`tests/examples/`](tests/examples/) use **~24 rows** so train/val split and multiple steps per epoch give meaningful **`train_loss`** (tiny toy CSVs are not reliable).
- **Tasks:** `regression` (numeric labels, `num_labels=1`) vs `classification` (`--num-labels`) vs **`mlm`** (masked language modeling on `--seq-col` only; tokenizer models only, not E1).
- **Large CSVs:** Optional **`--max-train-rows`** / **`--max-val-rows`** cap rows **after** the train/val split (e.g. smoke tests on multi-hundred-k row tables).
- **MLM:** **`--mlm-probability`** (default `0.15`) sets mask rate; LoRA targets linear layers and keeps **`lm_head`** trainable.
- **Metrics UI:** **`--report-to`** `none` | `wandb` | `tensorboard` | `both` | `all` (HF `TrainingArguments`). Optional **`--run-name`**, **`--eval-strategy`** `steps` | `epoch` (epoch validation gives one **`eval_loss`** point per epoch on the charts). **Weights & Biases:** use **`--report-to wandb`**, set **`WANDB_API_KEY`** (or run **`wandb login`**), optional **`WANDB_PROJECT`** / **`WANDB_ENTITY`**; avoid **`WANDB_MODE=offline`** if you want charts on [wandb.ai](https://wandb.ai). **TensorBoard** (if enabled) writes under **`{output_dir}/tb/`**.
- **ESM++ (ESMC):** Collator adds `sequence_id = attention_mask.bool()` when needed.
- **E1:** Same multi-sequence flags as embedding; collator uses `prep_tokens.get_batch_kwargs`.
- **Validation:** `--val-csv`, or ~10% holdout from train if omitted (tiny CSVs may duplicate train/val—use `--val-csv` for real runs).

**Example (ESM2-8M MLM on RhoDB `combined.csv`, `full_seq` column):** bind-mount the data repo and FastPLMs repo, then:

```bash
docker run --gpus all -v /path/to/FastPLMs:/workspace/repo -v /path/to/deep-rho:/data fastplms \
  env PYTHONPATH=/app:/workspace/repo USE_TF=0 USE_TORCH=1 \
  python /workspace/repo/drorlab_fastplms/finetune.py \
  --model Synthyra/ESM2-8M \
  --train-csv /data/data/RhoDB/v1_with_bitbiome_2step_clustering/combined_and_clustered_99id_80cov/combined.csv \
  --task mlm --seq-col full_seq --output-dir /workspace/esm2-8m_mlm_rhodb \
  --epochs 3 --batch-size 8 --max-length 1024
```

Adjust **`--epochs`**, **`--batch-size`**, and optional **`--max-train-rows`** for full runs on ~400k+ rows.

---

## Bind layout (recommended)

| Bind | Mount as | Role |
|------|-----------|------|
| `$SCRATCH/fastplms_workspace` | `/workspace` | HF cache, outputs, data paths in job |
| **Repo root** | `/workspace/repo` | `drorlab_fastplms/*.py` |

Templates set `PYTHONPATH=/app:/workspace/repo` so `import drorlab_fastplms` works and **`fastplms` stays on `/app`**.

---

## Docker → SIF → Sherlock

1. Develop and test in **Docker** until happy.
2. Build the SIF on a machine with Docker (see repo **Makefile**; submodules required before image build).
3. Copy the `.sif` to OAK and point the `fastplms.sif` symlink at the version you want.

**Rebuild the SIF** when the **image** changes: Dockerfile, CUDA base, `requirements.txt` / PyTorch pins, `official/` pieces baked in, or **`fastplms/`** you rely on **without** mounting the repo over `/app`.

**Skip a rebuild** if you only change files under **`drorlab_fastplms/`** and jobs use the `/workspace/repo` bind.

---

## Troubleshooting

- **`Trainer` / Keras / `tf_keras`:** `finetune.py` sets `USE_TF=0` and `USE_TORCH=1` before importing Transformers; Slurm templates export the same.
- **E1 / sequence head NaNs:** If the new classification head picks up **NaN** weights after `from_pretrained(..., ignore_mismatched_sizes=True)`, `finetune.py` **re-initializes** that head so training loss is well-defined.
- **`sbatch: This does not look like a batch script`:** Shebang must be `#!/bin/bash`.
- **GPU missing in container:** Singularity **`--nv`** on GPU nodes; Docker needs the NVIDIA toolkit.
- **OOM:** Lower `--batch-size` / `--max-length` / model size.
- **`test_backend_consistency` / numeric drift:** Try **`--attn-backend sdpa`**.
- **Singularity / disk:** If pulls fail, set something like `export APPTAINER_TMPDIR=$SCRATCH/tmp`.

---

## Appendix: layout table

| Path | Role |
|------|------|
| `drorlab_fastplms/embed.py` | Embedding CLI |
| `drorlab_fastplms/embedding_loader.py` | Load `.pth` / `.db`, strip specials → per-residue tensors |
| `drorlab_fastplms/finetune.py` | Fine-tuning CLI |
| `drorlab_fastplms/tests/examples/` | Tiny CSV/FASTA fixtures for Drorlab smoke + docs |
| `drorlab_fastplms/tests/run_drorlab_smoke.sh` | Drorlab smoke driver (embed + finetune matrix) |
| `drorlab_fastplms/cli_common.py` | Shared config / dtype / HF token helpers |

---

## Appendix: new Singularity image (CalVer)

On a **non-Sherlock** machine with Docker:

```bash
cd /path/to/FastPLMs
REMOTE_USER=youruser
TAG_DATE=2026.04.0
IMAGE_NAME=fastplms
SIF_REMOTE_DEST=/oak/stanford/groups/rondror/software/plms/singularity
SIF_NAME=${IMAGE_NAME}_${TAG_DATE}.sif

git tag "$TAG_DATE"
make
scp "/tmp/${IMAGE_NAME}/${SIF_NAME}" "${REMOTE_USER}@dtn.sherlock.stanford.edu:${SIF_REMOTE_DEST}"
```

On Sherlock, update the `fastplms.sif` symlink to the new file.
