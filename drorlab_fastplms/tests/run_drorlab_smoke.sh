#!/usr/bin/env bash
# Drorlab CLI smoke: embed + finetune for ESM2, ESMC (ESM3-class), E1 single, E1 multi.
# Run inside Docker/Singularity with PYTHONPATH=/app:${REPO_ROOT} and repo at REPO_ROOT (default /workspace/repo).

set -euo pipefail

REPO="${REPO_ROOT:-/workspace/repo}"
EX="$REPO/drorlab_fastplms/tests/examples"
OUT="${DRORLAB_SMOKE_OUT:-/workspace/drorlab_smoke}"
EMBED="$REPO/drorlab_fastplms/embed.py"
FT="$REPO/drorlab_fastplms/finetune.py"

ESM2="Synthyra/ESM2-8M"
# EvolutionaryScale ESMC checkpoint (open-weights ESM3-class sequence model in FastPLMs).
ESMC="Synthyra/ESMplusplus_small"
E1="Synthyra/Profluent-E1-150M"

ATTN=(--attn-backend auto)
BS_EMB=(--batch-size 2)
BS_FT=(--epochs 1 --batch-size 4 --no-lora)

mkdir -p "$OUT"

echo "=== Drorlab smoke: embed ESM2 (CSV) ==="
python "$EMBED" --model "$ESM2" --input "$EX/sequences_example.csv" \
  --output "$OUT/embed_esm2_csv.pth" --seq-col sequence "${ATTN[@]}" "${BS_EMB[@]}"

echo "=== Drorlab smoke: embed ESMC / ESM3-class (CSV) ==="
python "$EMBED" --model "$ESMC" --input "$EX/sequences_example.csv" \
  --output "$OUT/embed_esmc_csv.pth" --seq-col sequence "${ATTN[@]}" "${BS_EMB[@]}"

echo "=== Drorlab smoke: embed E1 single (CSV) ==="
python "$EMBED" --model "$E1" --input "$EX/e1_single_sequence_example.csv" \
  --output "$OUT/embed_e1_single.pth" --seq-col sequence "${ATTN[@]}" "${BS_EMB[@]}"

echo "=== Drorlab smoke: embed E1 multi-sequence (CSV) ==="
python "$EMBED" --model "$E1" --input "$EX/e1_multiseq_example.csv" \
  --output "$OUT/embed_e1_multi.pth" --seq-col sequence \
  --e1-context-cols context_a "${ATTN[@]}" "${BS_EMB[@]}"

echo "=== Drorlab smoke: finetune ESM2 (regression) ==="
python "$FT" --model "$ESM2" --train-csv "$EX/finetune_regression_example.csv" \
  --seq-col sequence --task regression --output-dir "$OUT/ft_esm2" \
  "${ATTN[@]}" "${BS_FT[@]}"

echo "=== Drorlab smoke: finetune ESMC / ESM3-class (regression) ==="
python "$FT" --model "$ESMC" --train-csv "$EX/finetune_regression_example.csv" \
  --seq-col sequence --task regression --output-dir "$OUT/ft_esmc" \
  "${ATTN[@]}" "${BS_FT[@]}"

echo "=== Drorlab smoke: finetune E1 single (regression) ==="
python "$FT" --model "$E1" --train-csv "$EX/e1_single_sequence_example.csv" \
  --seq-col sequence --task regression --output-dir "$OUT/ft_e1_single" \
  "${ATTN[@]}" "${BS_FT[@]}"

echo "=== Drorlab smoke: finetune E1 multi-sequence (regression) ==="
python "$FT" --model "$E1" --train-csv "$EX/e1_multiseq_finetune_example.csv" \
  --seq-col sequence --task regression --output-dir "$OUT/ft_e1_multi" \
  --e1-context-cols context_a "${ATTN[@]}" "${BS_FT[@]}"

echo "Drorlab smoke tests finished OK."
