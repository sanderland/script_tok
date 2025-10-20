#!/usr/bin/env bash
set -euo pipefail

# Train Unigram with 250k vocab on multilingual corpus
# Uses Python 3.14t free-threaded build with threading

PRETOKENIZERS=(scriptenc_cb scriptenc_gpt4o_cb bytes_gpt4_cb bytes_gpt4o_cb)
CORPUS=CulturaX-subsample-100-bal2
N=256000
N_WORKERS=16

echo "Training Unigram tokenizers on multilingual corpus"
echo "Corpus: $CORPUS"
echo "Vocabulary size: $N"
echo "Workers: $N_WORKERS"
echo "Pretokenizers: ${PRETOKENIZERS[*]}"
echo "===========================================\n"

total_jobs=${#PRETOKENIZERS[@]}
current_job=0

for pretokenizer in "${PRETOKENIZERS[@]}"; do
  current_job=$((current_job + 1))
  
  echo "[Job $current_job/$total_jobs] Training with pretokenizer: $pretokenizer"
  echo "Command: uv run train --model unigram --report -n $N --corpus $CORPUS --pretokenizer $pretokenizer --parallel $N_WORKERS"
  echo "Started at: $(date)"
  echo "-------------------------------------------"
  
  # Run the training command
  if uv run train --model unigram --report \
    -n "$N" \
    --corpus "$CORPUS" \
    --pretokenizer "$pretokenizer" \
    --parallel "$N_WORKERS"; then
    echo "✓ Completed successfully at: $(date)"
  else
    echo "✗ Failed at: $(date)"
    echo "Continuing with next pretokenizer..."
  fi
  
  echo "===========================================\n"
done

echo "Done training Unigram tokenizers on $CORPUS with vocab size $N"
echo "Pretokenizers: ${PRETOKENIZERS[*]}"

