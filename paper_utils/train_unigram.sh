#!/usr/bin/env bash
set -euo pipefail

PRETOKENIZERS=(scriptenc_cb scriptenc_gpt4o_cb)
corpora=( deu_latn_300mb arb_arab_300mb kor_hang_300mb zho_hans_300mb \
         rus_cyrl_300mb heb_hebr_300mb hin_deva_300mb tha_thai_300mb
         eng_latn_300mb jpn_jpan_300mb vie_latn_300mb pan_guru_300mb )

ns=(64000)

# Count total jobs for progress tracking
total_jobs=$((${#ns[@]} * ${#corpora[@]} * ${#PRETOKENIZERS[@]}))
current_job=0

echo "Starting sequential training of $total_jobs jobs..."
echo "Vocabulary sizes: ${ns[*]}"
echo "Corpora: ${corpora[*]}"
echo "Pretokenizers: ${PRETOKENIZERS[*]}"
echo "===========================================\n"

# Sequential training with progress output
for n in "${ns[@]}"; do
  for corpus in "${corpora[@]}"; do
    for pretokenizer in "${PRETOKENIZERS[@]}"; do
      current_job=$((current_job + 1))
      
      echo "[Job $current_job/$total_jobs] Training: n=$n, corpus=$corpus, pretokenizer=$pretokenizer"
      echo "Command: uv run script_bpe/train.py --model unigram --report -n $n --corpus $corpus --pretokenizer $pretokenizer"
      echo "Started at: $(date)"
      echo "-------------------------------------------"
      
      # Run the training command and capture both stdout and stderr
      if uv run script_bpe/train.py --model unigram --report -n "$n" --corpus "$corpus" --pretokenizer "$pretokenizer"; then
        echo "✓ Completed successfully at: $(date)"
      else
        echo "✗ Failed at: $(date)"
        echo "Continuing with next job..."
      fi
      
      echo "===========================================\n"
    done
  done
done

echo "Done training unigram on: ${corpora[*]} with pretokenizers: ${PRETOKENIZERS[*]}"
