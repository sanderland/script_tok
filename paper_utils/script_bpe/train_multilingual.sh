#!/usr/bin/env bash
set -euo pipefail

PRETOKENIZERS=(bytes_gpt4_cb bytes_gpt4o_cb scriptenc_cb scriptenc_gpt4o_cb scriptenc2_cb scriptenc2_cbi scriptenc3_cb scriptenc3_cbi)
corpora=(CulturaX-subsample-100-bal2)
N=256000
N_CPUS=16

for corpus in "${corpora[@]}"; do
  for pretokenizer in "${PRETOKENIZERS[@]}"; do
    uv run script_bpe/train.py --report \
      -n             "$N" \
      --corpus       "$corpus" \
      --pretokenizer "$pretokenizer" \
      --parallel     "$N_CPUS"
  done
done

echo "Done training on: ${corpora[*]} with pretokenizers: ${PRETOKENIZERS[*]}"
