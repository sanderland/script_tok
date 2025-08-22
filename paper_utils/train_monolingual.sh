#!/usr/bin/env bash
set -euo pipefail

PRETOKENIZERS=(bytes_gpt4 bytes_gpt4_cb bytes_gpt4o bytes_gpt4o_cb\
               scriptenc scriptenc_cb scriptenc_cbi scriptenc_gpt4o scriptenc_gpt4o_cb)
corpora=( deu_latn_300mb arb_arab_300mb kor_hang_300mb zho_hans_300mb \
         rus_cyrl_300mb heb_hebr_300mb hin_deva_300mb tha_thai_300mb
         eng_latn_300mb jpn_jpan_300mb vie_latn_300mb pan_guru_300mb )

ns=(64000)

MAX_JOBS=4
N_CPUS=4

parallel --progress -j $MAX_JOBS -v \
  uv run script_bpe/train.py --report \
    -n             {1} \
    --corpus       {2} \
    --pretokenizer {3} \
    --parallel     $N_CPUS \
  ::: "${ns[@]}" \
  ::: "${corpora[@]}" \
  ::: "${PRETOKENIZERS[@]}"

echo "Done training on: ${corpora[*]} with pretokenizers: ${PRETOKENIZERS[*]}"
