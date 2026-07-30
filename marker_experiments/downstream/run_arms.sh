#!/usr/bin/env bash
# Downstream LM comparison for boundary-marker tokenizers, one arm x seed per GPU job.
#
# Everything is driven by environment variables so the same script works interactively,
# under `srun`, or as a Slurm array (see README). Each run appends a log under $OUT/logs;
# collect_results.py turns those logs into a TSV.
#
#   ARMS=plain,bnd_wpd,bnd_wpd_caps SEEDS=0,1,2 DEPTH=12 ./run_arms.sh
#
# Set SMOKE=1 for the ~minutes-long pipeline check (CORE will be near random; the point
# is that download -> inject -> train -> eval -> parse all work for these tokenizers).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

ARMS=${ARMS:-plain,bnd_wpd,bnd_wpd_caps}
SEEDS=${SEEDS:-0}
TRAINER=${TRAINER:-bpe}
CORPUS=${CORPUS:-fineweb_en_5gb}
VOCAB=${VOCAB:-34685}
DEPTH=${DEPTH:-12}
GPUS=${GPUS:-1}
NUM_SHARDS=${NUM_SHARDS:-8}
TRAIN_WORKERS=${TRAIN_WORKERS:-$(nproc)}
SMOKE=${SMOKE:-0}
OUT=${OUT:-results/marker_downstream}
NANOCHAT_BASE=${NANOCHAT_BASE:-$HOME/.cache/nanochat_marker}

TOK_DIR="marker_experiments/downstream/tokenizers"
DOTTED_BPE=marker_experiments.downstream.boundary_tokenizer.BoundaryBPETokenizer
DOTTED_MINGRAM=marker_experiments.downstream.boundary_tokenizer.BoundaryMinGramModel
if [[ "$TRAINER" == "mingram" ]]; then DOTTED=$DOTTED_MINGRAM; else DOTTED=$DOTTED_BPE; fi

mkdir -p "$OUT/logs"

echo "== step 1/3: vocabulary-matched tokenizers (${TRAINER}, vocab ${VOCAB}, ${CORPUS})"
uv run python marker_experiments/downstream/train_matched.py \
    --arms "$ARMS" --trainer "$TRAINER" --corpus "$CORPUS" \
    --total-vocab "$VOCAB" --workers "$TRAIN_WORKERS" \
    --eval-texts marker_experiments/eval_texts/en.json

echo "== step 2/3: tokenizer-side checks (must be clean before burning GPU hours)"
uv run python marker_experiments/downstream/smoke_test.py \
    --tokenizer-dir "$TOK_DIR" --pattern "_${TRAINER}_v${VOCAB}"

echo "== step 3/3: downstream runs"
IFS=',' read -ra ARM_LIST <<< "$ARMS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for arm in "${ARM_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    tok="${TOK_DIR}/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
    [[ -f "$tok" ]] || { echo "missing $tok"; exit 1; }
    tag="${arm}_${TRAINER}_d${DEPTH}_s${seed}"
    log="$OUT/logs/${tag}.log"
    if [[ -s "$log" ]] && grep -q "CORE metric" "$log"; then
      echo "-- $tag: already done, skipping"
      continue
    fi
    echo "-- $tag -> $log"
    smoke_flag=()
    [[ "$SMOKE" == "1" ]] && smoke_flag=(--smoke)
    uv run python paper_utils/hybrid/downstream/run_downstream_eval.py \
        --tokenizer-path "$tok" \
        --tokenizer-class "$DOTTED" \
        --tokenizer-id "$tag" \
        --depth "$DEPTH" --gpus "$GPUS" --seed "$seed" \
        --num-shards "$NUM_SHARDS" \
        --base-dir "$NANOCHAT_BASE" \
        "${smoke_flag[@]}" 2>&1 | tee "$log"
  done
done

echo "== collecting"
uv run python marker_experiments/downstream/collect_results.py --logs-dir "$OUT/logs" --out "$OUT/results.tsv"
