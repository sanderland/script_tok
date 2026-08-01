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

ARMS=${ARMS:-plain,bnd_w,bnd_wpd,bnd_wpd_caps}
SEEDS=${SEEDS:-0}
# Arms whose tokenization of a context is a prefix of its tokenization of that context
# plus a continuation, which is what CORE's language_modeling tasks assert. Measured with
# core_prefix_check.py on the real CORE prompts: plain 0/512 and bnd_w 0/512 violations,
# against bnd_wp 311/512, bnd_wpd 399/512, bnd_wpd_caps 399/512. Any arm not listed here
# is scored on bpb alone, because base_eval raises on the first violation and the run then
# reports nothing at all, bpb included.
CORE_SAFE_ARMS=${CORE_SAFE_ARMS:-plain bnd_w}
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
# --manifest-path per arm: this script runs inside every job of the sweep, and the
# manifest is rewritten read-modify-write, so a shared path lets concurrent jobs drop each
# other's entries. Harmless while every tokenizer already exists and training is skipped,
# but not something to leave armed. merge_manifests.py folds these back together.
mkdir -p "$OUT/manifests"
uv run python marker_experiments/downstream/train_matched.py \
    --arms "$ARMS" --trainer "$TRAINER" --corpus "$CORPUS" \
    --total-vocab "$VOCAB" --workers "$TRAIN_WORKERS" \
    --eval-texts marker_experiments/eval_texts/en.json \
    --manifest-path "$OUT/manifests/${CORPUS}_${ARMS//,/_}_${TRAINER}_v${VOCAB}.json"

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
    # Completion is the printed result block, not the CORE line: a bpb-only run never
    # prints a CORE metric, and keying on that would rerun it forever.
    if [[ -s "$log" ]] && grep -q "Downstream eval result" "$log"; then
      echo "-- $tag: already done, skipping"
      continue
    fi
    if [[ " $CORE_SAFE_ARMS " == *" $arm "* ]]; then
      eval_modes="core,bpb"
    else
      eval_modes="bpb"
    fi
    echo "-- $tag -> $log (eval: $eval_modes)"
    smoke_flag=()
    [[ "$SMOKE" == "1" ]] && smoke_flag=(--smoke)
    uv run python paper_utils/hybrid/downstream/run_downstream_eval.py \
        --tokenizer-path "$tok" \
        --tokenizer-class "$DOTTED" \
        --tokenizer-id "$tag" \
        --depth "$DEPTH" --gpus "$GPUS" --seed "$seed" \
        --num-shards "$NUM_SHARDS" \
        --base-dir "$NANOCHAT_BASE" \
        --eval-modes "$eval_modes" \
        "${smoke_flag[@]}" 2>&1 | tee "$log"
  done
done

echo "== collecting"
uv run python marker_experiments/downstream/collect_results.py --logs-dir "$OUT/logs" --out "$OUT/results.tsv"
