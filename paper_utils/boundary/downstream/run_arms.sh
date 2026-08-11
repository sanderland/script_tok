#!/usr/bin/env bash
# Downstream LM comparison for boundary-marker tokenizers: one pretrained model per
# arm x seed, then a TSV of the results.
#
#   ARMS=plain,bnd_w,bnd_wpd,bnd_wpd_caps SEEDS=0,1,2 DEPTH=12 ./run_arms.sh
#
# This is a plain sequential driver. Each model is trained by one invocation of
# paper_utils/hybrid/downstream/run_downstream_eval.py, so if you want to parallelise
# across nodes, run that script yourself per (arm, seed) under whatever scheduler you
# have -- there is nothing here that has to run in one process. Runs are skip-if-done
# (a log containing `artifact_dir`), so re-running after an interruption, or after
# submitting some cells elsewhere, only fills in what is missing.
#
# Set SMOKE=1 for the ~minutes-long pipeline check (CORE will be near random; the point
# is that download -> inject -> train -> eval -> parse all work for these tokenizers).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

ARMS=${ARMS:-plain,bnd_w,bnd_wpd,bnd_wpd_caps}
SEEDS=${SEEDS:-0}
TRAINER=${TRAINER:-bpe}
CORPUS=${CORPUS:-fineweb_en_5gb_quick}
VOCAB=${VOCAB:-34685}
DEPTH=${DEPTH:-12}
GPUS=${GPUS:-1}
NUM_SHARDS=${NUM_SHARDS:-8}
TRAIN_WORKERS=${TRAIN_WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
SMOKE=${SMOKE:-0}
# Appended to every run tag. Set it when a run differs from the default configuration in a
# way the tag would otherwise hide, e.g. TAG_SUFFIX=_n32 for a 32-shard data regime, so the
# two do not share a log, a checkpoint directory or a row in the TSV.
TAG_SUFFIX=${TAG_SUFFIX:-}
OUT=${OUT:-results/marker_downstream}
NANOCHAT_BASE=${NANOCHAT_BASE:-$HOME/.cache/nanochat_marker}

TOK_DIR="paper_utils/boundary/downstream/tokenizers"
DOTTED_BPE=paper_utils.boundary.downstream.boundary_tokenizer.BoundaryBPETokenizer
DOTTED_MINGRAM=paper_utils.boundary.downstream.boundary_tokenizer.BoundaryMinGramModel
if [[ "$TRAINER" == "mingram" ]]; then DOTTED=$DOTTED_MINGRAM; else DOTTED=$DOTTED_BPE; fi

mkdir -p "$OUT/logs"

# Nothing below is worth starting without a GPU: run_downstream_eval.py pretrains a
# nanochat model per arm x seed. Fail here, with what to do instead, rather than a
# traceback out of torch several minutes in.
if ! uv run --extra downstream python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  cat >&2 <<'MSG'
run_arms.sh needs CUDA: it pretrains one nanochat model per arm x seed.

  no GPU visible, or the downstream extra is not installed.

  prerequisites:
    uv sync --extra downstream
    git clone the vendored nanochat under eval/py-nanochat/vendor/nanochat

  to reproduce the downstream table WITHOUT running the sweep, from the committed
  results TSVs:
    uv run python paper_utils/boundary/downstream/make_tex_tables.py
MSG
  exit 1
fi

echo "== step 1/4: vocabulary-matched tokenizers (${TRAINER}, vocab ${VOCAB}, ${CORPUS})"
uv run python paper_utils/boundary/downstream/train_matched.py \
    --arms "$ARMS" --trainer "$TRAINER" --corpus "$CORPUS" \
    --total-vocab "$VOCAB" --workers "$TRAIN_WORKERS"

echo "== step 2/4: tokenizer-side checks (must be clean before burning GPU hours)"
uv run python paper_utils/boundary/downstream/smoke_test.py \
    --tokenizer-dir "$TOK_DIR" --pattern "_${TRAINER}_v${VOCAB}" --corpus "${CORPUS}_" \
    --require-matched-vocab

echo "== step 3/4: byte factors"
# Measured once per arm into the shared cache rather than once per run: it is
# single-threaded over the whole validation shard, so every seed of an arm would otherwise
# repeat about a quarter-hour of it, and a bad shard or tokenizer surfaces here instead of
# after the first model is trained.
uv run python paper_utils/boundary/downstream/precompute_byte_factors.py \
    --arms "$ARMS" --trainer "$TRAINER" --corpus "$CORPUS" --vocab "$VOCAB" \
    --base-dir "$NANOCHAT_BASE" || echo "  (byte factors will be measured per run)"

echo "== step 4/4: downstream runs"
IFS=',' read -ra ARM_LIST <<< "$ARMS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"
for arm in "${ARM_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    tok="${TOK_DIR}/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
    [[ -f "$tok" ]] || { echo "missing $tok"; exit 1; }
    # Smoke belongs in the tag. Without it a 20-iteration run writes a full result block
    # into the real run's log, which then counts as finished forever, and drops a
    # model_000020.pt into the real run's checkpoint directory.
    tag="${arm}_${TRAINER}_d${DEPTH}_s${seed}${TAG_SUFFIX:-}"
    [[ "$SMOKE" == "1" ]] && tag="${tag}_smoke"
    log="$OUT/logs/${tag}.log"
    # Completion is the printed result block, not the CORE line: a bpb-only run never
    # prints a CORE metric, and keying on that would rerun it forever.
    if [[ -s "$log" ]] && grep -q "artifact_dir" "$log"; then
      echo "-- $tag: already done, skipping"
      continue
    fi
    echo "-- $tag -> $log"
    smoke_flag=()
    [[ "$SMOKE" == "1" ]] && smoke_flag=(--smoke)
    # bpb only. CORE's language_modeling tasks assert that encode(context) is a prefix of
    # encode(context + continuation), and a scheme that marks a character from its right
    # neighbour breaks that: core_prefix_check.py counts 311/512 violations for bnd_wp and
    # 399/512 for bnd_wpd against 0/512 for plain and bnd_w. base_eval raises on the first
    # one, so the run would report nothing at all, bpb included. Scoring only the arms that
    # survive would compare a subset, so no arm is scored on CORE.
    #
    # --shuffle-data-order: without it every seed reads the identical document sequence and
    # the spread measures weight initialization alone, which would understate the variance
    # the paired test divides by.
    uv run python paper_utils/hybrid/downstream/run_downstream_eval.py \
        --tokenizer-path "$tok" \
        --tokenizer-class "$DOTTED" \
        --tokenizer-id "$tag" \
        --depth "$DEPTH" --gpus "$GPUS" --seed "$seed" \
        --num-shards "$NUM_SHARDS" \
        --base-dir "$NANOCHAT_BASE" \
        --eval-modes bpb \
        --shuffle-data-order \
        "${smoke_flag[@]}" 2>&1 | tee "$log"
  done
done

echo "== collecting"
uv run python paper_utils/boundary/downstream/collect_results.py --logs-dir "$OUT/logs" --out "$OUT/results.tsv"

if [[ "${SKIP_PAPER_ARTIFACTS:-0}" != "1" ]]; then
  # Two destinations on purpose: $OUT keeps the run's own record next to its logs, and the
  # paper artifact directory holds the copy the table generator reads. Set
  # SKIP_PAPER_ARTIFACTS=1 for a sweep covering one arm or one trainer, whose partial TSV
  # would otherwise overwrite the tracked one.
  GENERATED="paper_utils/boundary/paper/generated"
  mkdir -p "$GENERATED"
  cp "$OUT/results.tsv" "$GENERATED/results.tsv"
  # Non-fatal: it also needs manifest.json, which a sweep that only ran the LM leg has not
  # produced. A missing table is not a reason to fail a finished sweep.
  uv run python paper_utils/boundary/downstream/make_tex_tables.py || echo "  (tables skipped)"
fi
