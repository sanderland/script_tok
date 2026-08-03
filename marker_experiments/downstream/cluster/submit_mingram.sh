#!/usr/bin/env bash
# The MinGram downstream sweep: plain and bnd_wpd, seeds 0-2, matched to the BPE runs.
#
#     marker_experiments/downstream/cluster/submit_mingram.sh
#
# Two arms rather than four. The question is whether the downstream conclusion is specific
# to greedy merge training, and plain against bnd_wpd is the comparison that carries it:
# bnd_wpd is the arm whose gain cannot come from spending more forward passes per byte,
# because it emits fewer tokens per byte than plain, not more.
#
# These stay separate from the BPE runs because the run tag carries the trainer
# (run_arms.sh builds `<arm>_<trainer>_d<depth>_s<seed>`), so the logs, the checkpoint
# directories under NANOCHAT_BASE/runs and the TSV rows all differ. OUT differs as well,
# which keeps the two sweeps' TSVs apart.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh

ARMS=${ARMS:-plain,bnd_wpd}
# Which arms CORE can score, measured on the MinGram tokenizers by mingram_preflight.py
# rather than inherited from run_arms.sh's default. That default was measured on BPE
# tokenizers, and MinGram segments with a dynamic program instead of greedy merge replay,
# so an arm CORE-safe under BPE need not be under MinGram. Getting it wrong loses the whole
# run: base_eval raises on the first violation and reports nothing, bits-per-byte included.
# No `:` in the test: a sweep where no arm satisfies the prefix property sets this to the
# empty string, which is a valid explicit value that `:?` would reject.
: "${CORE_SAFE_ARMS?run mingram_preflight.py and pass the CORE_SAFE_ARMS it prints}"
# No default. a0229 and infra01 bill different budgets and carry different fairshare, and
# picking one silently on the operator's behalf spends from a budget they did not name.
: "${ACCOUNT:?set ACCOUNT (a0229 or infra01)}"
SEEDS=${SEEDS:-0,1,2}
TRAINER=mingram
DEPTH=${DEPTH:-12}
VOCAB=${VOCAB:-34685}
CORPUS=${CORPUS:-fineweb_en_5gb}
# 8, matching the committed BPE runs. The shard count sets how much text the model sees,
# so it is one of the settings that must not differ between the two sweeps.
NUM_SHARDS=${NUM_SHARDS:-8}
NANOCHAT_BASE=${NANOCHAT_BASE:-/capstor/scratch/cscs/cmeister747/marker_downstream/nanochat_base}
OUT_DIR=results/mingram_downstream
# Runs per node. A node is exclusive and bills 4 GPU-hours an hour whether or not its 4
# GPUs are touched, so one run per node pays for 4 GPUs to drive 1. Never above the GPU
# count: pack_runs.sbatch refuses, and two runs sharing a GPU would not be clean
# single-GPU replicates.
PACK=${PACK:-4}

# `git status --porcelain`, not `git diff`. Neither `git diff HEAD` nor `git diff --cached`
# reports untracked files, so the guard passed while cluster/pack_runs.sbatch, the file the
# job actually runs, was untracked and described by no commit. That is the exact failure
# the guard exists to prevent.
DIRT=$(git status --porcelain)
if [[ -n "${DIRT}" ]]; then
  echo "refusing to submit: working tree differs from HEAD, so the jobs would run code" >&2
  echo "no commit describes. Untracked files count." >&2
  echo "${DIRT}" >&2
  exit 1
fi

# Verify each tokenizer is exactly the matched size before spending any GPU time. MinGram
# prunes down to a target and can land short; the adapter reports vocab+1 for the synthetic
# BOS. This is the one check that must not depend on an operator remembering to run the
# preflight and read its output.
for arm in ${ARMS//,/ }; do
  tok="marker_experiments/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
  [[ -f "$tok" ]] || { echo "missing tokenizer: $tok" >&2; exit 1; }
  n=$(PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat" uv run python -c "
import sys
from marker_experiments.downstream.boundary_tokenizer import BoundaryMinGramModel
from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter
print(ScriptBPETokenizerAdapter(BoundaryMinGramModel.load('$tok')).get_vocab_size())
")
  if [[ "$n" != "$((VOCAB + 1))" ]]; then
    echo "refusing to submit: ${arm} vocabulary is ${n}, expected $((VOCAB + 1))" >&2
    exit 1
  fi
  echo "-- ${arm}: vocabulary ${n}, matched"
done

# Measure CORE_SAFE_ARMS rather than trust the value passed in. A wrong one is the most
# expensive mistake available here: base_eval raises on the first prefix violation, after
# bits-per-byte has been computed but before the result block is printed, so an hour of
# training per run is lost, the log has no artifact_dir, and every later invocation of this
# script resubmits the run forever. The measurement is the same one base_eval's tasks
# perform, at the same max_per_task, and takes under a minute.
echo "-- checking the CORE prefix property on the ${TRAINER} tokenizers"
CHECK=$(PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat" NANOCHAT_BASE="${NANOCHAT_BASE}" \
        uv run python marker_experiments/downstream/core_prefix_check.py \
        --tokenizer-dir marker_experiments/downstream/tokenizers \
        --pattern "_${TRAINER}_v${VOCAB}" --trainer "${TRAINER}" || true)
MEASURED=""
for arm in ${ARMS//,/ }; do
  line=$(grep -F "${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz" <<< "${CHECK}" || true)
  if [[ -z "${line}" ]]; then
    echo "refusing to submit: core_prefix_check.py reported nothing for ${arm}." >&2
    echo "${CHECK}" >&2
    exit 1
  fi
  echo "   ${line}"
  [[ "${line}" == *"core,bpb"* ]] && MEASURED+="${arm} "
done
# Compared as sorted word lists so ordering and spacing do not matter.
norm() { tr ' ' '\n' <<< "$1" | grep -v '^$' | sort | tr '\n' ' '; }
if [[ "$(norm "${MEASURED}")" != "$(norm "${CORE_SAFE_ARMS}")" ]]; then
  echo "refusing to submit: CORE_SAFE_ARMS='${CORE_SAFE_ARMS}' but the measured set is" >&2
  echo "'$(norm "${MEASURED}")'. Pass the measured value." >&2
  exit 1
fi
echo "-- CORE_SAFE_ARMS='${CORE_SAFE_ARMS}' matches the measurement"

# The shards must already be there. runner.py refuses to download from inside a run, so a
# missing shard would fail every run in the group an hour in rather than here.
have=$(ls "${NANOCHAT_BASE}/base_data_climbmix"/shard_*.parquet 2>/dev/null | wc -l)
if (( have < NUM_SHARDS + 1 )); then
  echo "refusing to submit: ${NANOCHAT_BASE}/base_data_climbmix holds ${have} shard(s)," >&2
  echo "and this sweep declares ${NUM_SHARDS} train shards plus a val shard." >&2
  exit 1
fi
echo "-- shards: ${have} present, ${NUM_SHARDS} train + 1 val required"

mkdir -p "${OUT_DIR}/slurm" "${OUT_DIR}/logs"
# No `|| true`. On a scheduler hiccup that would leave IN_FLIGHT empty, every run would
# look absent, and a second invocation would submit a duplicate of each. Two jobs sharing a
# tokenizer_id share NANOCHAT_BASE/runs/<id>, so they share the checkpoint directory and
# token_bytes.pt, and both tee to one log. That is how round 1 was lost.
if ! IN_FLIGHT=$(squeue -u "$USER" -h -o "%j" 2>&1); then
  echo "refusing to submit: squeue failed, so in-flight jobs cannot be detected and this" >&2
  echo "would submit duplicates that share a checkpoint directory and a log." >&2
  echo "${IN_FLIGHT}" >&2
  exit 1
fi

IFS=',' read -ra ARM_LIST <<< "$ARMS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

# Seed-major: a partially drained queue then leaves a full two-arm comparison at one seed,
# which is usable, rather than three seeds of one arm, which answers nothing. Packing PACK
# runs per node in this order also makes each node's group a whole number of complete
# seeds when PACK is a multiple of the arm count.
PENDING=()
for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARM_LIST[@]}"; do
    tok="marker_experiments/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
    if [[ ! -f "$tok" ]]; then
      echo "-- ${arm}/s${seed}: no tokenizer at ${tok}, skipping" >&2
      continue
    fi
    tag="${arm}_${TRAINER}_d${DEPTH}_s${seed}"
    log="${OUT_DIR}/logs/${tag}.log"
    if [[ -s "$log" ]] && grep -q "artifact_dir" "$log"; then
      echo "-- ${tag}: already has a result, not submitting"; continue
    fi
    # In-flight test on the run, not on the job: a packed job holds several runs, so each
    # run appears in the job name as a `-arm.seed-` field and is matched with its
    # delimiters. Without the delimiters `bnd_wpd.1` would also match `bnd_wpd.11`, and
    # without the dot `plain.1` would match nothing distinguishable from an arm suffix.
    if grep -qF -- "-${arm}.${seed}-" <<< "$IN_FLIGHT"; then
      echo "-- ${tag}: already queued or running, not submitting"; continue
    fi
    PENDING+=("${arm}:${seed}")
  done
done

if (( ${#PENDING[@]} == 0 )); then
  echo "nothing to submit"; exit 0
fi

for (( i = 0; i < ${#PENDING[@]}; i += PACK )); do
  GROUP=("${PENDING[@]:i:PACK}")
  RUNS="${GROUP[*]}"
  # Job name carries every run in the group as a delimited `-arm.seed-` field, so the
  # in-flight test above can find one run inside a packed job.
  name="mgds"
  for r in "${GROUP[@]}"; do name+="-${r/:/.}"; done
  name+="-"
  jid=$(RUNS="${RUNS}" TRAINER="${TRAINER}" DEPTH="${DEPTH}" CORPUS="${CORPUS}" \
        VOCAB="${VOCAB}" NUM_SHARDS="${NUM_SHARDS}" OUT="${OUT_DIR}" \
        NANOCHAT_BASE_REQUESTED="${NANOCHAT_BASE}" \
        CORE_SAFE_ARMS="${CORE_SAFE_ARMS}" REPO="${REPO}" \
        sbatch --parsable --job-name="${name}" --account="${ACCOUNT}" \
        --output="${OUT_DIR}/slurm/%x_%j.out" --error="${OUT_DIR}/slurm/%x_%j.err" \
        --export=ALL \
        marker_experiments/downstream/cluster/pack_runs.sbatch)
  echo "-- job ${jid}: ${RUNS}"
done
