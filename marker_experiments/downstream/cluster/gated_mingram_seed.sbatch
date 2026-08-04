#!/usr/bin/env bash
# One seed of the MinGram downstream sweep for bnd_w, bnd_wpd_caps and bnd_wpd_extcaps,
# with the tokenizer-side gates run inside the job.
#
#     SEED=0 REPO=$PWD sbatch --dependency=afterok:<tokenizer job> .../gated_mingram_seed.sbatch
#
# The gates normally run at submit time in submit_mingram.sh, which needs the tokenizers to
# exist. Two of these three are still training, so the checks move in here and run after the
# dependency clears. They are not skipped: a wrong CORE_SAFE_ARMS makes base_eval raise
# after bits-per-byte is computed but before the result block prints, losing an hour per run
# and leaving a log with no artifact_dir that every later submission retries forever. MinGram
# can also prune a token past its target, which is how fineweb_ru_5gb_plain_mingram landed at
# 34,684, so the vocabulary is checked per arm rather than assumed.
#
#SBATCH --job-name=mgds_gated
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -euo pipefail

: "${REPO:?set REPO to the checkout this job should run}"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh
# Pin imports to THIS checkout. The venv installs script_bpe and pynanochat as editable
# pointers into /users/cmeister747/script_tok, and sys.path[0] is the script's directory,
# not the repo root, so without this a run resolves library code from another branch.
export PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat:${PYTHONPATH:-}"

: "${SEED:?set SEED}"
ARMS_LIST="bnd_w bnd_wpd_caps bnd_wpd_extcaps"
TRAINER=mingram
CORPUS=fineweb_en_5gb
VOCAB=34685
DEPTH=12
NUM_SHARDS=8
NANOCHAT_BASE_DIR=/capstor/scratch/cscs/cmeister747/marker_downstream/nanochat_base
OUT_DIR=results/mingram_downstream

echo "[gate] ${SLURM_JOB_ID:-nojob} seed=${SEED} node=$(hostname) commit=$(git rev-parse HEAD)"

# 1. every tokenizer exists and is exactly the matched size. The adapter reports vocab+1
#    for the synthetic BOS it adds.
for arm in ${ARMS_LIST}; do
  tok="marker_experiments/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
  [[ -f "$tok" ]] || { echo "[gate] missing tokenizer: $tok" >&2; exit 1; }
  n=$(uv run python -c "
from marker_experiments.downstream.boundary_tokenizer import BoundaryMinGramModel
from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter
print(ScriptBPETokenizerAdapter(BoundaryMinGramModel.load('$tok')).get_vocab_size())
")
  if [[ "$n" != "$((VOCAB + 1))" ]]; then
    echo "[gate] refusing: ${arm} vocabulary is ${n}, expected $((VOCAB + 1))" >&2
    exit 1
  fi
  echo "[gate] ${arm}: vocabulary ${n}, matched, sha256=$(sha256sum "$tok" | cut -d' ' -f1)"
done

# 2. measure the CORE prefix property and build CORE_SAFE_ARMS from the measurement.
CHECK=$(NANOCHAT_BASE="${NANOCHAT_BASE_DIR}" uv run python \
        marker_experiments/downstream/core_prefix_check.py \
        --tokenizer-dir marker_experiments/downstream/tokenizers \
        --pattern "_${TRAINER}_v${VOCAB}" --trainer "${TRAINER}" || true)
SAFE=""
for arm in ${ARMS_LIST}; do
  line=$(grep -F "${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz" <<< "${CHECK}" || true)
  [[ -n "$line" ]] || { echo "[gate] no CORE result for ${arm}" >&2; echo "${CHECK}" >&2; exit 1; }
  echo "[gate] ${line}"
  [[ "$line" == *"core,bpb"* ]] && SAFE+="${arm} "
done
echo "[gate] CORE_SAFE_ARMS='${SAFE}'"

# 3. the three runs, one per GPU, with the cores split between them.
CORES=$(nproc)
PER_RUN=$(( (CORES - 8) / 3 ))
(( PER_RUN >= 1 )) || { echo "[gate] ${PER_RUN} encode workers per run" >&2; exit 1; }
echo "[gate] 3 runs at ${PER_RUN} encode workers each"

pids=() names=()
gpu=0
for arm in ${ARMS_LIST}; do
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export ARMS="${arm}" SEEDS="${SEED}"
    export TRAINER DEPTH CORPUS VOCAB NUM_SHARDS
    export OUT="${OUT_DIR}" NANOCHAT_BASE="${NANOCHAT_BASE_DIR}"
    export CORE_SAFE_ARMS="${SAFE}"
    export GPUS=1
    export ENCODE_WORKERS="${PER_RUN}" TRAIN_WORKERS="${PER_RUN}"
    # Pinned, not inherited from the submitting shell: a stray SMOKE=1 would train 20
    # iterations and still print a result block that collect_results.py parses into a row.
    export SMOKE=0 TAG_SUFFIX=
    # Those paper artifacts are tracked and carry the BPE numbers; this sweep is one trainer.
    export SKIP_PAPER_ARTIFACTS=1
    marker_experiments/downstream/run_arms.sh
  ) > "${OUT_DIR}/slurm/gated_${arm}_s${SEED}_${SLURM_JOB_ID:-nojob}.out" 2>&1 &
  pids+=($!) names+=("${arm}/s${SEED}")
  gpu=$(( gpu + 1 ))
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[gate] ${names[$i]} ok"; else echo "[gate] ${names[$i]} FAILED"; fail=1; fi
done
uv run python marker_experiments/downstream/collect_results.py \
    --logs-dir "${OUT_DIR}/logs" --out "${OUT_DIR}/results.tsv" || fail=1
echo "[gate] done, fail=${fail}"
exit "${fail}"
