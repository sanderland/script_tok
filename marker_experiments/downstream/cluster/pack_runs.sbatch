#!/usr/bin/env bash
# Several downstream runs concurrently on ONE node, one GPU each.
#
#     REPO=$PWD RUNS="plain:0 bnd_wpd:0 plain:1" TRAINER=mingram \
#       OUT=results/mingram_downstream CORE_SAFE_ARMS="plain" \
#       NANOCHAT_BASE=/capstor/.../nanochat_base sbatch .../pack_runs.sbatch
#
# Submit through submit_mingram.sh, which sets all of those.
#
# Partition `normal` is OverSubscribe=EXCLUSIVE, so a job holds all 4 GH200s and all 288
# cores whether or not it uses them, and a node-hour bills as 4 GPU-hours. One run per
# node therefore pays for 4 GPUs to drive 1. Each run here gets its own GPU through
# CUDA_VISIBLE_DEVICES and its own share of the cores through ENCODE_WORKERS.
#
# Packing is cost-neutral in the limit where wall-clock scales perfectly with cores: the
# same total core-hours then take proportionally longer on a shared node. It pays to the
# extent a single run leaves cores idle, and the GPU-bound part of the run leaves them
# idle by construction.
#
# GPUS stays 1. The MinGram sweep is compared against BPE runs that used one GPU, and
# --gpus 4 would change the DDP world size, the effective batch and the data order.
#
#SBATCH --job-name=packruns
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -euo pipefail

# Required, with no path fallback. Slurm copies the batch script into its spool directory
# before running it, so BASH_SOURCE[0] here is that copy and deriving the repo from it
# would give a wrong path. There are four checkouts of this repo on different branches and
# picking the wrong one runs code no commit in this job's record describes.
: "${REPO:?set REPO to the checkout this job should run}"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh
# Pin imports to THIS checkout. The venv installs script_bpe and pynanochat as editable
# pointers into /users/cmeister747/script_tok, so a job launched from a worktree would
# otherwise run that tree's library code, on whatever branch it happens to be on. Both
# entries are needed: REPO alone still resolves pynanochat to the main checkout.
export PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat:${PYTHONPATH:-}"

: "${RUNS:?set RUNS, space separated arm:seed pairs}"
# Required, not defaulted. run_arms.sh defaults TRAINER to bpe, and a mingram sweep that
# inherited that default would find the bpe runs already logged and exit reporting success
# having trained nothing.
: "${TRAINER:?set TRAINER explicitly, bpe or mingram}"
# Required for the same reason: run_arms.sh defaults this to "plain bnd_w", measured on
# the BPE tokenizers. An arm listed here that does not satisfy CORE's prefix property
# makes base_eval raise on the first violation, and the run then reports nothing at all,
# bits-per-byte included. No `:` in the test: a sweep where no arm satisfies the property
# sets this to the empty string, which is a valid explicit value.
: "${CORE_SAFE_ARMS?set CORE_SAFE_ARMS explicitly; the empty string is a valid value}"
: "${OUT:?set OUT}"
# Captured BEFORE env.sh is sourced, and re-exported after. env.sh assigns NANOCHAT_BASE
# unconditionally, with no `:-`, so sourcing it discards whatever the caller passed. That
# made the check below unreachable and this variable a knob that did nothing: a sweep
# pointed at the wrong shard directory would have run against env.sh's directory without
# a word. The two paths happen to agree today, which is exactly why it went unnoticed.
: "${NANOCHAT_BASE_REQUESTED:?set NANOCHAT_BASE}"
export NANOCHAT_BASE="${NANOCHAT_BASE_REQUESTED}"
[[ -d "${NANOCHAT_BASE}" ]] || { echo "no such NANOCHAT_BASE: ${NANOCHAT_BASE}" >&2; exit 1; }

DEPTH="${DEPTH:-12}"
CORPUS="${CORPUS:-fineweb_en_5gb}"
VOCAB="${VOCAB:-34685}"
NUM_SHARDS="${NUM_SHARDS:-8}"
# Pinned, not inherited. --export=ALL carries the submitting shell's environment, so a
# SMOKE=1 left over from a pipeline check would make every run here train 20 iterations,
# tag itself _smoke, and still print a result block that collect_results.py parses into a
# row. run_one.sbatch pinned SMOKE for this reason; the packed path had dropped it.
export SMOKE=0
export TAG_SUFFIX=""

NRUNS=$(wc -w <<< "${RUNS}")
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( NRUNS > NGPU )); then
  echo "refusing: ${NRUNS} runs on a node with ${NGPU} GPUs. Two runs on one GPU would" >&2
  echo "contend for it and neither would be a clean single-GPU replicate." >&2
  exit 1
fi

CORES=$(nproc)
# Leave headroom so the concurrent parents are not fighting for the last core.
# `:-` and not `-`: an explicit empty value would reach ENCODE_WORKERS, run_arms.sh drops
# an empty flag, and run_downstream_eval.py would then size each run's pool for a whole
# node, 1148 worker processes across the four runs.
PER_RUN_WORKERS="${PER_RUN_WORKERS:-$(( (CORES - 8) / NRUNS ))}"
# Asserted, because 0 is not an error anywhere downstream: run_arms.sh's `-n` test passes,
# and bootstrap.py creates no pool when the count is not above 1, so the run encodes
# serially with nothing in the log. Measured single-process throughput here is 55,277
# tokens/s for bnd_wpd mingram, so a run needing about 2.1e9 encoded tokens would take
# roughly 10 hours instead of 9 minutes and hit the 12-hour wall with nothing to show.
if (( PER_RUN_WORKERS < 1 )); then
  echo "refusing: ${CORES} cores across ${NRUNS} runs gives ${PER_RUN_WORKERS} encode" >&2
  echo "workers per run. Below 1 the run silently encodes serially." >&2
  exit 1
fi

COMMIT=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain | wc -l)
VENDOR_COMMIT=$(git -C eval/py-nanochat/vendor/nanochat rev-parse HEAD)
echo "[pack] ${SLURM_JOB_ID:-nojob} node=$(hostname) cores=${CORES} gpus=${NGPU}"
echo "[pack] repo=${REPO} commit=${COMMIT} dirty=${DIRTY} nanochat=${VENDOR_COMMIT}"
echo "[pack] trainer=${TRAINER} depth=${DEPTH} vocab=${VOCAB} shards=${NUM_SHARDS}"
echo "[pack] core_safe_arms='${CORE_SAFE_ARMS}' base=${NANOCHAT_BASE} out=${OUT}"
echo "[pack] ${NRUNS} run(s) at ${PER_RUN_WORKERS} encode workers each: ${RUNS}"
mkdir -p "${OUT}/slurm" "${OUT}/logs"
uv pip freeze > "${OUT}/slurm/${SLURM_JOB_ID:-nojob}.freeze" 2>/dev/null || true

pids=() names=()
gpu=0
for run in ${RUNS}; do
  arm="${run%%:*}"; seed="${run##*:}"
  tok="marker_experiments/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
  # Assigned, not interpolated into the echo. A command substitution inside an echo
  # argument cannot trip `set -e`, so an unreadable tokenizer would print an empty hash
  # and the group would start anyway with no usable provenance for that run.
  TOK_SHA=$(sha256sum "${tok}" | cut -d" " -f1)
  echo "[pack]   ${arm}/s${seed} -> gpu ${gpu}, tokenizer_sha256=${TOK_SHA}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export ARMS="${arm}" SEEDS="${seed}"
    export TRAINER DEPTH CORPUS VOCAB NUM_SHARDS OUT NANOCHAT_BASE CORE_SAFE_ARMS
    export GPUS=1
    export ENCODE_WORKERS="${PER_RUN_WORKERS}" TRAIN_WORKERS="${PER_RUN_WORKERS}"
    # run_arms.sh ends by copying its TSV over marker_experiments/paper/generated/ and
    # regenerating the tables. Those artifacts are tracked and carry the BPE numbers; a
    # single-trainer sweep TSV would overwrite them.
    export SKIP_PAPER_ARTIFACTS=1
    marker_experiments/downstream/run_arms.sh
  ) > "${OUT}/slurm/pack_${arm}_s${seed}_${SLURM_JOB_ID:-nojob}.out" 2>&1 &
  pids+=($!) names+=("${arm}/s${seed}")
  gpu=$(( gpu + 1 ))
done

# Report every failure rather than dying on the first, so one bad run does not hide the
# rest of the node's work.
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[pack] ${names[$i]} ok"
  else
    echo "[pack] ${names[$i]} FAILED"
    fail=1
  fi
done

# One authoritative collection after the join. Each run_arms.sh also collects when it
# finishes, and two runs finishing close together can both read the log directory before
# either writes, so the file either leaves behind can be missing the other's row. This
# pass runs when every log is complete and overwrites whatever they left.
uv run python marker_experiments/downstream/collect_results.py \
    --logs-dir "${OUT}/logs" --out "${OUT}/results.tsv" || fail=1
echo "[pack] done, fail=${fail}"
exit "${fail}"
