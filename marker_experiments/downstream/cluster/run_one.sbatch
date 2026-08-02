#!/usr/bin/env bash
# One arm x seed of the boundary-marker downstream eval, on one Clariden node.
#
# Submit through submit_all.sh, which sets ARM and SEED per job:
#     ARM=bnd_wpd SEED=0 sbatch marker_experiments/downstream/cluster/run_one.sbatch
#
# Partition `normal` is OverSubscribe=EXCLUSIVE, so a job holds the whole node:
# 4x GH200 and 288 cores. The run uses one GPU, per the README. The other three
# sit idle on purpose: run_downstream_eval.py tokenizes on the fly with
# `(cpu_count - gpus) // gpus` worker processes, and script_bpe.encode is pure
# Python, so wall-clock is set by how many cores one run gets, not by GPUs.
#
#SBATCH --job-name=marker_ds
#SBATCH --account=a0229
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=results/marker_downstream/slurm/%x_%j.out
#SBATCH --error=results/marker_downstream/slurm/%x_%j.err

set -euo pipefail

REPO="${REPO:-/users/cmeister747/script_tok}"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh

: "${ARM:?set ARM (plain, bnd_wpd, bnd_wpd_caps, ...)}"
: "${SEED:?set SEED}"

export ARMS="${ARM}"
export SEEDS="${SEED}"
export DEPTH="${DEPTH:-12}"
export TRAINER="${TRAINER:-bpe}"
export CORPUS="${CORPUS:-fineweb_en_5gb}"
export VOCAB="${VOCAB:-34685}"
export GPUS=1
export NUM_SHARDS="${NUM_SHARDS:-8}"
export TRAIN_WORKERS="${TRAIN_WORKERS:-90}"
export SMOKE="${SMOKE:-0}"

# Reproducibility record. Assigned first, not interpolated into echo: a command
# substitution inside an echo argument cannot trip `set -e`, so a failing git call would
# have silently recorded an empty commit and let the run proceed.
COMMIT=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain | wc -l)
# The model, the optimizer and the bpb metric all live in the vendored nanochat clone,
# which is gitignored and not a submodule, so the repo commit alone does not describe what
# ran.
VENDOR_DIR="eval/py-nanochat/vendor/nanochat"
VENDOR_COMMIT=$(git -C "${VENDOR_DIR}" rev-parse HEAD)
VENDOR_DIRTY=$(git -C "${VENDOR_DIR}" status --porcelain | wc -l)
echo "[job] ${SLURM_JOB_ID:-nojob} arm=${ARM} seed=${SEED} depth=${DEPTH} trainer=${TRAINER}"
echo "[job] node=$(hostname) commit=${COMMIT} dirty=${DIRTY}"
echo "[job] nanochat=${VENDOR_COMMIT} dirty=${VENDOR_DIRTY}"
echo "[job] tokenizer_sha256=$(sha256sum "marker_experiments/downstream/tokenizers/${CORPUS:-fineweb_en_5gb}_${ARM}_${TRAINER}_v${VOCAB:-34685}.json.gz" | cut -d" " -f1)"
# Resolved dependency set, so a number can be tied to the environment that produced it.
# The venv is shared and mutable, so recording it per job is the only durable trace.
uv pip freeze > "${OUT}/slurm/${SLURM_JOB_ID:-nojob}.freeze" 2>/dev/null || true
echo "[job] NANOCHAT_BASE=${NANOCHAT_BASE} OUT=${OUT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun -ul marker_experiments/downstream/run_arms.sh
