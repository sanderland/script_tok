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

# Reproducibility record: what ran, from which commit, on which node.
echo "[job] ${SLURM_JOB_ID:-nojob} arm=${ARM} seed=${SEED} depth=${DEPTH} trainer=${TRAINER}"
echo "[job] node=$(hostname) commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
echo "[job] NANOCHAT_BASE=${NANOCHAT_BASE} OUT=${OUT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

srun -ul marker_experiments/downstream/run_arms.sh
