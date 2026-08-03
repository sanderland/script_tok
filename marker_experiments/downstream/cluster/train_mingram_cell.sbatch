#!/usr/bin/env bash
# One MinGram tokenizer, matched to the BPE downstream arms.
#
#     ARM=plain sbatch marker_experiments/downstream/cluster/train_mingram_cell.sbatch
#
# MinGram does not take an existing vocabulary. It trains its own BPE at
# additional_vocab_size * overshoot_factor (1.15) and then shrinks that to the target with
# EM and pruning, so this is a full retrain, not a transformation of the BPE tokenizer.
# What it does reuse is the pretokenized corpus: that cache keys on the pretokenizer hash,
# and MinGram uses the same pretokenizer as BPE for a given arm, so the 4 h pretokenization
# is already on disk and only the trainer runs here.
#
# Wall-clock is unmeasured at this corpus size. The published MinGram grid ran at 250M
# characters, 20x smaller than this, so 12 h is a guess rather than an estimate.
#
#SBATCH --job-name=mgtok
#SBATCH --account=a0229
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=results/mingram_downstream/slurm/%x_%j.out
#SBATCH --error=results/mingram_downstream/slurm/%x_%j.err

set -euo pipefail

REPO="${REPO:-/capstor/scratch/cscs/cmeister747/wt-mingram}"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh
# Same reason as pack_cells.sbatch: the editable installs point at the main checkout.
export PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat:${PYTHONPATH:-}"

: "${ARM:?set ARM (plain or bnd_wpd)}"
CORPUS="${CORPUS:-fineweb_en_5gb}"
VOCAB="${VOCAB:-34685}"

COMMIT=$(git rev-parse HEAD)
echo "[mgtok] ${SLURM_JOB_ID:-nojob} arm=${ARM} trainer=mingram corpus=${CORPUS} vocab=${VOCAB}"
echo "[mgtok] node=$(hostname) cpus=$(nproc) commit=${COMMIT}"

srun -ul uv run python -u marker_experiments/downstream/train_multilang.py \
    --lang en \
    --arms "${ARM}" \
    --trainers mingram \
    --total-vocab "${VOCAB}" \
    --no-commit
