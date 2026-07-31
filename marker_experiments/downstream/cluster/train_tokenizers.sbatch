#!/usr/bin/env bash
# Step 1 of the README, run once and shared by every downstream job:
# stream fineweb_en_5gb, pretokenize it per arm, and train one vocabulary-matched
# tokenizer per arm.
#
#     sbatch marker_experiments/downstream/cluster/train_tokenizers.sbatch
#
# The arms run sequentially in a single job on purpose. train_matched.py rewrites
# marker_experiments/downstream/manifest.json after each arm by read-modify-write,
# so three concurrent jobs would clobber each other's entries and the
# matched-vocabulary check at the end would read an incomplete manifest.
#
# Resumable: an arm whose .json.gz already exists is skipped, and each arm's
# pretokenized corpus is cached under results/corpora/<name>/<pretokenizer-hash>/.
# Re-submitting after a timeout picks up where it stopped.
#
#SBATCH --job-name=marker_toks
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

ARMS="${ARMS:-plain,bnd_w,bnd_wpd,bnd_wpd_caps}"
TRAINER="${TRAINER:-bpe}"
CORPUS="${CORPUS:-fineweb_en_5gb}"
VOCAB="${VOCAB:-34685}"
# PretokenizedCorpus splits into 128 partitions, so workers above that do not help
# the training phase. This now also sets the corpus pretokenization worker count, which
# before the registry change was capped at 16 regardless of what was passed here.
WORKERS="${WORKERS:-128}"

echo "[job] ${SLURM_JOB_ID:-nojob} arms=${ARMS} trainer=${TRAINER} corpus=${CORPUS} vocab=${VOCAB} workers=${WORKERS}"
echo "[job] node=$(hostname) cpus=$(nproc) commit=$(git rev-parse HEAD)"

srun -ul uv run python marker_experiments/downstream/train_matched.py \
    --arms "${ARMS}" \
    --trainer "${TRAINER}" \
    --corpus "${CORPUS}" \
    --total-vocab "${VOCAB}" \
    --workers "${WORKERS}" \
    --eval-texts marker_experiments/eval_texts/en.json
