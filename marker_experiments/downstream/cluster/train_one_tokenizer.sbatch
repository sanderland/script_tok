#!/usr/bin/env bash
# One arm's vocabulary-matched tokenizer, on one Clariden node.
#
#     ARM=bnd_wpd sbatch marker_experiments/downstream/cluster/train_one_tokenizer.sbatch
#
# One arm per job rather than all arms in one job, because each arm needs its own
# pretokenized corpus (the corpus path keys on the pretokenizer hash, so no two arms can
# share one) and those builds are independent. Measured on the first arm: 54 min to
# stream and sample 5e9 chars out of 45.8e9 scanned, then pretokenization, then ~21 min
# of BPE training. Pretokenization took 8.6 h at the old hardcoded 16 workers; this job
# passes --workers so it uses the node.
#
# Each job writes its own manifest, since train_matched.py rewrites that file by
# read-modify-write and concurrent jobs would drop each other's entries. Fold them in
# with merge_manifests.py once the jobs finish.
#
#SBATCH --job-name=marker_tok1
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

: "${ARM:?set ARM (plain, bnd_w, bnd_wpd, bnd_wpd_caps, ...)}"

TRAINER="${TRAINER:-bpe}"
CORPUS="${CORPUS:-fineweb_en_5gb}"
VOCAB="${VOCAB:-34685}"
# 128 partitions in a PretokenizedCorpus, so more workers than that do not help training.
WORKERS="${WORKERS:-128}"

MANIFEST_DIR="${OUT}/manifests"
mkdir -p "${MANIFEST_DIR}" "${OUT}/slurm"

echo "[job] ${SLURM_JOB_ID:-nojob} arm=${ARM} trainer=${TRAINER} corpus=${CORPUS} vocab=${VOCAB} workers=${WORKERS}"
echo "[job] node=$(hostname) cpus=$(nproc) commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"

srun -ul uv run python -u marker_experiments/downstream/train_matched.py \
    --arms "${ARM}" \
    --trainer "${TRAINER}" \
    --corpus "${CORPUS}" \
    --total-vocab "${VOCAB}" \
    --workers "${WORKERS}" \
    --eval-texts marker_experiments/eval_texts/en.json \
    --manifest-path "${MANIFEST_DIR}/${CORPUS}_${ARM}_${TRAINER}_v${VOCAB}.json"
