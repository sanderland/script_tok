#!/usr/bin/env bash
# Submit one job per arm x seed and print the job IDs.
#
#     marker_experiments/downstream/cluster/submit_all.sh
#     ARMS=plain,bnd_wpd SEEDS=0,1 marker_experiments/downstream/cluster/submit_all.sh
#
# run_arms.sh skips any run whose log already holds a "CORE metric" line, so
# re-submitting after a pre-emption or a partial sweep costs nothing.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
source marker_experiments/downstream/cluster/env.sh

ARMS=${ARMS:-plain,bnd_w,bnd_wpd,bnd_wpd_caps}
SEEDS=${SEEDS:-0,1,2}
DEPTH=${DEPTH:-12}
TRAINER=${TRAINER:-bpe}

mkdir -p "${OUT}/slurm" "${OUT}/logs"

IFS=',' read -ra ARM_LIST <<< "$ARMS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

# Seed-major, not arm-major. Slurm broadly starts same-priority jobs in submission
# order, so this completes seed 0 across every arm before any arm's second seed. A
# partially drained queue then leaves a full four-arm comparison at one seed, which is
# usable; the arm-major order would instead leave three seeds of one arm, which answers
# nothing. It also means the first cross-arm read is available after one run's wall-clock
# rather than after three.
for seed in "${SEED_LIST[@]}"; do
  for arm in "${ARM_LIST[@]}"; do
    tag="${arm}_${TRAINER}_d${DEPTH}_s${seed}"
    log="${OUT}/logs/${tag}.log"
    # Same completion test as run_arms.sh: the result block, not the CORE line, because
    # the arms scored on bpb alone never print a CORE metric.
    if [[ -s "$log" ]] && grep -q "Downstream eval result" "$log"; then
      echo "-- ${tag}: already has a result, not submitting"
      continue
    fi
    jid=$(sbatch --parsable \
        --job-name="mds_${tag}" \
        --export=ALL,ARM="${arm}",SEED="${seed}",DEPTH="${DEPTH}",TRAINER="${TRAINER}",REPO="${REPO}" \
        marker_experiments/downstream/cluster/run_one.sbatch)
    echo "-- ${tag}: job ${jid}"
  done
done
