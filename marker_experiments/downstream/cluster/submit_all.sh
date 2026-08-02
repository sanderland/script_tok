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

# The jobs run the working tree, not a snapshot of a commit, so an uncommitted edit
# would be what actually executes while the provenance line records only a file count.
# A job that starts hours after submission picks up whatever the tree holds at start.
if ! git diff --quiet HEAD || ! git diff --cached --quiet HEAD; then
  echo "refusing to submit: the working tree differs from HEAD, so the jobs would run" >&2
  echo "code that no commit describes. Commit or stash first:" >&2
  git status --porcelain >&2
  exit 1
fi

mkdir -p "${OUT}/slurm" "${OUT}/logs"

# Tags already queued or running. Completion is judged by the log, which an in-flight run
# has not written yet, so without this a second invocation submits a duplicate of every
# unfinished run. Two jobs with one tag share their log, checkpoint and byte table.
IN_FLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

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
    # Completion keyed on the LAST line of the result block, not the header: a run killed
    # mid-block would otherwise be treated as finished by this guard and as absent by
    # collect_results.py, and would never be rerun.
    if [[ -s "$log" ]] && grep -q "artifact_dir" "$log"; then
      echo "-- ${tag}: already has a result, not submitting"
      continue
    fi
    if grep -qx "mds_${tag}" <<< "$IN_FLIGHT"; then
      echo "-- ${tag}: already queued or running, not submitting"
      continue
    fi
    jid=$(sbatch --parsable \
        --job-name="mds_${tag}" \
        --export=ALL,ARM="${arm}",SEED="${seed}",DEPTH="${DEPTH}",TRAINER="${TRAINER}",REPO="${REPO}" \
        marker_experiments/downstream/cluster/run_one.sbatch)
    echo "-- ${tag}: job ${jid}"
  done
done
