#!/usr/bin/env bash
# Submit one language's downstream sweep: every arm x trainer x seed, four runs per node.
#
#     paper_utils/boundary/downstream/cluster/submit_language.sh --lang ko
#     paper_utils/boundary/downstream/cluster/submit_language.sh --lang ru --dry-run
#
# One job per (trainer, seed) holding that seed's four arms. Grouping this way means a
# finished job is a complete paired set for one seed, which is the unit the comparison
# comes in, rather than four seeds of one arm and nothing to pair them with.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO"

LANG_CODE=""
ARMS="plain,bnd_w,bnd_wpd,bnd_wpd_caps"
TRAINERS="bpe,mingram"
SEEDS="0,1,2"
ACCOUNT="a0229"
TAG_SUFFIX=""
PARTITION="normal"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang) LANG_CODE="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --trainers) TRAINERS="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --tag-suffix) TAG_SUFFIX="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$LANG_CODE" ]] || { echo "set --lang" >&2; exit 1; }

# Training shards per language, chosen so each language makes about as many passes over its
# training text as the English runs (about 3.5). A run consumes a fixed 653,568 rows of 2,048
# tokens, and nanochat's loader crops long documents when packing rows, so passes depend on
# how many rows a file yields, measured with the real loader: Korean 68,960 rows per file
# (3 shards, 3.11 passes observed), Russian 18,816 (10 shards, about 3.5 passes). English is
# the retrain of the published models: 8 ClimbMix shards, as they were.
# See DESIGN_CHOICES_MULTILANG.md for how these were arrived at.
declare -A SHARDS=( [ko]=3 [ru]=10 [en]=8 )
NUM_SHARDS="${SHARDS[$LANG_CODE]:-}"
[[ -n "$NUM_SHARDS" ]] || { echo "no shard count recorded for language '$LANG_CODE'" >&2; exit 1; }
# Tokenizer corpus per language. Korean and Russian use the quick-sample grid behind the
# intrinsic tables. English uses the full-sample tokenizers, because those are the ones the
# published English models were trained with.
declare -A CORPORA=( [ko]=fineweb_ko_5gb_quick [ru]=fineweb_ru_5gb_quick [en]=fineweb_en_5gb )
CORPUS="${CORPORA[$LANG_CODE]}"

DATA_ROOT="/capstor/scratch/cscs/${USER}/marker_downstream"
BASE="${DATA_ROOT}/nanochat_base_${LANG_CODE}"
# One output directory per trainer. collect_results.py refuses a log directory holding both
# trainers, because `arm` does not carry the trainer and the means would pool. The Korean and
# English sweeps of 2026-09-19 shared one directory, so every job's final collection failed
# and all 12 jobs were marked FAILED although all 48 runs had finished.
OUT_BASE="results/marker_downstream_${LANG_CODE}${TAG_SUFFIX}"

[[ -d "$BASE" ]] || { echo "no base directory ${BASE}; run build_language_shards.py first" >&2; exit 1; }

# Untracked files count. `git diff` alone reports a clean tree while the file the job runs is
# untracked, which is how a sweep once ran code that was in no commit.
DIRTY=$(git status --porcelain | wc -l)
COMMIT=$(git rev-parse HEAD)
echo "repo ${REPO} at ${COMMIT}, ${DIRTY} uncommitted change(s)"
if (( DIRTY > 0 )); then
  echo "note: the jobs record this commit, and these files are not in it:"
  git status --porcelain | sed 's/^/  /'
fi
echo "language=${LANG_CODE} corpus=${CORPUS} shards=${NUM_SHARDS} base=${BASE}"
echo "arms=${ARMS} trainers=${TRAINERS} seeds=${SEEDS} account=${ACCOUNT} partition=${PARTITION} tag_suffix='${TAG_SUFFIX}' out=${OUT_BASE}_<trainer>"

for trainer in ${TRAINERS//,/ }; do
  OUT="${OUT_BASE}_${trainer}"
  # sbatch refuses a job whose --output directory does not exist.
  [[ "$DRY_RUN" == "1" ]] || mkdir -p "${OUT}/slurm" "${OUT}/logs"
  for seed in ${SEEDS//,/ }; do
    runs=""
    for arm in ${ARMS//,/ }; do
      tok="paper_utils/boundary/downstream/tokenizers/${CORPUS}_${arm}_${trainer}_v34685.json.gz"
      [[ -f "$tok" ]] || { echo "missing tokenizer ${tok}" >&2; exit 1; }
      runs+="${arm}:${seed} "
    done
    name="ds_${LANG_CODE}${TAG_SUFFIX}_${trainer}_s${seed}"
    echo "  ${name}: ${runs}"
    [[ "$DRY_RUN" == "1" ]] && continue
    jid=$(sbatch --parsable \
      --account="${ACCOUNT}" --partition="${PARTITION}" \
      --job-name="${name}" \
      --output="${OUT}/slurm/${name}_%j.out" --error="${OUT}/slurm/${name}_%j.err" \
      --export=ALL,REPO="${REPO}",RUNS="${runs}",TRAINER="${trainer}",CORPUS="${CORPUS}",NUM_SHARDS="${NUM_SHARDS}",OUT="${OUT}",NANOCHAT_BASE_REQUESTED="${BASE}",PACK_TAG_SUFFIX="${TAG_SUFFIX}" \
      paper_utils/boundary/downstream/cluster/pack_runs.sbatch)
    echo "    submitted ${jid}"
  done
done
