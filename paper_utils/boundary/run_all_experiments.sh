#!/usr/bin/env bash
# Reproduce the boundary-marker paper: the two main tables and their appendix companions.
#
#   paper_utils/boundary/paper/generated/table_intrinsic_main.tex     compression + MorphScore
#   paper_utils/boundary/paper/generated/table_downstream_main.tex    bits per byte
#   paper_utils/boundary/paper/generated/table_intrinsic_quick.tex    per-language appendix
#   paper_utils/boundary/paper/generated/table_vocab_duplicates.tex   duplicate vocabulary entries
#   paper_utils/boundary/paper/generated/downstream_appendix.tex      seed/shard robustness
#   paper_utils/boundary/paper/generated/table_example.tex            worked pre-tokenization
#
# The default run regenerates every table from the JSON and TSV caches committed under
# paper/generated/. It needs no GPU, no trained tokenizer and about a second: those caches
# are tracked precisely so the paper reproduces on any machine.
#
#     paper_utils/boundary/run_all_experiments.sh
#
# Everything upstream of the caches is opt-in, because it is expensive:
#
#     GRID=1        retrain the tokenizer grid and remeasure it (hours to days, CPU)
#     DOWNSTREAM=1  rerun the LM sweep (many GPU-hours; exits with a message if no CUDA)
#     FORCE=1       ignore cached measurements instead of extending them
#
# Every step is skip-if-exists, so an interrupted run resumes where it stopped and a rerun
# costs only what is missing.
#
# The grid trains on the `quick` corpus sample -- read-until-the-budget-is-full rather than
# reservoir-sampled over the whole source. It is non-uniform, and it is what the paper
# reports: uniform sampling costs roughly a day per language for a measurement that the
# `--quick` comparison showed does not move the ordering.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO"

DOWN="${SCRIPT_DIR}/downstream"
GENERATED="${SCRIPT_DIR}/paper/generated"

LANGS=${LANGS:-en,de,fi,ru,ar,ko}
TRAINERS=${TRAINERS:-bpe,mingram}
SEEDS=${SEEDS:-0,1,2}
DEPTH=${DEPTH:-12}
GRID=${GRID:-0}
DOWNSTREAM=${DOWNSTREAM:-0}
FORCE=${FORCE:-0}
force_flag=()
[[ "$FORCE" == "1" ]] && force_flag=(--force)

step() { echo; echo "== $*"; }

if [[ "$GRID" == "1" ]]; then
  step "1/5 tokenizer grid: ${LANGS} x every arm x ${TRAINERS}, quick sample, vocab 34,685"
  # Skips any cell whose tokenizer file already exists, so this is also the resume path.
  # One language at a time: the first cell of a language pays the corpus scan and every
  # later cell reuses the cache.
  for lg in ${LANGS//,/ }; do
    uv run python "${DOWN}/train_multilang.py" --lang "$lg" --trainers "$TRAINERS" --quick
  done

  step "2/5 manifest: fold the per-cell fragments into manifest.json"
  uv run python "${DOWN}/merge_manifests.py" --parts "${GENERATED}/manifest_parts/*.json"

  step "3/5 measurements: compression and MorphScore"
  # Each caches per cell and only computes what is missing, so a grid that grew by one arm
  # costs one arm. FORCE=1 recomputes everything.
  uv run python "${DOWN}/eval_goldfish.py" "${force_flag[@]}"
  uv run python "${SCRIPT_DIR}/morphscore_boundary.py" "${force_flag[@]}"
else
  step "1-3/5 grid: skipped, using the committed measurements (GRID=1 to rebuild)"
fi

if [[ "$DOWNSTREAM" == "1" ]]; then
  step "4/5 downstream LM sweep: seeds ${SEEDS}, depth ${DEPTH} (needs GPUs)"
  # run_arms.sh exits with a useful message if CUDA is missing. One trainer per invocation:
  # make_tex_tables refuses to pool trainers, since `arm` alone does not distinguish them.
  ARMS=plain,bnd_w,bnd_wpd,bnd_wpd_caps SEEDS="$SEEDS" DEPTH="$DEPTH" TRAINER=bpe \
    OUT=results/marker_downstream "${DOWN}/run_arms.sh"
  ARMS=plain,bnd_wpd SEEDS="$SEEDS" DEPTH="$DEPTH" TRAINER=mingram \
    OUT=results/marker_downstream_mingram SKIP_PAPER_ARTIFACTS=1 "${DOWN}/run_arms.sh"
  cp results/marker_downstream_mingram/results.tsv "${GENERATED}/results_mingram.tsv"
else
  step "4/5 downstream: skipped, using the committed run TSVs (DOWNSTREAM=1 to rerun)"
fi

step "5/5 tables"
# Pre-tokenizer output only, so it needs neither a trained tokenizer nor the caches.
uv run python "${SCRIPT_DIR}/make_example_table.py"                   # table_example
uv run python "${SCRIPT_DIR}/make_intrinsic_table.py" --layout mean   # table_intrinsic_main
uv run python "${SCRIPT_DIR}/make_intrinsic_table.py"                 # table_intrinsic_quick
# Counts entries in the trained vocabularies, so it reads the grid where GRID=1 built one
# and its committed cache everywhere else. New cells are measured and cached; old ones are
# not remeasured, and FORCE=1 does not reach here -- pass --force to recount.
uv run python "${SCRIPT_DIR}/vocab_duplicates.py"                     # table_vocab_duplicates
uv run python "${DOWN}/make_tex_tables.py"                            # downstream main + appendix

echo
echo "== done. Artifacts under paper_utils/boundary/paper/generated:"
(cd "$GENERATED" && ls -1 ./*.tex 2>/dev/null) | sed "s#^\./#   #"
