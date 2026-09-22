#!/usr/bin/env bash
# Several downstream runs concurrently on ONE node, one GPU each.
#
#     REPO=$PWD RUNS="plain:0 bnd_wpd:0" TRAINER=mingram CORPUS=fineweb_ko_5gb_quick \
#       NUM_SHARDS=3 OUT=results/marker_downstream_ko \
#       NANOCHAT_BASE_REQUESTED=/capstor/.../nanochat_base_ko \
#       sbatch --account=a0229 paper_utils/boundary/downstream/cluster/pack_runs.sbatch
#
# Submit through submit_language.sh, which sets all of those.
#
# Partition `normal` is OverSubscribe=EXCLUSIVE, so a job holds all 4 GH200s and all 288
# cores whether or not it uses them, and a node-hour bills as 4 GPU-hours. One run per node
# therefore pays for 4 GPUs to drive 1. Each run here gets its own GPU through
# CUDA_VISIBLE_DEVICES and its own share of the cores through ENCODE_WORKERS.
#
# Packing is cost-neutral in the limit where wall-clock scales perfectly with cores; it pays
# to the extent a single run leaves cores idle, and the GPU-bound part leaves them idle by
# construction. Measured in August: 4 runs on one node finished in 2h01m.
#
# GPUS stays 1, as in the English runs these are compared against: --gpus 4 would change the
# DDP world size, the effective batch and the data order.
#
#SBATCH --job-name=packruns
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

set -euo pipefail

# Required, with no path fallback. Slurm copies the batch script into its spool directory
# before running it, so BASH_SOURCE[0] here is that copy, and deriving the repo from it
# would give a wrong path.
: "${REPO:?set REPO to the checkout this job should run}"
cd "${REPO}"

# Captured BEFORE env.sh, which supplies a default for it.
: "${NANOCHAT_BASE_REQUESTED:?set NANOCHAT_BASE_REQUESTED to the base directory for this language}"
export NANOCHAT_BASE="${NANOCHAT_BASE_REQUESTED}"
source paper_utils/boundary/downstream/cluster/env.sh
# Pin imports to THIS checkout: the venv installs script_bpe and pynanochat as editable
# pointers into the main checkout, so a job launched from a worktree would otherwise run that
# tree's library code on whatever branch it happens to be on. Both entries are needed.
export PYTHONPATH="${REPO}:${REPO}/eval/py-nanochat:${PYTHONPATH:-}"

: "${RUNS:?set RUNS, space separated arm:seed pairs}"
# Required, not defaulted. run_arms.sh defaults TRAINER to bpe, and a mingram sweep that
# inherited that default would find the bpe runs already logged and exit reporting success
# having trained nothing.
: "${TRAINER:?set TRAINER explicitly, bpe or mingram}"
# Required for the same reason, and more so here than in the English runs: the corpus names
# the language of the tokenizers, and its default is English.
: "${CORPUS:?set CORPUS explicitly, e.g. fineweb_ko_5gb_quick}"
# Required: how much text a language gets is chosen per language, and run_arms.sh's default
# of 8 belongs to the English runs.
: "${NUM_SHARDS:?set NUM_SHARDS explicitly}"
: "${OUT:?set OUT}"

[[ -d "${NANOCHAT_BASE}" ]] || { echo "no such NANOCHAT_BASE: ${NANOCHAT_BASE}" >&2; exit 1; }

# The tokenizers and the training text are now language-specific and independently chosen, so
# pairing Korean tokenizers with Russian text is one mistyped variable away and would produce
# a complete, plausible, meaningless set of numbers. build_language_shards.py records the
# language beside the shards; require it to match the corpus name.
PROV="${NANOCHAT_BASE}/shard_provenance.json"
[[ -f "${PROV}" ]] || { echo "no shard_provenance.json under ${NANOCHAT_BASE}" >&2; exit 1; }
DATA_LANG=$(uv run --no-project python -c "import json,sys; print(json.load(open(sys.argv[1]))['language'])" "${PROV}")
case "${CORPUS}" in
  *_"${DATA_LANG}"_*) ;;
  *) echo "refusing: corpus ${CORPUS} against ${DATA_LANG} training text in ${NANOCHAT_BASE}" >&2; exit 1 ;;
esac
SHARDS_PRESENT=$(find "${NANOCHAT_BASE}/base_data_climbmix" -name 'shard_*.parquet' | wc -l)
if (( SHARDS_PRESENT != NUM_SHARDS + 1 )); then
  echo "refusing: ${SHARDS_PRESENT} shard(s) present, ${NUM_SHARDS} training + 1 validation declared." >&2
  echo "nanochat trains on every file but the last, so the declared count must match." >&2
  exit 1
fi

DEPTH="${DEPTH:-12}"
VOCAB="${VOCAB:-34685}"
# Pinned, not inherited. --export=ALL carries the submitting shell's environment, so a
# SMOKE=1 left over from a pipeline check would make every run here train 20 iterations, tag
# itself _smoke, and still print a result block that collect_results.py parses into a row.
export SMOKE=0
# The same protection for the tag suffix, which a sweep may set on purpose (the English
# retrain uses _retrain, so its rows can never pool with the published ones). It is read
# from a name nothing else sets, never from an inherited TAG_SUFFIX.
export TAG_SUFFIX="${PACK_TAG_SUFFIX:-}"

NRUNS=$(wc -w <<< "${RUNS}")
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if (( NRUNS > NGPU )); then
  echo "refusing: ${NRUNS} runs on a node with ${NGPU} GPUs. Two runs on one GPU would" >&2
  echo "contend for it and neither would be a clean single-GPU replicate." >&2
  exit 1
fi

CORES=$(nproc)
# Leave headroom so the concurrent parents are not fighting for the last core.
PER_RUN_WORKERS="${PER_RUN_WORKERS:-$(( (CORES - 8) / NRUNS ))}"
# Asserted, because 0 is not an error anywhere downstream: bootstrap.py creates no pool when
# the count is not above 1, so the run encodes serially with nothing in the log saying so.
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
echo "[pack] trainer=${TRAINER} corpus=${CORPUS} depth=${DEPTH} vocab=${VOCAB} shards=${NUM_SHARDS}"
echo "[pack] data_language=${DATA_LANG} base=${NANOCHAT_BASE} out=${OUT} tag_suffix='${TAG_SUFFIX}'"
echo "[pack] ${NRUNS} run(s) at ${PER_RUN_WORKERS} encode workers each: ${RUNS}"
mkdir -p "${OUT}/slurm" "${OUT}/logs"
cp "${PROV}" "${OUT}/slurm/shard_provenance_${DATA_LANG}.json"
# The commit above does not describe the code when the tree is dirty, and a pending job runs
# whatever is in the tree when it starts, not when it was submitted. Snapshot the difference
# here, at start, so every job's record holds the code that actually ran.
SNAP="${OUT}/slurm/code_${SLURM_JOB_ID:-nojob}"
mkdir -p "${SNAP}"
git diff HEAD > "${SNAP}/tracked.diff"
git ls-files --others --exclude-standard -z | tar --null -T - -czf "${SNAP}/untracked.tgz"
uv pip freeze > "${OUT}/slurm/${SLURM_JOB_ID:-nojob}.freeze" 2>/dev/null || true

pids=() names=()
gpu=0
for run in ${RUNS}; do
  arm="${run%%:*}"; seed="${run##*:}"
  tok="paper_utils/boundary/downstream/tokenizers/${CORPUS}_${arm}_${TRAINER}_v${VOCAB}.json.gz"
  # Assigned, not interpolated into the echo: a command substitution inside an echo argument
  # cannot trip `set -e`, so an unreadable tokenizer would print an empty hash and the group
  # would start anyway with no usable provenance for that run.
  TOK_SHA=$(sha256sum "${tok}" | cut -d" " -f1)
  echo "[pack]   ${arm}/s${seed} -> gpu ${gpu}, tokenizer_sha256=${TOK_SHA}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export ARMS="${arm}" SEEDS="${seed}"
    export TRAINER DEPTH CORPUS VOCAB NUM_SHARDS OUT NANOCHAT_BASE
    export GPUS=1
    export ENCODE_WORKERS="${PER_RUN_WORKERS}" TRAIN_WORKERS="${PER_RUN_WORKERS}"
    # run_arms.sh ends by copying its TSV over paper/generated/ and regenerating the tables.
    # Those artifacts are tracked and carry the English numbers; a per-language sweep's TSV
    # would overwrite them.
    export SKIP_PAPER_ARTIFACTS=1
    paper_utils/boundary/downstream/run_arms.sh
  ) > "${OUT}/slurm/pack_${arm}_s${seed}_${SLURM_JOB_ID:-nojob}.out" 2>&1 &
  pids+=($!) names+=("${arm}/s${seed}")
  gpu=$(( gpu + 1 ))
done

# Report every failure rather than dying on the first, so one bad run does not hide the rest
# of the node's work.
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
# either writes, so either one's file can be missing the other's row.
uv run python paper_utils/boundary/downstream/collect_results.py \
    --logs-dir "${OUT}/logs" --out "${OUT}/results.tsv" || fail=1

# Copy this job's final model weights off scratch, which deletes files after 14 days
# without access and did exactly that to every published English checkpoint. Only this
# job's own runs, so two jobs finishing together never copy the same one. A run that failed
# above has no result block and is skipped with a message rather than failing this step.
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/capstor/store/cscs/swissai/a0229/cmeister/script_tok_boundary/downstream_models}"
tag_args=()
for run in ${RUNS}; do
  tag_args+=(--tags "${run%%:*}_${TRAINER}_d${DEPTH}_s${run##*:}${TAG_SUFFIX}")
done
if ! uv run python -m paper_utils.boundary.downstream.archive_checkpoints \
    --logs-dir "${OUT}/logs" --corpus "${CORPUS}" \
    --dest "${ARCHIVE_ROOT}/$(basename "${OUT}")" "${tag_args[@]}"; then
  echo "[pack] ARCHIVE FAILED: this job's checkpoints exist only on scratch, which is purged" >&2
  fail=1
fi
echo "[pack] done, fail=${fail}"
exit "${fail}"
