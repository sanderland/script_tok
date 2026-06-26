#!/usr/bin/env bash
# Train every tokenizer the paper needs, then regenerate intrinsic figures and
# tables. Downstream LM tables render when their gitignored result bundles are
# present under results/downstream/.
#
# Headline training corpora: fineweb (mono per-lang + multilingual hybrid6).
# Finewiki and the *_300mb held-out corpora are eval-only -- the figures pull
# them on demand; no training happens on them.
#
# Methods (see train_model.py):
#   bpe, default, fsp              - no factor
#   bpe_init, bpe_init_fsp         - f sweep
#   mingram                        - f x p sweep
#   mingram + --prune-criterion mi - MinGram-PP, p=0.9, f sweep including f=8
# External trainers (own scripts, all skip-if-exists):
#   pathpiece_bpe                  - build_pathpiece.py
#   pathpiece_bpe f-sweep          - build_pathpiece_sweep.py
#   convextok                      - build_convextok.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN="${SCRIPT_DIR}/train_model.py"
BUILD_PATHPIECE="${SCRIPT_DIR}/build_pathpiece.py"
BUILD_PATHPIECE_SWEEP="${SCRIPT_DIR}/build_pathpiece_sweep.py"
BUILD_CONVEXTOK="${SCRIPT_DIR}/build_convextok.py"
DOWNSTREAM_DIR="${SCRIPT_DIR}/../../results/downstream"
DOWNSTREAM_RESULTS="${DOWNSTREAM_DIR}/d24_master_results.tsv"
TOKEN_USAGE_COUNTS="${DOWNSTREAM_DIR}/token_usage_counts.parquet"
GLITCH_CACHE="${DOWNSTREAM_DIR}/cache_glitch.json"
MAX_JOBS="${MAX_JOBS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# PathPiece and ConvexTok have large internal parallelism (worker pools / PDLP).
# Their per-job worker count is independent of the orchestrator's parallelism.
PATHPIECE_WORKERS="${PATHPIECE_WORKERS:-24}"
CONVEXTOK_WORKERS="${CONVEXTOK_WORKERS:-32}"

# Monolingual training corpora: fineweb only (en/de/fi have morphalign gold;
# ru/ar/ko are compression-only). Each is 5GB sampled from FineWeb / FineWeb-2.
MONOLINGUAL_CORPORA=(fineweb_en_5gb fineweb_de_5gb fineweb_fi_5gb fineweb_ru_5gb fineweb_ar_5gb fineweb_ko_5gb)
# Goldfish fish-food (~7-15GB/lang) — secondary headline-train corpus, mono only.
# Used in the appendix train-corpus comparison panels.
FISHFOOD_CORPORA=(eng_latn_fishfood deu_latn_fishfood fin_latn_fishfood rus_cyrl_fishfood arb_arab_fishfood kor_hang_fishfood)
# Note: FineWiki is eval-only (loaders work for held-out cross-domain evals);
# we do not train on it. The grid's finewiki panels read fineweb-trained tokenizers.
ALL_MONO_CORPORA=("${MONOLINGUAL_CORPORA[@]}" "${FISHFOOD_CORPORA[@]}")
MONOLINGUAL_OVERSHOOT_FACTORS=(1.0 1.05 1.1 1.15 1.25 1.5 2.0 3.0 5.0)
MONOLINGUAL_PRUNING_FACTORS=(0.0 0.9)
MINGRAM_PP_OVERSHOOT_FACTORS=(1.0 1.05 1.1 1.15 1.25 1.5 2.0 3.0 5.0 8.0)
MINGRAM_PP_PRUNING_FACTOR=0.9
# PathPiece appendix f-sweep uses init_vocab_size = round(f * 32768) + SCRIPT atoms.
# Keep these in the same order as MONOLINGUAL_OVERSHOOT_FACTORS.
PATHPIECE_SWEEP_INIT_VOCAB_SIZES=(34684 36322 37961 39599 42876 51068 67452 100220 165756)
# EM-iters ablation: now on fineweb (was finewiki).
EM_ABLATION_CORPORA=(fineweb_en_5gb fineweb_de_5gb fineweb_fi_5gb)
EM_ABLATION_ITERS=(0 1 3 4)

# Multilingual training corpus: fineweb:hybrid6 only.
# Vocab sweep removed (only n=32768 is used by current paper figures; larger
# vocab scaling is follow-up material, not for this submission).
MULTILINGUAL_CORPORA=(fineweb:hybrid6)
MULTILINGUAL_VOCAB_SIZES=(32768)
MULTILINGUAL_OVERSHOOT_FACTORS=(1.0 1.1 1.15 1.25 1.5 2.0 5.0)
MULTILINGUAL_PRUNING_FACTORS=(0.0)

WORKSPACE_TMP_DIR="${WORKSPACE_TMP_DIR:-${SCRIPT_DIR}/../../results/tmp}"
mkdir -p "$WORKSPACE_TMP_DIR"

BPE_JOB_FILE=$(mktemp "${WORKSPACE_TMP_DIR}/hybrid-bpe.XXXXXX")
NONBPE_JOB_FILE=$(mktemp "${WORKSPACE_TMP_DIR}/hybrid-nonbpe.XXXXXX")
trap 'rm -f "$BPE_JOB_FILE" "$NONBPE_JOB_FILE"' EXIT

# Pre-warm corpus encoding caches SERIALLY before launching the parallel block.
# load_corpus_by_name encodes on first access; without this warm-up, 32 parallel
# jobs all race to encode the same fresh corpus. After this loop each
# training corpus has a cached encoding under results/corpora/<name>/PT-<hash>/,
# so every later job hits the cache instantly.
# Eval-only corpora (finewiki, *_300mb, CulturaX) are warmed lazily by the
# figure-generation block at the end.
echo "=== Pre-warming corpus encoding caches ==="
for corpus in "${ALL_MONO_CORPORA[@]}" "${MULTILINGUAL_CORPORA[@]}"; do
    uv run python -c "
from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
load_corpus_by_name('$corpus', get_pretokenizer('scriptenc_cb'))
print('warm: $corpus')
"
done

for corpus in "${ALL_MONO_CORPORA[@]}"; do
    printf "uv run python %s --method bpe     --corpus %s --num-workers %s\n" "$TRAIN" "$corpus" "$NUM_WORKERS" >> "$BPE_JOB_FILE"
    printf "uv run python %s --method default --corpus %s --num-workers %s\n" "$TRAIN" "$corpus" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
    printf "uv run python %s --method fsp     --corpus %s --num-workers %s\n" "$TRAIN" "$corpus" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
    for f in "${MONOLINGUAL_OVERSHOOT_FACTORS[@]}"; do
        printf "uv run python %s --method bpe_init     --corpus %s --overshoot-factor %s --num-workers %s\n" "$TRAIN" "$corpus" "$f" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
        printf "uv run python %s --method bpe_init_fsp --corpus %s --overshoot-factor %s --num-workers %s\n" "$TRAIN" "$corpus" "$f" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
        for p in "${MONOLINGUAL_PRUNING_FACTORS[@]}"; do
            printf "uv run python %s --method mingram --corpus %s --overshoot-factor %s --pruning-shrinking-factor %s --num-workers %s\n" "$TRAIN" "$corpus" "$f" "$p" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
        done
    done
    for f in "${MINGRAM_PP_OVERSHOOT_FACTORS[@]}"; do
        printf "uv run python %s --method mingram --corpus %s --overshoot-factor %s --pruning-shrinking-factor %s --prune-criterion mi --num-workers %s\n" \
            "$TRAIN" "$corpus" "$f" "$MINGRAM_PP_PRUNING_FACTOR" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
    done
done

for corpus in "${MULTILINGUAL_CORPORA[@]}"; do
    for vocab_size in "${MULTILINGUAL_VOCAB_SIZES[@]}"; do
        printf "uv run python %s --method bpe     --corpus %s --additional-vocab-size %s --num-workers %s\n" \
            "$TRAIN" "$corpus" "$vocab_size" "$NUM_WORKERS" >> "$BPE_JOB_FILE"
        printf "uv run python %s --method default --corpus %s --additional-vocab-size %s --num-workers %s\n" \
            "$TRAIN" "$corpus" "$vocab_size" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
        printf "uv run python %s --method fsp     --corpus %s --additional-vocab-size %s --num-workers %s\n" \
            "$TRAIN" "$corpus" "$vocab_size" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"

        for f in "${MULTILINGUAL_OVERSHOOT_FACTORS[@]}"; do
            printf "uv run python %s --method bpe_init --corpus %s --overshoot-factor %s --additional-vocab-size %s --num-workers %s\n" \
                "$TRAIN" "$corpus" "$f" "$vocab_size" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
            printf "uv run python %s --method bpe_init_fsp --corpus %s --overshoot-factor %s --additional-vocab-size %s --num-workers %s\n" \
                "$TRAIN" "$corpus" "$f" "$vocab_size" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
            for p in "${MULTILINGUAL_PRUNING_FACTORS[@]}"; do
                printf "uv run python %s --method mingram --corpus %s --overshoot-factor %s --pruning-shrinking-factor %s --additional-vocab-size %s --num-workers %s\n" \
                    "$TRAIN" "$corpus" "$f" "$p" "$vocab_size" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
            done
        done
    done
done

for corpus in "${EM_ABLATION_CORPORA[@]}"; do
    for num_em in "${EM_ABLATION_ITERS[@]}"; do
        printf "uv run python %s --method mingram --corpus %s --overshoot-factor 1.15 --num-em-iterations %s --pruning-shrinking-factor 0.0 --num-workers %s\n" \
            "$TRAIN" "$corpus" "$num_em" "$NUM_WORKERS" >> "$NONBPE_JOB_FILE"
    done
done

# PathPiece (BPE-init, main config) + ConvexTok (mp200k LP) for all corpora the
# paper figures cover. Both build scripts skip if their target file exists.
for corpus in "${ALL_MONO_CORPORA[@]}" "${MULTILINGUAL_CORPORA[@]}"; do
    printf "uv run python %s %s %s\n" "$BUILD_PATHPIECE" "$corpus" "$PATHPIECE_WORKERS" >> "$NONBPE_JOB_FILE"
    printf "uv run python %s %s 200000 %s\n" "$BUILD_CONVEXTOK" "$corpus" "$CONVEXTOK_WORKERS" >> "$NONBPE_JOB_FILE"
done

# Appendix f-sweep overlay for PathPiece-BPE. The plot uses FineWeb-trained
# monolingual models evaluated on Goldfish held-out corpora.
for corpus in "${MONOLINGUAL_CORPORA[@]}"; do
    for init_vocab_size in "${PATHPIECE_SWEEP_INIT_VOCAB_SIZES[@]}"; do
        printf "uv run python %s %s %s %s\n" "$BUILD_PATHPIECE_SWEEP" "$corpus" "$init_vocab_size" "$PATHPIECE_WORKERS" >> "$NONBPE_JOB_FILE"
    done
done

echo "=== Running $(wc -l < "$BPE_JOB_FILE") BPE training jobs ==="
echo "max_jobs=${MAX_JOBS} train_workers=${NUM_WORKERS}"
parallel --jobs "$MAX_JOBS" --line-buffer < "$BPE_JOB_FILE"

echo ""
echo "=== Running $(wc -l < "$NONBPE_JOB_FILE") non-BPE training jobs ==="
echo "max_jobs=${MAX_JOBS} train_workers=${NUM_WORKERS}"
parallel --jobs "$MAX_JOBS" --line-buffer < "$NONBPE_JOB_FILE"

# Generate outputs ────────────────────────────────────────────────────────────
echo ""
echo "=== Generating paper figures and tables ==="
uv run python "${SCRIPT_DIR}/generate_train_eval_compression_grid.py"
uv run python "${SCRIPT_DIR}/run_bpe_init_fsweep.py"
uv run python "${SCRIPT_DIR}/run_tiebreak_ablation.py"
uv run python "${SCRIPT_DIR}/generate_morphalign_scatter.py"

uv run python "${SCRIPT_DIR}/generate_main_combined_table.py"
uv run python "${SCRIPT_DIR}/generate_morphalign_table.py"
uv run python "${SCRIPT_DIR}/generate_tokenization_examples_table.py"
uv run python "${SCRIPT_DIR}/generate_train_corpus_table.py"
uv run python "${SCRIPT_DIR}/generate_em_ablation_table.py"
uv run python "${SCRIPT_DIR}/generate_pruning_schedule_table.py"
uv run python "${SCRIPT_DIR}/generate_tiebreak_table.py"
uv run python "${SCRIPT_DIR}/generate_renyi_entropy_table.py"

if [[ ! -f "$TOKEN_USAGE_COUNTS" ]]; then
    uv run python "${SCRIPT_DIR}/build_token_usage_counts.py"
fi
uv run python "${SCRIPT_DIR}/generate_undertrained_token_examples_table.py"

if [[ -f "$DOWNSTREAM_RESULTS" ]]; then
    uv run python "${SCRIPT_DIR}/generate_downstream_table.py"
    uv run python "${SCRIPT_DIR}/generate_downstream_seed_table.py"
    uv run python "${SCRIPT_DIR}/generate_downstream_task_grid.py"
else
    echo "Skipping downstream tables: missing ${DOWNSTREAM_RESULTS}"
fi

if [[ -f "$GLITCH_CACHE" ]]; then
    uv run python "${SCRIPT_DIR}/generate_glitch_table.py"
else
    echo "Skipping glitch table: missing ${GLITCH_CACHE}"
fi

uv run python "${SCRIPT_DIR}/generate_lead_scatter.py"
uv run python "${SCRIPT_DIR}/plot_bpe_init_fsweep_variants.py"
uv run python "${SCRIPT_DIR}/generate_morphalign_2d.py"
uv run python "${SCRIPT_DIR}/plot_train_corpus_summary_bars.py"
uv run python "${SCRIPT_DIR}/generate_domain_shift_figure.py"
uv run python "${SCRIPT_DIR}/generate_train_eval_compression_grid.py" --all-panels

echo "Done. See results/hybrid/, results/mingram/, and results/mingram_paper/."
