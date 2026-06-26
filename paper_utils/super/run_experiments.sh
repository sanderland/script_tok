#!/bin/bash
# Run supertoken experiment grid
#
# Usage:
#   bash paper_utils/super/run_experiments.sh
#
# Experiments:
# 1. Filter comparison: Test semantic vs all vs individual patterns
# 2. Vocab ratio: VA = VB vs VA = 2*VB
# 3. FSP: With and without flat-score pruning

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Default corpus
CORPUS="eng_latn_300mb"

# Vocabulary sizes
VB=32000
VA_SAME=$VB          # VA = VB
VA_DOUBLE=$((VB * 2)) # VA = 2*VB

# Filters to test
# Main comparison: all (baseline) vs semantic (clean) vs all_patterns
MAIN_FILTERS=("all" "semantic" "all_patterns")
# Individual pattern ablation
PATTERN_FILTERS=("space_phrase" "punct_trans" "contraction" "suffix" "abbrev" "hyphen")

# N-gram sizes to test
NGRAMS=(4 8)  # Focus on 4 and 8, skip 2

echo "=========================================="
echo "Supertoken Experiments"
echo "=========================================="
echo "Corpus: $CORPUS"
echo "VB: $VB"
echo "Main filters: ${MAIN_FILTERS[*]}"
echo "Pattern filters: ${PATTERN_FILTERS[*]}"
echo "N-grams: ${NGRAMS[*]}"
echo ""

# Experiment 1: Main filter comparison (VA=2*VB)
echo "=== Experiment 1: Main Filter Comparison (VA=2*VB) ==="
for filter in "${MAIN_FILTERS[@]}"; do
    for ngram in "${NGRAMS[@]}"; do
        echo "Training: filter=$filter, max_ngram=$ngram, VA=$VA_DOUBLE"
        uv run python -m paper_utils.super.train_supertokens \
            --corpus-a "$CORPUS" \
            --vocab-a "$VA_DOUBLE" \
            --vocab-b "$VB" \
            --max-ngram "$ngram" \
            --filter-name "$filter"
    done
done

# Experiment 2: Individual pattern ablation
echo ""
echo "=== Experiment 2: Individual Pattern Ablation ==="
for filter in "${PATTERN_FILTERS[@]}"; do
    echo "Training: filter=$filter, max_ngram=4, VA=$VA_DOUBLE"
    uv run python -m paper_utils.super.train_supertokens \
        --corpus-a "$CORPUS" \
        --vocab-a "$VA_DOUBLE" \
        --vocab-b "$VB" \
        --max-ngram 4 \
        --filter-name "$filter"
done

# Experiment 3: VA = VB comparison (main filters only)
echo ""
echo "=== Experiment 3: VA = VB Comparison ==="
for filter in "${MAIN_FILTERS[@]}"; do
    echo "Training: filter=$filter, max_ngram=4, VA=$VA_SAME"
    uv run python -m paper_utils.super.train_supertokens \
        --corpus-a "$CORPUS" \
        --vocab-a "$VA_SAME" \
        --vocab-b "$VB" \
        --max-ngram 4 \
        --filter-name "$filter"
done

# Experiment 4: FSP comparison
echo ""
echo "=== Experiment 4: Flat-Score Pruning ==="
for filter in "all" "semantic"; do
    echo "Training with FSP: filter=$filter, max_ngram=4"
    uv run python -m paper_utils.super.train_supertokens \
        --corpus-a "$CORPUS" \
        --vocab-a "$VA_DOUBLE" \
        --vocab-b "$VB" \
        --max-ngram 4 \
        --filter-name "$filter" \
        --fsp
done

echo ""
echo "=========================================="
echo "Training complete! Generating results..."
echo "=========================================="

# Generate results
uv run python paper_utils/super/generate_results.py

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "Results saved to: results/supertoken_experiments/"
echo "=========================================="

