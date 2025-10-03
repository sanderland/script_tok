#!/bin/bash

# Run superscript experiments with single_token_span in parallel
# This runs all filter/ngram combinations simultaneously

set -e  # Exit on error

cd /fsx/sander/script_bpe

echo "=========================================="
echo "Running Superscript Experiments (single_token_span)"
echo "Running all combinations in parallel..."
echo "=========================================="
echo ""

# Run each filter/ngram combination in parallel
uv run python superscript.py --filters "all" --max-ngrams "2" --single-token-span &
uv run python superscript.py --filters "all" --max-ngrams "4" --single-token-span &
uv run python superscript.py --filters "all" --max-ngrams "8" --single-token-span &

uv run python superscript.py --filters "words" --max-ngrams "2" --single-token-span &
uv run python superscript.py --filters "words" --max-ngrams "4" --single-token-span &
uv run python superscript.py --filters "words" --max-ngrams "8" --single-token-span &

uv run python superscript.py --filters "words_nocomma" --max-ngrams "2" --single-token-span &
uv run python superscript.py --filters "words_nocomma" --max-ngrams "4" --single-token-span &
uv run python superscript.py --filters "words_nocomma" --max-ngrams "8" --single-token-span &

uv run python superscript.py --filters "len_8c" --max-ngrams "2" --single-token-span &
uv run python superscript.py --filters "len_8c" --max-ngrams "4" --single-token-span &
uv run python superscript.py --filters "len_8c" --max-ngrams "8" --single-token-span &

uv run python superscript.py --filters "len_16c" --max-ngrams "2" --single-token-span &
uv run python superscript.py --filters "len_16c" --max-ngrams "4" --single-token-span &
uv run python superscript.py --filters "len_16c" --max-ngrams "8" --single-token-span &

# Wait for all background jobs to complete
wait

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="