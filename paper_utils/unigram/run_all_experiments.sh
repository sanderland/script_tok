#!/usr/bin/env bash
# Run all unigram experiments with maximal granularity using GNU parallel
# Each corpus is processed individually for better parallelization and error isolation

set -euo pipefail

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_hyperparameters.py"

# Parallel execution settings
MAX_JOBS="${MAX_JOBS:-32}"  # Maximum number of parallel jobs (override with env var)

# All corpus sets
CORPUS_300MB=(
    "eng_latn_300mb"
    "deu_latn_300mb"
    "arb_arab_300mb"
    "hin_deva_300mb"
    "kor_hang_300mb"
    "zho_hans_300mb"
)

CORPUS_30MB=(
    "smol_eng_latn_300mb"
    "smol_deu_latn_300mb"
    "smol_arb_arab_300mb"
    "smol_hin_deva_300mb"
    "smol_kor_hang_300mb"
    "smol_zho_hans_300mb"
)

CORPUS_FINEWIKI=(
    "finewiki_en_1gb"
    "finewiki_de_1gb"
    "finewiki_ar_1gb"
    "finewiki_hi_1gb"
    "finewiki_ko_1gb"
    "finewiki_zh_1gb"
)

# All experiment types
EXPERIMENTS=(
    "bpe"
    "initial_vocab_factor"
    "m_step_digamma"
    "m_step_low_count_threshold"
    "num_sub_iterations"
    "pre_final_vocab_factor"
    "pruning_shrinking_factor"
    "additional_vocab_size"
    "init_algo"
    "fsp"
    "fsp_vocab"
)

# Parse arguments
MODE="${1:-all}"  # all, 300mb, 30mb, finewiki, or specific experiment name

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $*"
}

run_experiment_for_corpus() {
    local experiment="$1"
    local corpus="$2"
    local finewiki_flag="${3:-}"

    log "Running ${experiment} for ${corpus}"

    if [[ -n "$finewiki_flag" ]]; then
        if uv run python "$TRAIN_SCRIPT" "$experiment" --corpus-filter "$corpus" --finewiki; then
            log_success "${experiment} completed for ${corpus}"
            return 0
        else
            log_error "${experiment} failed for ${corpus}"
            return 1
        fi
    else
        if uv run python "$TRAIN_SCRIPT" "$experiment" --corpus-filter "$corpus"; then
            log_success "${experiment} completed for ${corpus}"
            return 0
        else
            log_error "${experiment} failed for ${corpus}"
            return 1
        fi
    fi
}

generate_job_list() {
    local mode="$1"
    local job_list_file="$2"

    > "$job_list_file"  # Clear file

    case "$mode" in
        all|300mb)
            # Generate all experiment x corpus combinations for 300MB
            for corpus in "${CORPUS_300MB[@]}"; do
                for experiment in "${EXPERIMENTS[@]}"; do
                    echo "$experiment::$corpus::" >> "$job_list_file"
                done
            done
            ;;& # Continue to next pattern
        all|30mb)
            # Generate init_algo jobs for 30MB corpora
            for corpus in "${CORPUS_30MB[@]}"; do
                echo "init_algo::$corpus::" >> "$job_list_file"
            done
            ;;& # Continue to next pattern
        all|finewiki)
            # Generate all experiment x corpus combinations for FineWiki (except init_algo)
            for corpus in "${CORPUS_FINEWIKI[@]}"; do
                for experiment in "${EXPERIMENTS[@]}"; do
                    if [[ "$experiment" != "init_algo" ]]; then
                        echo "$experiment::$corpus::--finewiki" >> "$job_list_file"
                    fi
                done
            done
            ;;
        *)
            # Specific experiment mode
            if [[ " ${EXPERIMENTS[*]} " =~ " ${mode} " ]]; then
                # Run on 300MB
                for corpus in "${CORPUS_300MB[@]}"; do
                    echo "$mode::$corpus::" >> "$job_list_file"
                done

                # Run on 30MB for init_algo
                if [[ "$mode" == "init_algo" ]]; then
                    for corpus in "${CORPUS_30MB[@]}"; do
                        echo "init_algo::$corpus::" >> "$job_list_file"
                    done
                fi

                # Run on FineWiki (except init_algo)
                if [[ "$mode" != "init_algo" ]]; then
                    for corpus in "${CORPUS_FINEWIKI[@]}"; do
                        echo "$mode::$corpus::--finewiki" >> "$job_list_file"
                    done
                fi
            fi
            ;;
    esac
}

run_job() {
    local job_spec="$1"

    IFS='::' read -r experiment corpus finewiki_flag <<< "$job_spec"

    log "Running ${experiment} for ${corpus}"

    local cmd="uv run python \"$TRAIN_SCRIPT\" \"$experiment\" --corpus-filter \"$corpus\""
    if [[ -n "$finewiki_flag" ]]; then
        cmd="$cmd $finewiki_flag"
    fi

    if eval "$cmd"; then
        log_success "${experiment} completed for ${corpus}"
        return 0
    else
        log_error "${experiment} failed for ${corpus}"
        return 1
    fi
}

export -f run_job
export -f log
export -f log_success
export -f log_error
export TRAIN_SCRIPT RED GREEN YELLOW BLUE NC

# Main execution
echo "================================================================"
echo "Unigram Experiment Runner - Maximal Granularity Mode with GNU Parallel"
echo "================================================================"
log "Mode: ${MODE}"
log "Script: ${TRAIN_SCRIPT}"
log "Max parallel jobs: ${MAX_JOBS}"
echo ""

# Generate job list
JOB_LIST_FILE=$(mktemp)
trap "rm -f $JOB_LIST_FILE" EXIT

case "$MODE" in
    all|300mb|30mb|finewiki)
        log "Generating job list for mode: ${MODE}"
        generate_job_list "$MODE" "$JOB_LIST_FILE"
        ;;
    *)
        # Run a specific experiment across all applicable corpora
        if [[ " ${EXPERIMENTS[*]} " =~ " ${MODE} " ]]; then
            log "Generating job list for experiment: ${MODE}"
            generate_job_list "$MODE" "$JOB_LIST_FILE"
        else
            log_error "Unknown mode or experiment: ${MODE}"
            echo ""
            echo "Usage: $0 [MODE]"
            echo ""
            echo "Available modes:"
            echo "  all       - Run all experiments on all corpora (default)"
            echo "  300mb     - Run all experiments on 300MB corpora only"
            echo "  30mb      - Run init_algo on 30MB corpora only"
            echo "  finewiki  - Run all experiments on FineWiki corpora only"
            echo ""
            echo "Available experiments (run specific experiment across all corpora):"
            for exp in "${EXPERIMENTS[@]}"; do
                echo "  $exp"
            done
            echo ""
            echo "Environment variables:"
            echo "  MAX_JOBS  - Maximum parallel jobs (default: 4)"
            echo "              Example: MAX_JOBS=8 $0 all"
            exit 1
        fi
        ;;
esac

# Count total jobs
TOTAL_JOBS=$(wc -l < "$JOB_LIST_FILE")
log "Total jobs to run: ${TOTAL_JOBS}"

if [[ $TOTAL_JOBS -eq 0 ]]; then
    log_warn "No jobs to run!"
    exit 0
fi

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
    log_error "GNU parallel is not installed!"
    echo "Please install it with: sudo apt-get install parallel"
    echo "Or on macOS: brew install parallel"
    exit 1
fi

# Run jobs with GNU parallel
log "Starting parallel execution with ${MAX_JOBS} concurrent jobs..."
echo ""

parallel --jobs "$MAX_JOBS" --colsep '::' --progress --joblog "parallel_${MODE}_$(date +%Y%m%d_%H%M%S).log" \
    run_job {1}::{2}::{3} :::: "$JOB_LIST_FILE"

PARALLEL_EXIT=$?

echo ""
echo "================================================================"
if [[ $PARALLEL_EXIT -eq 0 ]]; then
    log_success "All requested experiments completed successfully!"
else
    log_warn "Some experiments may have failed (exit code: $PARALLEL_EXIT)"
    echo "Check the joblog file for details."
fi
echo "================================================================"

exit $PARALLEL_EXIT
