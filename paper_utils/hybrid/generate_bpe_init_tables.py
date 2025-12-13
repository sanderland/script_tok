#!/usr/bin/env python3
"""Generate BPE-Init comparison tables with FineWiki validation and inference-time bias.

Compares:
- Default Unigram (baseline)
- Default + Inference Bias
- FSP (Final-Style Pruning)
- FSP + Inference Bias
- BPE-Init (best factor=1.1)
- BPE-Init + Inference Bias
- BPE (baseline)

Features:
- FineWiki 1GB validation with JSON caching
- Inference-time bias to encourage compression (no retraining needed)
- LaTeX tables with compression metrics and tokenization examples
- Per-language breakdown and mean summary
"""

import copy
import json
import math
from pathlib import Path

from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    DEFAULTS,
    ADDITIONAL_VOCAB_SIZE,
    get_model_path as get_unigram_model_path,
)
from paper_utils.unigram.utils import FINEWIKI_MAP, LANG_NAMES
from paper_utils.hybrid.train_hybrid import get_model_path as get_hybrid_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel

# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIR = Path("results/hybrid")
UNIGRAM_RESULTS_DIR = Path("results/unigram_sweeps")

# FSP configuration
FSP_PARAMS = {
    **DEFAULTS,
    "final_style_prune": True,
    "pre_final_vocab_factor": 1.0,
}

# BPE-Init configuration (best factor from analysis)
BPE_INIT_FACTOR = 1.1
BPE_INIT_PARAMS = {**DEFAULTS, "bpe_init_factor": BPE_INIT_FACTOR}

# Bias values to test
BIAS_VALUES = [0, 25, 50, 100]
DEFAULT_BIAS = 50  # Primary bias value for comparison tables

# Cache file
CACHE_FILE = RESULTS_DIR / "cache_bpe_init_finewiki.json"


# =============================================================================
# Inference-Time Bias Helper
# =============================================================================


def apply_token_bias(model: UnigramModel, bias: float) -> UnigramModel:
    """Apply inference-time bias by modifying token log probs.
    
    Creates a copy of the model to avoid modifying the original.
    Subtracting bias from log probs penalizes using more tokens,
    pushing Viterbi towards fewer, longer tokens.
    
    Args:
        model: Original UnigramModel
        bias: Value to subtract from each token's log_prob
        
    Returns:
        New model with biased log probs
    """
    if bias == 0:
        return model
    
    # Deep copy the model to avoid modifying original
    biased_model = copy.deepcopy(model)
    for token in biased_model.tokens.values():
        token.log_prob -= bias
    return biased_model


# =============================================================================
# Cache Management
# =============================================================================


def load_cache() -> dict:
    """Load cache from JSON file."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(data: dict):
    """Save cache to JSON file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def cache_key(corpus_name: str, method: str, bias: float) -> str:
    """Generate cache key."""
    return f"{corpus_name}/{method}/bias{bias}"


# =============================================================================
# Model Loading
# =============================================================================


def load_model_for_method(corpus_name: str, method: str):
    """Load model for a given method.
    
    Returns (model, path) tuple, or (None, None) if not found.
    """
    if method == "BPE":
        # BPE models are in unigram_sweeps
        path = UNIGRAM_RESULTS_DIR / corpus_name / f"bpe_n{ADDITIONAL_VOCAB_SIZE}.model.json.gz"
        if path.exists():
            return BPETokenizer.load(str(path)), path
        return None, None
    
    if method in ["Default", "Default+Bias"]:
        path = get_unigram_model_path(corpus_name, DEFAULTS)
    elif method in ["FSP", "FSP+Bias"]:
        path = get_unigram_model_path(corpus_name, FSP_PARAMS)
    elif method in ["BPE-Init", "BPE-Init+Bias"]:
        path = get_hybrid_model_path(corpus_name, BPE_INIT_PARAMS)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if path.exists():
        return UnigramModel.load(str(path)), path
    return None, None


# =============================================================================
# Corpus Evaluation
# =============================================================================


def evaluate_on_corpus(model, corpus_name: str, bias: float = 0.0) -> dict:
    """Evaluate model on a corpus with optional inference-time bias.
    
    Returns loss (objective) and compression metrics.
    """
    is_bpe = isinstance(model, BPETokenizer)
    
    # Apply bias for Unigram models
    if not is_bpe and bias != 0:
        model = apply_token_bias(model, bias)
    
    corpus = load_corpus_by_name(corpus_name, model.pretokenizer)

    objective = 0.0
    total_tokens = 0
    total_atomic_len = 0
    total_bytes = 0

    for atomic_tokens, count in corpus:
        text = model.pretokenizer.decode(atomic_tokens)
        chunk_bytes = len(text.encode("utf-8"))

        if is_bpe:
            token_ids = model.encode(text)
            total_tokens += len(token_ids) * count
        else:
            lattice = model.make_lattice(atomic_tokens)
            z, _ = lattice.calc_marginal()
            assert not math.isnan(z), f"NaN likelihood for pretoken {atomic_tokens}"
            viterbi_path, _ = lattice.viterbi()
            objective -= z * count
            total_tokens += len(viterbi_path) * count

        total_atomic_len += len(atomic_tokens) * count
        total_bytes += chunk_bytes * count

    if total_atomic_len > 0 and not is_bpe:
        objective /= total_atomic_len

    return {
        "objective": objective if not is_bpe else None,
        "tokens": total_tokens,
        "atomic_tokens": total_atomic_len,
        "bytes": total_bytes,
        "bytes_per_token": total_bytes / total_tokens if total_tokens > 0 else 0,
    }


def evaluate_cached(corpus_name: str, eval_corpus: str, method: str, bias: float, cache: dict) -> dict | None:
    """Evaluate with caching."""
    key = cache_key(eval_corpus, method, bias)
    if key in cache:
        print(f"  Cache hit: {key}")
        return cache[key]
    
    model, path = load_model_for_method(corpus_name, method)
    if model is None:
        print(f"  Model not found: {corpus_name}/{method}")
        return None
    
    print(f"  Evaluating: {key}...")
    result = evaluate_on_corpus(model, eval_corpus, bias)
    result["model_path"] = str(path)
    
    cache[key] = result
    return result


# =============================================================================
# Result Aggregation
# =============================================================================


def compute_mean_results(results_per_corpus: dict) -> dict:
    """Compute mean across corpus results."""
    objectives = []
    tokens = []
    bytes_total = []
    
    for result in results_per_corpus.values():
        if result is None:
            continue
        tokens.append(result["tokens"])
        bytes_total.append(result["bytes"])
        if result["objective"] is not None:
            objectives.append(result["objective"])
    
    if not tokens:
        return None
    
    return {
        "objective": sum(objectives) / len(objectives) if objectives else None,
        "tokens": sum(tokens) / len(tokens),
        "bytes": sum(bytes_total) / len(bytes_total),
        "n_corpora": len(tokens),
    }


# =============================================================================
# LaTeX Table Generation
# =============================================================================


def generate_comparison_table(all_results: dict, baseline_method: str = "Default") -> str:
    """Generate LaTeX table comparing methods.
    
    Shows relative change in tokens vs baseline for each language and mean.
    """
    methods = ["Default", f"Default+Bias({DEFAULT_BIAS})", "FSP", f"FSP+Bias({DEFAULT_BIAS})", 
               "BPE-Init", f"BPE-Init+Bias({DEFAULT_BIAS})", "BPE"]
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Compression comparison on FineWiki 1GB. Shows relative token count change (\%) vs Default Unigram baseline. Bias values encourage compression at inference time.}")
    lines.append(r"\label{tab:bpe-init-comparison}")
    
    # Column spec
    n_methods = len(methods)
    lines.append(rf"\begin{{tabular}}{{@{{}}l{'r' * n_methods}@{{}}}}")
    lines.append(r"\toprule")
    
    # Header
    method_headers = " & ".join(f"\\textbf{{{m}}}" for m in methods)
    lines.append(rf"Language & {method_headers} \\")
    lines.append(r"\midrule")
    
    # Per-language rows
    for corpus_name in CORPUS_NAMES:
        lang = LANG_NAMES.get(corpus_name, corpus_name[:3])
        finewiki = FINEWIKI_MAP.get(corpus_name)
        if finewiki is None:
            continue
        
        # Get baseline tokens
        baseline_key = (corpus_name, finewiki, baseline_method, 0)
        baseline_result = all_results.get(baseline_key)
        baseline_tokens = baseline_result["tokens"] if baseline_result else None
        
        row_values = []
        for method in methods:
            # Parse method name and bias
            if "+Bias" in method:
                base_method = method.split("+")[0]
                bias = DEFAULT_BIAS
            else:
                base_method = method
                bias = 0
            
            key = (corpus_name, finewiki, base_method, bias)
            result = all_results.get(key)
            
            if result is None or baseline_tokens is None:
                row_values.append("---")
            else:
                rel_change = (result["tokens"] - baseline_tokens) / baseline_tokens * 100
                if rel_change < -1:
                    row_values.append(f"\\textbf{{{rel_change:+.2f}}}")
                else:
                    row_values.append(f"{rel_change:+.2f}")
        
        lines.append(f"{lang} & " + " & ".join(row_values) + r" \\")
    
    lines.append(r"\midrule")
    
    # Mean row
    row_values = []
    baseline_mean = None
    for method in methods:
        if "+Bias" in method:
            base_method = method.split("+")[0]
            bias = DEFAULT_BIAS
        else:
            base_method = method
            bias = 0
        
        method_results = {}
        for corpus_name in CORPUS_NAMES:
            finewiki = FINEWIKI_MAP.get(corpus_name)
            if finewiki is None:
                continue
            key = (corpus_name, finewiki, base_method, bias)
            if key in all_results:
                method_results[corpus_name] = all_results[key]
        
        mean_result = compute_mean_results(method_results)
        
        if method == baseline_method:
            baseline_mean = mean_result["tokens"] if mean_result else None
        
        if mean_result is None or baseline_mean is None:
            row_values.append("---")
        else:
            rel_change = (mean_result["tokens"] - baseline_mean) / baseline_mean * 100
            if rel_change < -1:
                row_values.append(f"\\textbf{{{rel_change:+.2f}}}")
            else:
                row_values.append(f"{rel_change:+.2f}")
    
    lines.append(r"\textit{Mean} & " + " & ".join(row_values) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_bias_sweep_table(all_results: dict) -> str:
    """Generate table showing effect of different bias values."""
    methods = ["Default", "FSP", "BPE-Init"]
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Effect of inference-time bias on compression. Shows mean relative token count change (\%) vs unbiased version across all languages.}")
    lines.append(r"\label{tab:bias-sweep}")
    
    lines.append(rf"\begin{{tabular}}{{@{{}}l{'r' * len(BIAS_VALUES)}@{{}}}}")
    lines.append(r"\toprule")
    
    # Header
    bias_headers = " & ".join(f"\\textbf{{bias={b}}}" for b in BIAS_VALUES)
    lines.append(rf"Method & {bias_headers} \\")
    lines.append(r"\midrule")
    
    for method in methods:
        # Get baseline (bias=0) mean tokens
        baseline_results = {}
        for corpus_name in CORPUS_NAMES:
            finewiki = FINEWIKI_MAP.get(corpus_name)
            if finewiki is None:
                continue
            key = (corpus_name, finewiki, method, 0)
            if key in all_results:
                baseline_results[corpus_name] = all_results[key]
        
        baseline_mean = compute_mean_results(baseline_results)
        baseline_tokens = baseline_mean["tokens"] if baseline_mean else None
        
        row_values = []
        for bias in BIAS_VALUES:
            bias_results = {}
            for corpus_name in CORPUS_NAMES:
                finewiki = FINEWIKI_MAP.get(corpus_name)
                if finewiki is None:
                    continue
                key = (corpus_name, finewiki, method, bias)
                if key in all_results:
                    bias_results[corpus_name] = all_results[key]
            
            mean_result = compute_mean_results(bias_results)
            
            if mean_result is None or baseline_tokens is None:
                row_values.append("---")
            else:
                rel_change = (mean_result["tokens"] - baseline_tokens) / baseline_tokens * 100
                row_values.append(f"{rel_change:+.2f}")
        
        lines.append(f"{method} & " + " & ".join(row_values) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# =============================================================================
# Tokenization Examples
# =============================================================================


def escape_latex(s: str) -> str:
    """Escape LaTeX special characters."""
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#").replace("$", r"\$")


def collect_example_tokenizations(models: dict, sample_texts: list[str]) -> list[dict]:
    """Collect tokenizations of sample texts from all models."""
    examples = []
    
    for text in sample_texts:
        example = {"text": text, "tokenizations": {}}
        
        for method_name, model in models.items():
            if model is None:
                example["tokenizations"][method_name] = None
                continue
            
            if isinstance(model, BPETokenizer):
                tokens = model.encode(text, return_tokens=True)
                example["tokenizations"][method_name] = tokens
            else:
                # For Unigram, encode to get token strings
                atomic_tokens = model.pretokenizer.pretokenize(text)
                lattice = model.make_lattice(atomic_tokens)
                path, _ = lattice.viterbi()
                tokens = [model.pretokenizer.decode(t.atomic_tokens) for t in path]
                example["tokenizations"][method_name] = tokens
        
        examples.append(example)
    
    return examples


def generate_examples_table(examples: list[dict], methods: list[str]) -> str:
    """Generate LaTeX table with tokenization examples."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Tokenization examples comparing different methods. Vertical bars indicate token boundaries.}")
    lines.append(r"\label{tab:tokenization-examples}")
    
    # Dynamic column widths
    n_methods = len(methods)
    col_width = f"{12 // n_methods}cm"
    lines.append(rf"\begin{{tabular}}{{@{{}}l{'p{' + col_width + '}' * n_methods}@{{}}}}")
    lines.append(r"\toprule")
    
    # Header
    method_headers = " & ".join(f"\\textbf{{{m}}}" for m in methods)
    lines.append(rf"\textbf{{Text}} & {method_headers} \\")
    lines.append(r"\midrule")
    
    for example in examples:
        text = escape_latex(example["text"])
        row_values = []
        
        for method in methods:
            tokens = example["tokenizations"].get(method)
            if tokens is None:
                row_values.append("---")
            else:
                tok_str = " | ".join(escape_latex(t) for t in tokens)
                row_values.append(tok_str)
        
        lines.append(f"{text} & " + " & ".join(row_values) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate BPE-Init comparison tables."""
    print("=" * 80)
    print("BPE-INIT COMPARISON WITH FINEWIKI VALIDATION")
    print("=" * 80)
    print(f"Corpora: {CORPUS_NAMES}")
    print(f"Bias values: {BIAS_VALUES}")
    print(f"Default bias for comparison: {DEFAULT_BIAS}")
    print()
    
    # Load cache
    cache = load_cache()
    
    # Methods to evaluate (without bias suffix - bias handled separately)
    base_methods = ["Default", "FSP", "BPE-Init", "BPE"]
    
    # Collect all results
    all_results = {}
    
    print("\n--- FineWiki Evaluation ---")
    for corpus_name in CORPUS_NAMES:
        finewiki = FINEWIKI_MAP.get(corpus_name)
        if finewiki is None:
            print(f"⚠ No FineWiki mapping for {corpus_name}, skipping")
            continue
        
        print(f"\n{LANG_NAMES.get(corpus_name, corpus_name)}:")
        
        for method in base_methods:
            # BPE doesn't support bias
            if method == "BPE":
                biases = [0]
            else:
                biases = BIAS_VALUES
            
            for bias in biases:
                result = evaluate_cached(corpus_name, finewiki, method, bias, cache)
                if result:
                    all_results[(corpus_name, finewiki, method, bias)] = result
    
    # Save cache
    save_cache(cache)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nMean tokens across languages (FineWiki 1GB):")
    print(f"{'Method':<20} {'Bias':>6} {'Tokens (M)':>12} {'vs Default':>12}")
    print("-" * 55)
    
    # Get baseline
    baseline_results = {}
    for corpus_name in CORPUS_NAMES:
        finewiki = FINEWIKI_MAP.get(corpus_name)
        if finewiki:
            key = (corpus_name, finewiki, "Default", 0)
            if key in all_results:
                baseline_results[corpus_name] = all_results[key]
    baseline_mean = compute_mean_results(baseline_results)
    baseline_tokens = baseline_mean["tokens"] if baseline_mean else None
    
    for method in base_methods:
        biases = [0] if method == "BPE" else BIAS_VALUES
        for bias in biases:
            method_results = {}
            for corpus_name in CORPUS_NAMES:
                finewiki = FINEWIKI_MAP.get(corpus_name)
                if finewiki:
                    key = (corpus_name, finewiki, method, bias)
                    if key in all_results:
                        method_results[corpus_name] = all_results[key]
            
            mean_result = compute_mean_results(method_results)
            if mean_result and baseline_tokens:
                rel = (mean_result["tokens"] - baseline_tokens) / baseline_tokens * 100
                print(f"{method:<20} {bias:>6} {mean_result['tokens']/1e6:>12.2f} {rel:>+11.2f}%")
    
    # Generate tables
    print("\n--- Generating LaTeX Tables ---")
    
    comparison_table = generate_comparison_table(all_results)
    comparison_path = RESULTS_DIR / "table_bpe_init_comparison.tex"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(comparison_table)
    print(f"✓ Written: {comparison_path}")
    
    bias_table = generate_bias_sweep_table(all_results)
    bias_path = RESULTS_DIR / "table_bias_sweep.tex"
    bias_path.write_text(bias_table)
    print(f"✓ Written: {bias_path}")
    
    # Generate tokenization examples
    print("\n--- Generating Tokenization Examples ---")
    
    # Load models for English
    eng_corpus = "eng_latn_300mb"
    example_methods = ["Default", "FSP", "BPE-Init", "BPE"]
    models = {}
    
    for method in example_methods:
        model, _ = load_model_for_method(eng_corpus, method)
        models[method] = model
        # Also add biased version for Unigram
        if model and not isinstance(model, BPETokenizer):
            models[f"{method}+Bias"] = apply_token_bias(model, DEFAULT_BIAS)
    
    # Sample texts for examples
    sample_texts = [
        "understanding",
        "internationalization",
        "tokenization",
        "computational",
        "preprocessing",
        "acknowledgement",
        "unfortunately",
        "implementation",
    ]
    
    examples = collect_example_tokenizations(models, sample_texts)
    
    # Generate table with subset of methods for readability
    display_methods = ["Default", "Default+Bias", "BPE-Init", "BPE-Init+Bias", "BPE"]
    examples_table = generate_examples_table(examples, display_methods)
    examples_path = RESULTS_DIR / "table_bpe_init_examples.tex"
    examples_path.write_text(examples_table)
    print(f"✓ Written: {examples_path}")
    
    print("\n" + "=" * 80)
    print("Tables:")
    print(comparison_table)
    print()
    print(bias_table)
    print("=" * 80)


if __name__ == "__main__":
    main()

