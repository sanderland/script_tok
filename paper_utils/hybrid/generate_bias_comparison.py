#!/usr/bin/env python3
"""Generate comparison table for token bias experiments.

Compares:
- baseline (default unigram)
- fsp (final style prune)
- bias 2 (token_bias=2.0, no FSP)
- fsp bias 2 (token_bias=2.0 + FSP)
- bpe

Evaluated on both 300MB training corpus and FineWiki held-out data.
"""

from pathlib import Path

from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    DEFAULTS,
    ADDITIONAL_VOCAB_SIZE,
    get_model_path as get_unigram_model_path,
)
from paper_utils.unigram.utils import (
    FINEWIKI_MAP,
    LANG_NAMES,
    evaluate_on_corpus_cached,
)
from paper_utils.hybrid.train_hybrid import get_model_path as get_hybrid_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel

RESULTS_DIR = Path("results/hybrid")
UNIGRAM_RESULTS_DIR = Path("results/unigram_sweeps")

# Model configurations
FSP_PARAMS = {
    **DEFAULTS,
    "final_style_prune": True,
    "pre_final_vocab_factor": 1.0,
    "pruning_shrinking_factor": 0.75,
}

BIAS2_PARAMS = {
    **DEFAULTS,
    "token_bias": 2.0,
}

FSP_BIAS2_PARAMS = {
    **DEFAULTS,
    "final_style_prune": True,
    "pre_final_vocab_factor": 1.0,
    "pruning_shrinking_factor": 0.75,
    "token_bias": 2.0,
}


def get_model_paths(corpus_name: str) -> dict[str, Path | None]:
    """Get paths to all model variants for a corpus."""
    paths = {}

    # Baseline (default) - from unigram sweeps
    path = get_unigram_model_path(corpus_name, DEFAULTS)
    paths["baseline"] = path if path.exists() else None

    # FSP - from unigram sweeps
    path = get_unigram_model_path(corpus_name, FSP_PARAMS)
    paths["fsp"] = path if path.exists() else None

    # Bias 2 (no FSP) - from hybrid
    path = get_hybrid_model_path(corpus_name, BIAS2_PARAMS)
    paths["bias2"] = path if path.exists() else None

    # FSP + Bias 2 - from hybrid
    path = get_hybrid_model_path(corpus_name, FSP_BIAS2_PARAMS)
    paths["fsp_bias2"] = path if path.exists() else None

    # BPE - from unigram sweeps
    bpe_path = UNIGRAM_RESULTS_DIR / corpus_name / f"bpe_n{ADDITIONAL_VOCAB_SIZE}.model.json.gz"
    paths["bpe"] = bpe_path if bpe_path.exists() else None

    return paths


def load_model(path: Path):
    """Load model from path."""
    if "bpe" in path.name:
        return BPETokenizer.load(str(path))
    return UnigramModel.load(str(path))


def evaluate_model(model, corpus_name: str, model_path: Path) -> dict:
    """Evaluate model on corpus with caching."""
    return evaluate_on_corpus_cached(model, corpus_name, str(model_path))


def generate_comparison_table(corpora: list[str]) -> str:
    """Generate comparison table."""
    print("\n" + "=" * 100)
    print("BIAS COMPARISON: baseline vs fsp vs bias2 vs fsp+bias2 vs bpe")
    print("=" * 100)

    methods = ["baseline", "fsp", "bias2", "fsp_bias2", "bpe"]
    method_names = {
        "baseline": "Baseline",
        "fsp": "FSP",
        "bias2": "Bias=2",
        "fsp_bias2": "FSP+Bias=2",
        "bpe": "BPE",
    }

    # Pre-load corpora
    print("\nPre-loading corpora...")
    first_paths = get_model_paths(corpora[0])
    first_path = next((p for p in first_paths.values() if p), None)
    if not first_path:
        raise ValueError("No models found!")
    
    pretokenizer = load_model(first_path).pretokenizer
    
    all_corpus_names = set()
    for corpus_name in corpora:
        all_corpus_names.add(corpus_name)
        if FINEWIKI_MAP.get(corpus_name):
            all_corpus_names.add(FINEWIKI_MAP[corpus_name])
    
    for cname in sorted(all_corpus_names):
        print(f"  Loading {cname}...")
        load_corpus_by_name(cname, pretokenizer)

    # Collect results
    all_results = {}
    
    for corpus_name in corpora:
        print(f"\n{corpus_name}:")
        finewiki_name = FINEWIKI_MAP.get(corpus_name)
        model_paths = get_model_paths(corpus_name)
        
        all_results[corpus_name] = {"300mb": {}, "finewiki": {}}
        
        for method, path in model_paths.items():
            if path is None:
                print(f"  {method}: NOT FOUND")
                continue
            
            print(f"  {method}: {path.name}")
            model = load_model(path)
            
            # Evaluate on 300MB
            result_300mb = evaluate_model(model, corpus_name, path)
            all_results[corpus_name]["300mb"][method] = result_300mb
            
            # Evaluate on FineWiki
            if finewiki_name:
                result_fw = evaluate_model(model, finewiki_name, path)
                all_results[corpus_name]["finewiki"][method] = result_fw

    # Print table
    print("\n" + "=" * 140)
    print("RESULTS: Bytes per Token (higher = better compression)")
    print("=" * 140)
    
    # Header
    header = f"{'Language':<12} {'Corpus':<10}"
    for method in methods:
        header += f" {method_names[method]:>12}"
    print(header)
    print("-" * 140)
    
    for corpus_name in corpora:
        if corpus_name not in all_results:
            continue
        
        lang = LANG_NAMES.get(corpus_name, corpus_name[:8])
        results = all_results[corpus_name]
        
        for corpus_type, corpus_label in [("300mb", "300MB"), ("finewiki", "FineWiki")]:
            row = f"{lang if corpus_type == '300mb' else '':<12} {corpus_label:<10}"
            
            for method in methods:
                r = results[corpus_type].get(method, {})
                if r:
                    bpt = r["bytes"] / r["tokens"] if r["tokens"] else 0
                    row += f" {bpt:>12.3f}"
                else:
                    row += f" {'---':>12}"
            
            print(row)
        print()

    # Print relative to baseline
    print("\n" + "=" * 140)
    print("RESULTS: Relative to Baseline (% change in B/Tok, positive = better)")
    print("=" * 140)
    
    header = f"{'Language':<12} {'Corpus':<10}"
    for method in methods[1:]:  # Skip baseline
        header += f" {method_names[method]:>12}"
    print(header)
    print("-" * 140)
    
    for corpus_name in corpora:
        if corpus_name not in all_results:
            continue
        
        lang = LANG_NAMES.get(corpus_name, corpus_name[:8])
        results = all_results[corpus_name]
        
        for corpus_type, corpus_label in [("300mb", "300MB"), ("finewiki", "FineWiki")]:
            row = f"{lang if corpus_type == '300mb' else '':<12} {corpus_label:<10}"
            
            baseline = results[corpus_type].get("baseline", {})
            baseline_bpt = baseline["bytes"] / baseline["tokens"] if baseline.get("tokens") else 0
            
            for method in methods[1:]:  # Skip baseline
                r = results[corpus_type].get(method, {})
                if r and baseline_bpt:
                    bpt = r["bytes"] / r["tokens"] if r["tokens"] else 0
                    delta = (bpt / baseline_bpt - 1) * 100
                    row += f" {delta:>+11.2f}%"
                else:
                    row += f" {'---':>12}"
            
            print(row)
        print()

    return ""


def main():
    """Run comparison."""
    corpora = CORPUS_NAMES
    generate_comparison_table(corpora)


if __name__ == "__main__":
    main()

