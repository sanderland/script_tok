#!/usr/bin/env python3
"""Analyze per-pretoken token differences between FSP and BPE models on FineWiki."""

from pathlib import Path

from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    FINEWIKI_CORPUS_NAMES,
    DEFAULTS,
    get_model_path,
)

RESULTS_DIR = Path("results/hybrid")
UNIGRAM_RESULTS_DIR = Path("results/unigram_sweeps")


def analyze_pretoken_differences(corpus_name: str, fw_corpus_name: str, vocab_size: int = 32768):
    """
    Analyze per-pretoken token count differences between FSP and BPE.
    
    Returns list of (pretoken_text, freq, fsp_tokens, bpe_tokens, diff * freq) sorted by absolute weighted diff.
    """
    # Load FSP model
    fsp_params = {
        **DEFAULTS,
        "additional_vocab_size": vocab_size,
        "final_style_prune": True,
        "pre_final_vocab_factor": 1.0,
        "pruning_shrinking_factor": 0.75,
    }
    fsp_path = get_model_path(corpus_name, fsp_params)
    print(f"Loading FSP model from {fsp_path}")
    fsp_model = UnigramModel.load(str(fsp_path))
    
    # Load BPE model
    bpe_path = UNIGRAM_RESULTS_DIR / corpus_name / f"bpe_n{vocab_size}.model.json.gz"
    print(f"Loading BPE model from {bpe_path}")
    bpe_model = BPETokenizer.load(str(bpe_path))
    
    # Load FineWiki corpus
    print(f"Loading corpus {fw_corpus_name}...")
    corpus = load_corpus_by_name(fw_corpus_name, fsp_model.pretokenizer)
    
    # Collect per-pretoken stats
    results = []
    total_fsp_tokens = 0
    total_bpe_tokens = 0
    total_atomic = 0
    
    for atomic_tokens, count in corpus:
        text = fsp_model.pretokenizer.decode(atomic_tokens)
        
        # FSP: use viterbi path
        lattice = fsp_model.make_lattice(atomic_tokens)
        viterbi_path, _ = lattice.viterbi()
        fsp_len = len(viterbi_path)
        
        # BPE: encode directly
        bpe_ids = bpe_model.encode(text)
        bpe_len = len(bpe_ids)
        
        diff = fsp_len - bpe_len  # positive = FSP uses more tokens
        weighted_diff = diff * count
        
        total_fsp_tokens += fsp_len * count
        total_bpe_tokens += bpe_len * count
        total_atomic += len(atomic_tokens) * count
        
        results.append({
            "text": text,
            "atomic_len": len(atomic_tokens),
            "freq": count,
            "fsp_tokens": fsp_len,
            "bpe_tokens": bpe_len,
            "diff": diff,
            "weighted_diff": weighted_diff,
        })
    
    # Sort by absolute weighted diff (descending)
    results.sort(key=lambda x: abs(x["weighted_diff"]), reverse=True)
    
    print(f"\nTotal FSP tokens: {total_fsp_tokens:,}")
    print(f"Total BPE tokens: {total_bpe_tokens:,}")
    print(f"Difference: {total_fsp_tokens - total_bpe_tokens:,} ({(total_fsp_tokens - total_bpe_tokens) / total_bpe_tokens * 100:.2f}%)")
    print(f"Total atomic tokens: {total_atomic:,}")
    
    return results, {
        "total_fsp": total_fsp_tokens,
        "total_bpe": total_bpe_tokens,
        "total_atomic": total_atomic,
    }


def show_top_differences(results: list, n: int = 30):
    """Display top N pretokens by weighted difference."""
    print(f"\n{'='*100}")
    print(f"Top {n} pretokens by |weighted_diff| = |diff| * freq")
    print(f"{'='*100}")
    print(f"{'Pretoken':<40} {'Freq':>10} {'FSP':>6} {'BPE':>6} {'Diff':>6} {'Wtd Diff':>12}")
    print("-" * 100)
    
    for r in results[:n]:
        # Truncate text for display
        text_display = repr(r["text"])
        if len(text_display) > 38:
            text_display = text_display[:35] + "..."
        
        print(f"{text_display:<40} {r['freq']:>10,} {r['fsp_tokens']:>6} {r['bpe_tokens']:>6} {r['diff']:>+6} {r['weighted_diff']:>+12,}")


def show_tokenization_details(results: list, fsp_model, bpe_model, n: int = 10):
    """Show detailed tokenization for top N pretokens."""
    print(f"\n{'='*100}")
    print(f"Detailed tokenization for top {n} pretokens")
    print(f"{'='*100}")
    
    for r in results[:n]:
        text = r["text"]
        atomic_tokens = fsp_model.pretokenizer.pretokenize(text)[0]
        
        # FSP tokenization
        lattice = fsp_model.make_lattice(atomic_tokens)
        viterbi_path, _ = lattice.viterbi()
        fsp_tokens = [fsp_model.pretokenizer.decode(t.atomic_tokens) for t in viterbi_path]
        
        # BPE tokenization
        bpe_ids = bpe_model.encode(text)
        bpe_tokens = [bpe_model.decode([tid]) for tid in bpe_ids]
        
        print(f"\nPretoken: {repr(text)}")
        print(f"  Freq: {r['freq']:,}, FSP: {r['fsp_tokens']}, BPE: {r['bpe_tokens']}, Diff: {r['diff']:+}, Wtd: {r['weighted_diff']:+,}")
        print(f"  FSP tokens ({len(fsp_tokens)}): {fsp_tokens}")
        print(f"  BPE tokens ({len(bpe_tokens)}): {bpe_tokens}")


def analyze_default_vs_bpe(corpus_name: str, fw_corpus_name: str, vocab_size: int = 32768):
    """Compare Default Unigram to BPE on FineWiki."""
    # Load Default model
    default_params = {**DEFAULTS, "additional_vocab_size": vocab_size}
    default_path = get_model_path(corpus_name, default_params)
    print(f"Loading Default model from {default_path}")
    default_model = UnigramModel.load(str(default_path))
    
    # Load BPE model
    bpe_path = UNIGRAM_RESULTS_DIR / corpus_name / f"bpe_n{vocab_size}.model.json.gz"
    print(f"Loading BPE model from {bpe_path}")
    bpe_model = BPETokenizer.load(str(bpe_path))
    
    # Load FineWiki corpus
    print(f"Loading corpus {fw_corpus_name}...")
    corpus = load_corpus_by_name(fw_corpus_name, default_model.pretokenizer)
    
    # Collect per-pretoken stats
    results = []
    total_default_tokens = 0
    total_bpe_tokens = 0
    
    for atomic_tokens, count in corpus:
        text = default_model.pretokenizer.decode(atomic_tokens)
        
        # Default: use viterbi path
        lattice = default_model.make_lattice(atomic_tokens)
        viterbi_path, _ = lattice.viterbi()
        default_len = len(viterbi_path)
        
        # BPE: encode directly
        bpe_ids = bpe_model.encode(text)
        bpe_len = len(bpe_ids)
        
        diff = default_len - bpe_len
        weighted_diff = diff * count
        
        total_default_tokens += default_len * count
        total_bpe_tokens += bpe_len * count
        
        results.append({
            "text": text,
            "freq": count,
            "default_tokens": default_len,
            "bpe_tokens": bpe_len,
            "diff": diff,
            "weighted_diff": weighted_diff,
        })
    
    results.sort(key=lambda x: abs(x["weighted_diff"]), reverse=True)
    
    print(f"\nTotal Default tokens: {total_default_tokens:,}")
    print(f"Total BPE tokens: {total_bpe_tokens:,}")
    print(f"Difference: {total_default_tokens - total_bpe_tokens:,} ({(total_default_tokens - total_bpe_tokens) / total_bpe_tokens * 100:.2f}%)")
    
    return results, default_model, bpe_model


def main():
    # Analyze English by default
    corpus_idx = 0  # eng_latn_300mb
    corpus_name = CORPUS_NAMES[corpus_idx]
    fw_corpus_name = FINEWIKI_CORPUS_NAMES[corpus_idx]
    
    print(f"Analyzing {corpus_name} -> {fw_corpus_name}")
    print("\n" + "="*100)
    print("PART 1: FSP vs BPE")
    print("="*100)
    
    results, totals = analyze_pretoken_differences(corpus_name, fw_corpus_name)
    show_top_differences(results, n=50)
    
    # Load models for detailed view
    fsp_params = {
        **DEFAULTS,
        "additional_vocab_size": 32768,
        "final_style_prune": True,
        "pre_final_vocab_factor": 1.0,
        "pruning_shrinking_factor": 0.75,
    }
    fsp_path = get_model_path(corpus_name, fsp_params)
    fsp_model = UnigramModel.load(str(fsp_path))
    
    bpe_path = UNIGRAM_RESULTS_DIR / corpus_name / "bpe_n32768.model.json.gz"
    bpe_model = BPETokenizer.load(str(bpe_path))
    
    show_tokenization_details(results, fsp_model, bpe_model, n=20)
    
    print("\n" + "="*100)
    print("PART 2: Default Unigram vs BPE")
    print("="*100)
    
    results2, default_model, _ = analyze_default_vs_bpe(corpus_name, fw_corpus_name)
    
    print(f"\n{'='*100}")
    print(f"Top 50 pretokens by |weighted_diff| = |diff| * freq")
    print(f"{'='*100}")
    print(f"{'Pretoken':<40} {'Freq':>10} {'Def':>6} {'BPE':>6} {'Diff':>6} {'Wtd Diff':>12}")
    print("-" * 100)
    
    for r in results2[:50]:
        text_display = repr(r["text"])
        if len(text_display) > 38:
            text_display = text_display[:35] + "..."
        print(f"{text_display:<40} {r['freq']:>10,} {r['default_tokens']:>6} {r['bpe_tokens']:>6} {r['diff']:>+6} {r['weighted_diff']:>+12,}")


if __name__ == "__main__":
    main()

