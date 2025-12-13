#!/usr/bin/env python3
"""Generate bidirectional cross-generalization table: FineWiki <-> 300MB.

This script compares generalization in BOTH directions:
1. Train on 300MB -> Evaluate on FineWiki (in-domain -> out-of-domain)
2. Train on FineWiki -> Evaluate on 300MB (out-of-domain -> in-domain)

The goal is to understand whether models generalize equally well in both directions.
"""

from pathlib import Path

import pandas as pd

from script_bpe.analysis import evaluate_on_corpus, MorphScore
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    FINEWIKI_CORPUS_NAMES,
    DEFAULTS,
    load_model_if_cached,
    get_model_path,
)
from paper_utils.unigram.utils import evaluate_on_corpus_cached, evaluate_morphscore_cached

RESULTS_DIR = Path("results/unigram_sweeps")

# Configuration
# Match sweep output types: 0.5*32768=16384.0 (float), 1*32768=32768 (int), 2*32768=65536 (int)
VOCAB_SIZES = [16384.0, 32768, 65536]
VOCAB_LABELS = {16384.0: "16K", 32768: "32K", 65536: "64K"}
METHODS = ["Default", "FSP", "BPE"]

# Corpus pairing: 300MB <-> FineWiki
CORPUS_PAIRS = list(zip(CORPUS_NAMES, FINEWIKI_CORPUS_NAMES))


def check_experiment_status():
    """Check which experiments have been run and which are missing.

    Prints detailed status of:
    - 300MB -> FineWiki (models trained on 300MB, evaluated on FineWiki)
    - FineWiki -> 300MB (models trained on FineWiki, evaluated on 300MB)
    
    Returns:
        bool: True if all models exist, False if any are missing
    """
    print("=" * 80)
    print("EXPERIMENT STATUS CHECK")
    print("=" * 80)

    missing = []
    
    for corpus_300mb, corpus_fw in CORPUS_PAIRS:
        lang_name = corpus_300mb.split("_")[0]  # e.g., "eng", "deu", etc.
        print(f"\n{'-' * 80}")
        print(f"Language: {lang_name.upper()} ({corpus_300mb} <-> {corpus_fw})")
        print(f"{'-' * 80}")

        for method in METHODS:
            print(f"\n  {method}:")

            for vocab_size in VOCAB_SIZES:
                # Check 300MB -> FineWiki (models trained on 300MB corpus)
                if method == "Default":
                    params = {**DEFAULTS, "additional_vocab_size": vocab_size}
                    path_300mb = get_model_path(corpus_300mb, params)
                    exists_300mb = path_300mb.exists()
                elif method == "FSP":
                    params = {
                        **DEFAULTS,
                        "additional_vocab_size": vocab_size,
                        "final_style_prune": True,
                        "pre_final_vocab_factor": 1.0,
                        "pruning_shrinking_factor": 0.75,
                    }
                    path_300mb = get_model_path(corpus_300mb, params)
                    exists_300mb = path_300mb.exists()
                elif method == "BPE":
                    path_300mb = RESULTS_DIR / corpus_300mb / f"bpe_n{vocab_size}.model.json.gz"
                    exists_300mb = path_300mb.exists()

                # Check FineWiki -> 300MB (models trained on FineWiki corpus)
                if method == "Default":
                    params = {**DEFAULTS, "additional_vocab_size": vocab_size}
                    path_fw = get_model_path(corpus_fw, params)
                    exists_fw = path_fw.exists()
                elif method == "FSP":
                    params = {
                        **DEFAULTS,
                        "additional_vocab_size": vocab_size,
                        "final_style_prune": True,
                        "pre_final_vocab_factor": 1.0,
                        "pruning_shrinking_factor": 0.75,
                    }
                    path_fw = get_model_path(corpus_fw, params)
                    exists_fw = path_fw.exists()
                elif method == "BPE":
                    path_fw = RESULTS_DIR / corpus_fw / f"bpe_n{vocab_size}.model.json.gz"
                    exists_fw = path_fw.exists()

                status_300mb = "YES" if exists_300mb else "NO "
                status_fw = "YES" if exists_fw else "NO "

                print(f"    {VOCAB_LABELS[vocab_size]:>4}: "
                      f"300MB->FW {status_300mb}  |  FW->300MB {status_fw}")
                
                # Track missing models
                if not exists_300mb:
                    missing.append(f"{lang_name.upper()} {method} {VOCAB_LABELS[vocab_size]} 300MB->FW")
                if not exists_fw:
                    missing.append(f"{lang_name.upper()} {method} {VOCAB_LABELS[vocab_size]} FW->300MB")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    
    if missing:
        print(f"❌ INCOMPLETE: {len(missing)} models missing")
        print("\nMissing models:")
        for m in missing:
            print(f"  - {m}")
        print("\nTo train missing models:")
        print("  300MB models: uv run python paper_utils/unigram/train_hyperparameters.py <experiment>")
        print("  FineWiki models: uv run python paper_utils/unigram/train_hyperparameters.py <experiment> --finewiki")
    else:
        print("✓ COMPLETE: All models are available")
    
    print("=" * 80)
    
    return len(missing) == 0


def evaluate_cross_generalization(model, model_path, train_corpus, eval_corpus, ms):
    """Evaluate model on both training and cross-generalization corpus.

    Args:
        model: Trained tokenizer model
        model_path: Path to model file (for caching)
        train_corpus: Corpus the model was trained on
        eval_corpus: Cross-generalization corpus to evaluate on
        ms: MorphScore instance

    Returns:
        dict with 'train' and 'eval' metrics
    """
    # 1. Training corpus metrics (from metadata if available)
    train_obj = None
    train_tok = None

    if "performance" in model.metadata and model.metadata.get("corpus") == train_corpus:
        perf = model.metadata["performance"]
        train_obj = perf.get("objective")
        train_tok = perf.get("total_tokens_len")

    if train_obj is None or train_tok is None:
        print(f"  [uncached] evaluate_on_corpus: {train_corpus}")
        res = evaluate_on_corpus(model, train_corpus)
        train_obj = res["objective"]
        train_tok = res["tokens"]

    # 2. Cross-generalization corpus metrics (cached)
    res_eval = evaluate_on_corpus_cached(model, eval_corpus, str(model_path))
    eval_obj = res_eval["objective"]
    eval_tok = res_eval["tokens"]

    results = {
        "train": {"objective": train_obj, "tokens": train_tok},
        "eval": {"objective": eval_obj, "tokens": eval_tok}
    }

    # 3. MorphScore evaluation (only for English)
    lang_code = train_corpus[:8]  # e.g., "eng_latn"
    if lang_code == "eng_latn":
        ms_results = evaluate_morphscore_cached(ms, model, lang_code, str(model_path))
        ms_score = ms_results["morphscore_recall"]
        results["train"]["morph_recall"] = ms_score

    return results


def generate_bidirectional_table() -> str:
    """Generate bidirectional cross-generalization table.

    Shows tokenization performance for:
    - Row 1-3: Train on 300MB, evaluate on 300MB and FineWiki
    - Row 4-6: Train on FineWiki, evaluate on FineWiki and 300MB

    This reveals whether generalization is symmetric.
    """
    print("\n" + "=" * 70)
    print("GENERATING TABLE: Bidirectional Cross-Generalization")
    print("=" * 70)

    # Load MorphScore
    print("Loading MorphScore...")
    ms = MorphScore()

    # Gather data
    # Structure: results[train_corpus][method][vocab_size] = {train: {}, eval: {}}
    results = {}

    for corpus_300mb, corpus_fw in CORPUS_PAIRS:
        # Initialize both directions
        results[corpus_300mb] = {m: {} for m in METHODS}
        results[corpus_fw] = {m: {} for m in METHODS}

        print(f"\nProcessing {corpus_300mb} <-> {corpus_fw}...")

        for vocab_size in VOCAB_SIZES:
            # ========== 300MB -> FineWiki ==========
            print(f"  [{corpus_300mb}] {vocab_size}...")

            # Default
            params = {**DEFAULTS, "additional_vocab_size": vocab_size}
            model, path = load_model_if_cached(corpus_300mb, params)
            if model:
                results[corpus_300mb]["Default"][vocab_size] = evaluate_cross_generalization(
                    model, path, corpus_300mb, corpus_fw, ms
                )

            # FSP
            params_fsp = {
                **DEFAULTS,
                "additional_vocab_size": vocab_size,
                "final_style_prune": True,
                "pre_final_vocab_factor": 1.0,
                "pruning_shrinking_factor": 0.75,
            }
            path_fsp = get_model_path(corpus_300mb, params_fsp)
            if path_fsp.exists():
                model = UnigramModel.load(str(path_fsp))
                results[corpus_300mb]["FSP"][vocab_size] = evaluate_cross_generalization(
                    model, path_fsp, corpus_300mb, corpus_fw, ms
                )

            # BPE
            path_bpe = RESULTS_DIR / corpus_300mb / f"bpe_n{vocab_size}.model.json.gz"
            if path_bpe.exists():
                model = BPETokenizer.load(str(path_bpe))
                results[corpus_300mb]["BPE"][vocab_size] = evaluate_cross_generalization(
                    model, path_bpe, corpus_300mb, corpus_fw, ms
                )

            # ========== FineWiki -> 300MB ==========
            print(f"  [{corpus_fw}] {vocab_size}...")

            # Default
            params = {**DEFAULTS, "additional_vocab_size": vocab_size}
            model, path = load_model_if_cached(corpus_fw, params)
            if model:
                results[corpus_fw]["Default"][vocab_size] = evaluate_cross_generalization(
                    model, path, corpus_fw, corpus_300mb, ms
                )

            # FSP
            params_fsp = {
                **DEFAULTS,
                "additional_vocab_size": vocab_size,
                "final_style_prune": True,
                "pre_final_vocab_factor": 1.0,
                "pruning_shrinking_factor": 0.75,
            }
            path_fsp = get_model_path(corpus_fw, params_fsp)
            if path_fsp.exists():
                model = UnigramModel.load(str(path_fsp))
                results[corpus_fw]["FSP"][vocab_size] = evaluate_cross_generalization(
                    model, path_fsp, corpus_fw, corpus_300mb, ms
                )

            # BPE
            path_bpe = RESULTS_DIR / corpus_fw / f"bpe_n{vocab_size}.model.json.gz"
            if path_bpe.exists():
                model = BPETokenizer.load(str(path_bpe))
                results[corpus_fw]["BPE"][vocab_size] = evaluate_cross_generalization(
                    model, path_bpe, corpus_fw, corpus_300mb, ms
                )

    # Helper to compute means across languages
    def get_means(train_set, method, vocab_size, eval_type, metric_key):
        """Get mean metric across all languages.

        Args:
            train_set: "300mb" or "finewiki" - which corpus set models were trained on
            method: "Default", "FSP", or "BPE"
            vocab_size: Vocabulary size
            eval_type: "train" or "eval" - which corpus to evaluate on
            metric_key: "tokens" or "objective"
        """
        values = []
        for corpus_300mb, corpus_fw in CORPUS_PAIRS:
            train_corpus = corpus_300mb if train_set == "300mb" else corpus_fw
            res = results.get(train_corpus, {}).get(method, {}).get(vocab_size)
            if res and res.get(eval_type):
                val = res[eval_type].get(metric_key)
                if val is not None:
                    values.append(val)

        return sum(values) / len(values) if values else None

    # Helper for English-only morph score
    def get_english_morph(train_set, method, vocab_size):
        """Get MorphScore for English only.

        Args:
            train_set: "300mb" or "finewiki"
        """
        train_corpus = "eng_latn_300mb" if train_set == "300mb" else "finewiki_en_1gb"
        res = results.get(train_corpus, {}).get(method, {}).get(vocab_size)
        if res and res.get("train"):
            return res["train"].get("morph_recall")
        return None

    # Helper to format tokens with relative change
    def format_tok(val, baseline):
        if val is None or baseline is None:
            return "---"
        if val == baseline:
            return f"{val/1e6:.1f}\\phantom{{\\relchange{{+0.0}}}}"
        rel = (val - baseline) / baseline * 100
        sign = "+" if rel >= 0 else ""
        return f"{val/1e6:.1f}\\relchange{{{sign}{rel:.1f}}}"

    # Build LaTeX table
    lines = []
    lines.append(r"\begin{table}[pt]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Train} & \textbf{Voc} & \textbf{Train Tok.} & \textbf{Eval Tok.} & \textbf{Morph.} \\")
    lines.append(r"\midrule")

    # ========== Section 1: Train on 300MB ==========
    lines.append(r"\multicolumn{5}{@{}l}{\textit{\textbf{Train on 300MB, Evaluate on FineWiki}}} \\")
    lines.append(r"\addlinespace[2pt]")

    # Default (baseline for 300MB training)
    for vocab_size in VOCAB_SIZES:
        train_tok = get_means("300mb", "Default", vocab_size, "train", "tokens")
        eval_tok = get_means("300mb", "Default", vocab_size, "eval", "tokens")
        morph = get_english_morph("300mb", "Default", vocab_size)

        tok_train = f"{train_tok/1e6:.1f}\\phantom{{\\relchange{{+0.0}}}}" if train_tok else "---"
        tok_eval = f"{eval_tok/1e6:.1f}\\phantom{{\\relchange{{+0.0}}}}" if eval_tok else "---"
        morph_str = f"{morph:.3f}" if morph else "---"

        prefix = "Default" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_train} & {tok_eval} & {morph_str} \\\\")

    lines.append(r"\addlinespace[1pt]")

    # FSP
    for vocab_size in VOCAB_SIZES:
        train_tok = get_means("300mb", "FSP", vocab_size, "train", "tokens")
        eval_tok = get_means("300mb", "FSP", vocab_size, "eval", "tokens")
        base_train = get_means("300mb", "Default", vocab_size, "train", "tokens")
        base_eval = get_means("300mb", "Default", vocab_size, "eval", "tokens")
        morph = get_english_morph("300mb", "FSP", vocab_size)

        tok_train = format_tok(train_tok, base_train)
        tok_eval = format_tok(eval_tok, base_eval)
        morph_str = f"{morph:.3f}" if morph else "---"

        prefix = "FSP" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_train} & {tok_eval} & {morph_str} \\\\")

    lines.append(r"\addlinespace[1pt]")

    # BPE
    for vocab_size in VOCAB_SIZES:
        train_tok = get_means("300mb", "BPE", vocab_size, "train", "tokens")
        eval_tok = get_means("300mb", "BPE", vocab_size, "eval", "tokens")
        base_train = get_means("300mb", "Default", vocab_size, "train", "tokens")
        base_eval = get_means("300mb", "Default", vocab_size, "eval", "tokens")
        morph = get_english_morph("300mb", "BPE", vocab_size)

        tok_train = format_tok(train_tok, base_train)
        tok_eval = format_tok(eval_tok, base_eval)
        morph_str = f"{morph:.3f}" if morph else "---"

        prefix = "BPE" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_train} & {tok_eval} & {morph_str} \\\\")

    # ========== Section 2: Train on FineWiki ==========
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\textit{\textbf{Train on FineWiki, Evaluate on 300MB}}} \\")
    lines.append(r"\addlinespace[2pt]")

    # Default (baseline for FineWiki training)
    for vocab_size in VOCAB_SIZES:
        train_tok = get_means("finewiki", "Default", vocab_size, "train", "tokens")
        eval_tok = get_means("finewiki", "Default", vocab_size, "eval", "tokens")
        morph = get_english_morph("finewiki", "Default", vocab_size)

        tok_train = f"{train_tok/1e6:.1f}\\phantom{{\\relchange{{+0.0}}}}" if train_tok else "---"
        tok_eval = f"{eval_tok/1e6:.1f}\\phantom{{\\relchange{{+0.0}}}}" if eval_tok else "---"
        morph_str = f"{morph:.3f}" if morph else "---"

        prefix = "Default" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_train} & {tok_eval} & {morph_str} \\\\")

    lines.append(r"\addlinespace[1pt]")

    # FSP
    for vocab_size in VOCAB_SIZES:
        train_tok = get_means("finewiki", "FSP", vocab_size, "train", "tokens")
        eval_tok = get_means("finewiki", "FSP", vocab_size, "eval", "tokens")
        base_train = get_means("finewiki", "Default", vocab_size, "train", "tokens")
        base_eval = get_means("finewiki", "Default", vocab_size, "eval", "tokens")
        morph = get_english_morph("finewiki", "FSP", vocab_size)

        tok_train = format_tok(train_tok, base_train)
        tok_eval = format_tok(eval_tok, base_eval)
        morph_str = f"{morph:.3f}" if morph else "---"

        prefix = "FSP" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_train} & {tok_eval} & {morph_str} \\\\")

    lines.append(r"\addlinespace[1pt]")

    # BPE
    for vocab_size in VOCAB_SIZES:
        train_tok = get_means("finewiki", "BPE", vocab_size, "train", "tokens")
        eval_tok = get_means("finewiki", "BPE", vocab_size, "eval", "tokens")
        base_train = get_means("finewiki", "Default", vocab_size, "train", "tokens")
        base_eval = get_means("finewiki", "Default", vocab_size, "eval", "tokens")
        morph = get_english_morph("finewiki", "BPE", vocab_size)

        tok_train = format_tok(train_tok, base_train)
        tok_eval = format_tok(eval_tok, base_eval)
        morph_str = f"{morph:.3f}" if morph else "---"

        prefix = "BPE" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_train} & {tok_eval} & {morph_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Bidirectional cross-generalization: 300MB $\leftrightarrow$ FineWiki. "
    )
    lines.append(
        r"Top section trains on 300MB corpora and evaluates on held-out FineWiki data. "
    )
    lines.append(
        r"Bottom section trains on FineWiki and evaluates on 300MB. "
    )
    lines.append(
        r"This reveals whether generalization is symmetric across corpus domains. "
    )
    lines.append(
        r"\textbf{Morph.} is MorphScore boundary recall on English training data.}"
    )
    lines.append(r"\label{tab:bidirectional_gen}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    """Main entry point: check status and generate table."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate bidirectional cross-generalization table")
    parser.add_argument("--check-status", action="store_true", help="Check which experiments have been run")
    parser.add_argument("--generate-table", action="store_true", help="Generate LaTeX table (only if all models exist)")
    parser.add_argument("--all", action="store_true", help="Do both status check and table generation")
    parser.add_argument("--force", action="store_true", help="Generate table even if some models are missing")
    args = parser.parse_args()

    # Default to --all if no args specified
    if not (args.check_status or args.generate_table or args.all):
        args.all = True

    all_complete = True
    if args.check_status or args.all:
        all_complete = check_experiment_status()

    if args.generate_table or args.all:
        if not all_complete and not args.force:
            print("\n" + "!" * 80)
            print("⚠️  TABLE GENERATION SKIPPED")
            print("!" * 80)
            print("Some models are missing. Generating the table now would produce")
            print("incorrect relative changes due to uneven means across languages.")
            print("\nOptions:")
            print("  1. Train the missing models (see commands above)")
            print("  2. Use --force to generate table anyway (not recommended)")
            print("!" * 80)
        else:
            if not all_complete:
                print("\n⚠️  Generating with --force despite missing models...")
            table = generate_bidirectional_table()
            if table:
                output_path = RESULTS_DIR / "table_bidirectional_generalization.tex"
                output_path.write_text(table)
                print(f"\n✓ Written: {output_path}")
                print("\n" + "-" * 40)
                print(table)
                print("-" * 40)


if __name__ == "__main__":
    main()
