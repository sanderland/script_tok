#!/usr/bin/env python3
"""Generate LaTeX tables for unigram paper: init algorithms and final style pruning."""

from pathlib import Path

import pandas as pd

from script_bpe.analysis import evaluate_on_corpus, format_with_relchange, format_tokens_millions, MorphScore
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    FINEWIKI_CORPUS_NAMES,
    DEFAULTS,
    load_experiment_results,
    load_baseline_model,
    load_vocab_from_model_file,
)
from paper_utils.unigram.utils import evaluate_on_corpus_cached, evaluate_morphscore_cached

SMOL_CORPUS_NAMES = ["smol_" + corpus for corpus in CORPUS_NAMES]
RESULTS_DIR = Path("results/unigram_sweeps")

# Table 1: Seed Vocabulary Algorithms
INIT_ALGO_ROWS = [
    ("Pretokens (Ours)", "corpus_long"),
    ("Pretokens, Recovery", "corpus_fallback"),
    ("Full-text (SP)", "corpus_long_no_pt"),
    ("Full-text, Recovery", "corpus_fallback_no_pt"),
]

# Table 2: Generalization (Vocab sizes and Methods)
# Match sweep output types: 0.5*32768=16384.0 (float), 1*32768=32768 (int), 2*32768=65536 (int)
VOCAB_SIZES = [16384.0, 32768, 65536]
VOCAB_LABELS = {16384.0: "16K", 32768: "32K", 65536: "64K"}
METHODS = ["Default", "FSP", "BPE"]


def check_data_completeness(
    df: pd.DataFrame,
    corpus_names: list[str],
    param_name: str,
    param_values: list,
    description: str,
) -> dict[str, list[str]]:
    """Check which corpora have complete data for all param values.

    Returns dict with 'complete' and 'missing' lists, prints diagnostics.
    """
    complete = []
    missing = []

    print(f"\n{'=' * 60}")
    print(f"Data check: {description}")
    print(f"{'=' * 60}")

    for corpus in corpus_names:
        corpus_df = df[df["corpus"] == corpus]
        missing_values = []

        for val in param_values:
            if len(corpus_df[corpus_df[param_name] == val]) == 0:
                missing_values.append(str(val))

        if missing_values:
            missing.append(corpus)
            print(f"  ✗ {corpus}: missing {param_name}={', '.join(missing_values)}")
        else:
            complete.append(corpus)
            print(f"  ✓ {corpus}: complete")

    print(f"\nSummary: {len(complete)}/{len(corpus_names)} corpora complete")

    return {"complete": complete, "missing": missing}


def compute_mean_metrics(
    df: pd.DataFrame,
    corpora: list[str],
    param_name: str,
    param_value,
    baseline_fn,
) -> dict | None:
    """Compute mean objective, tokens, and overlap across corpora."""
    objectives = []
    tokens = []
    overlaps = []

    for corpus in corpora:
        row = df[(df["corpus"] == corpus) & (df[param_name] == param_value)]
        if len(row) == 0:
            return None
        row = row.iloc[0]

        baseline = baseline_fn(corpus)
        if baseline is None:
            return None

        objectives.append(row["objective"])
        tokens.append(row["tokens"])

        # Compute vocab overlap (Jaccard-style: common / union)
        row_vocab = load_vocab_from_model_file(row["model_file"])
        baseline_vocab = load_vocab_from_model_file(baseline["model_file"])
        if row_vocab and baseline_vocab:
            common = len(row_vocab.intersection(baseline_vocab))
            union = len(row_vocab.union(baseline_vocab))
            overlap_pct = common / union * 100
            overlaps.append(overlap_pct)

    return {
        "objective": sum(objectives) / len(objectives),
        "tokens": sum(tokens) / len(tokens),
        "overlap": sum(overlaps) / len(overlaps) if overlaps else None,
    }


def compute_baseline_means(corpora: list[str], baseline_fn) -> dict | None:
    """Compute mean baseline metrics across corpora."""
    objectives = []
    tokens = []

    for corpus in corpora:
        baseline = baseline_fn(corpus)
        if baseline is None:
            return None
        objectives.append(baseline["objective"])
        tokens.append(baseline["tokens"])

    return {
        "objective": sum(objectives) / len(objectives),
        "tokens": sum(tokens) / len(tokens),
    }


def generate_init_algorithms_table() -> str:
    """Generate Table 1: Seed Vocabulary Algorithms."""
    print("\n" + "=" * 70)
    print("GENERATING TABLE: Seed Vocabulary Algorithms")
    print("=" * 70)

    # Load data for both corpus sizes
    init_algo_values = [row[1] for row in INIT_ALGO_ROWS]

    smol_df = load_experiment_results("init_algo", corpus_names=SMOL_CORPUS_NAMES)
    normal_df = load_experiment_results("init_algo", corpus_names=CORPUS_NAMES)

    # Check completeness
    smol_status = check_data_completeness(
        smol_df, SMOL_CORPUS_NAMES, "init_vocab_algo", init_algo_values, "30 MB Corpora (smol)"
    )
    normal_status = check_data_completeness(
        normal_df, CORPUS_NAMES, "init_vocab_algo", init_algo_values, "300 MB Corpora"
    )

    # Define baseline functions (corpus_long is baseline)
    def smol_baseline(corpus):
        row = smol_df[(smol_df["corpus"] == corpus) & (smol_df["init_vocab_algo"] == "corpus_long")]
        return row.iloc[0] if len(row) > 0 else None

    def normal_baseline(corpus):
        row = normal_df[(normal_df["corpus"] == corpus) & (normal_df["init_vocab_algo"] == "corpus_long")]
        return row.iloc[0] if len(row) > 0 else None

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Loss} & \textbf{Tok.} & \textbf{Over.} \\")
    lines.append(r"\midrule")

    # 30 MB section
    if len(smol_status["complete"]) == len(SMOL_CORPUS_NAMES):
        lines.append(r"\multicolumn{4}{@{}l}{\textit{\textbf{Mean over Monolingual 30\,MB Corpora}}} \\")
        lines.append(r"\addlinespace[2pt]")

        baseline_means = compute_baseline_means(SMOL_CORPUS_NAMES, smol_baseline)

        for display_name, algo in INIT_ALGO_ROWS:
            metrics = compute_mean_metrics(smol_df, SMOL_CORPUS_NAMES, "init_vocab_algo", algo, smol_baseline)
            if metrics is None:
                continue

            if algo == "corpus_long":  # baseline row
                loss_str = f"{metrics['objective']:.3f}\\phantom{{\\relchange{{+0.00}}}}"
                tokens_str = f"{metrics['tokens'] / 1e6:.2f}\\phantom{{\\relchange{{+0.00}}}}"
                overlap_str = "---\\phantom{0}"
            else:
                loss_str = format_with_relchange(metrics["objective"], baseline_means["objective"], decimals=3)
                tokens_str = format_tokens_millions(metrics["tokens"], baseline_means["tokens"], decimals=2)
                overlap_str = f"{metrics['overlap']:.1f}" if metrics["overlap"] else "---"

            lines.append(f"{display_name} & {loss_str} & {tokens_str} & {overlap_str} \\\\")
    else:
        print("\n⚠ Skipping 30 MB section: incomplete data")

    # 300 MB section
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{4}{@{}l}{\textit{\textbf{Mean over Monolingual 300\,MB Corpora}}} \\")
    lines.append(r"\addlinespace[2pt]")

    # Check which algorithms we can show
    available_algos = []
    baseline_means = compute_baseline_means(CORPUS_NAMES, normal_baseline)

    for display_name, algo in INIT_ALGO_ROWS:
        metrics = compute_mean_metrics(normal_df, CORPUS_NAMES, "init_vocab_algo", algo, normal_baseline)
        if metrics is None:
            # If main data missing, check if we have pretokens data at least
            if algo in ["corpus_long", "corpus_fallback"]:
                 available_algos.append((display_name, algo))
            else:
                 print(f"⚠ Skipping {display_name} in 300MB: missing data")
        else:
            available_algos.append((display_name, algo))

    for display_name, algo in available_algos:
        metrics = compute_mean_metrics(normal_df, CORPUS_NAMES, "init_vocab_algo", algo, normal_baseline)
        if metrics is None:
            continue

        if algo == "corpus_long":  # baseline row
            loss_str = f"{metrics['objective']:.3f}\\phantom{{\\relchange{{+0.00}}}}"
            tokens_str = f"{metrics['tokens'] / 1e6:.1f}\\phantom{{\\relchange{{+0.00}}}}"
            overlap_str = "---\\phantom{0}"
        else:
            loss_str = format_with_relchange(metrics["objective"], baseline_means["objective"], decimals=3)
            tokens_str = format_tokens_millions(metrics["tokens"], baseline_means["tokens"], decimals=2)
            # Format simple float for million tokens, but keep relchange for others if needed
            # Overriding format_tokens_millions for this specific table look if needed
            # tailored to exactly match user request: 58.7
            if algo == "corpus_fallback":
                 # Hack to match exact user spacing style
                 pass
            
            overlap_str = f"{metrics['overlap']:.1f}" if metrics["overlap"] else "---"

        lines.append(f"{display_name} & {loss_str} & {tokens_str} & {overlap_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\caption{Seed Vocabulary Algorithms. ")
    lines.append("")
    lines.append(
        r"Comparing pretoken-based (Ours) and full-text (SentencePiece-style) initialization, with and without Maximal Valid Prefix Recovery (`Recovery'; Algorithm~\ref{alg:init:repair}). Superscripts show \% change vs baseline; \textbf{Over.} is vocabulary overlap with baseline."
    )
    lines.append("")
    lines.append(
        r"Pretoken-based initialization consistently achieves better loss and compression than full-text. The recovery variation has minimal effect on small corpora (30\,MB) and slightly degrades on larger corpora."
    )
    lines.append("")
    lines.append(
        r"We use pretoken-based initialization without prefix recovery for subsequent experiments.}"
    )
    lines.append(r"\label{tab:seed_init}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def evaluate_model_stats(model, model_path, train_corpus, eval_corpus, ms):
    """Evaluate model on both training (FineWiki) and held-out (300MB) corpora."""
    # 1. Training corpus metrics (FineWiki)
    # Prefer metadata if available and correct corpus
    train_obj = None
    train_tok = None
    
    # Always reload performance from metadata if possible to match cached results exactly
    if model.metadata.get("corpus") == train_corpus:
        # objective is at top level (Unigram only), total_tokens_len is in performance
        train_obj = model.metadata.get("objective")  # None for BPE, which is fine
        perf = model.metadata.get("performance", {})
        train_tok = perf.get("total_tokens_len")
    
    if train_tok is None:
        # Fallback: re-evaluate (only if we're missing tokens - objective can be None for BPE)
        print(f"  [uncached] evaluate_on_corpus: {train_corpus} (no metadata)")
        res = evaluate_on_corpus(model, train_corpus)
        train_obj = res["objective"]
        train_tok = res["tokens"]

    # 2. Held-out corpus metrics (300MB, cached)
    res_eval = evaluate_on_corpus_cached(model, eval_corpus, str(model_path))
    eval_obj = res_eval["objective"]
    eval_tok = res_eval["tokens"]

    results = {
        "finewiki": {"objective": train_obj, "tokens": train_tok},
        "300mb": {"objective": eval_obj, "tokens": eval_tok}
    }
    # MorphScore evaluation - on separate data, uses language code
    lang_code = eval_corpus[:8]  # e.g., "eng_latn" from "eng_latn_300mb"
    if lang_code == "eng_latn":
        ms_results = evaluate_morphscore_cached(ms, model, lang_code, str(model_path))
        ms_score = ms_results["morphscore_recall"]
        results["morph_recall"] = ms_score

    return results

def generate_generalization_table() -> str:
    """Generate Table 2: Generalization (Train on FineWiki, Evaluate on 300MB)."""
    print("\n" + "=" * 70)
    print("GENERATING TABLE: Generalization (Table 2)")
    print("=" * 70)

    # 1. Load MorphScore
    print("Loading MorphScore...")
    ms = MorphScore()

    # 2. Gather data
    # Structure: results[fw_corpus][method][vocab_size] = {finewiki: {}, 300mb: {}}
    # Models trained on FineWiki, evaluated on 300MB
    results = {}

    for i, fw_corpus in enumerate(FINEWIKI_CORPUS_NAMES):
        eval_corpus = CORPUS_NAMES[i]  # 300MB corpus for held-out evaluation
        results[fw_corpus] = {m: {} for m in METHODS}
        
        print(f"\nProcessing {fw_corpus} (eval on {eval_corpus})...")

        # -- Default --
        for vocab_size in VOCAB_SIZES:
            params = {**DEFAULTS, "additional_vocab_size": vocab_size}
            # Use cached loader from train_hyperparameters
            from paper_utils.unigram.train_hyperparameters import load_model_if_cached
            model, path = load_model_if_cached(fw_corpus, params)
            if model:
                print(f"  Default {vocab_size}")
                results[fw_corpus]["Default"][vocab_size] = evaluate_model_stats(model, path, fw_corpus, eval_corpus, ms)
            else:
                 print(f"  Missing Default {vocab_size}")

        # -- FSP --
        fsp_alpha = 0.75
        for vocab_size in VOCAB_SIZES:
            params = {
                **DEFAULTS,
                "additional_vocab_size": vocab_size,
                "final_style_prune": True,
                "pre_final_vocab_factor": 1.0,
                "pruning_shrinking_factor": fsp_alpha,
            }
            from paper_utils.unigram.train_hyperparameters import get_model_path
            path = get_model_path(fw_corpus, params)
            if path.exists():
                print(f"  FSP {vocab_size}")
                model = UnigramModel.load(str(path))
                results[fw_corpus]["FSP"][vocab_size] = evaluate_model_stats(model, path, fw_corpus, eval_corpus, ms)
            else:
                 print(f"  Missing FSP {vocab_size}")

        # -- BPE --
        for vocab_size in VOCAB_SIZES:
            path = RESULTS_DIR / fw_corpus / f"bpe_n{vocab_size}.model.json.gz"
            if path.exists():
                print(f"  BPE {vocab_size}")
                model = BPETokenizer.load(str(path))
                results[fw_corpus]["BPE"][vocab_size] = evaluate_model_stats(model, path, fw_corpus, eval_corpus, ms)
            else:
                 print(f"  Missing BPE {vocab_size}")

    # Check for missing data and warn
    def check_missing_data():
        print("\nData availability check:")
        for vocab_size in VOCAB_SIZES:
            for method in METHODS:
                missing = []
                for fw_corpus in FINEWIKI_CORPUS_NAMES:
                    if not results[fw_corpus][method].get(vocab_size):
                        missing.append(fw_corpus)
                if missing:
                    print(f"  ⚠️  MISSING: {method} {VOCAB_LABELS[vocab_size]}: {', '.join(missing)}")
        print()

    check_missing_data()

    # Helper to compute means - only over corpora that have data for BOTH Default and the method
    def get_means(method, vocab_size, corpus_type, metric_keys):
        values = {k: [] for k in metric_keys}
        included_corpora = []
        for fw_corpus in FINEWIKI_CORPUS_NAMES:
            # Only include corpus if Default baseline also has data (for fair comparison)
            default_res = results[fw_corpus]["Default"].get(vocab_size)
            if not (default_res and default_res.get(corpus_type)):
                continue  # Skip corpora without Default baseline
            
            res = results[fw_corpus][method].get(vocab_size)
            if res and res.get(corpus_type):
                included_corpora.append(fw_corpus)
                for k in metric_keys:
                    val = res[corpus_type].get(k)
                    if val is not None:
                        values[k].append(val)
        
        if len(included_corpora) < len(FINEWIKI_CORPUS_NAMES):
            excluded = set(FINEWIKI_CORPUS_NAMES) - set(included_corpora)
            print(f"  {method} {vocab_size}: mean over {len(included_corpora)}/6 corpora (excluded: {', '.join(excluded)})")
        
        return {k: (sum(v)/len(v) if v else None) for k, v in values.items()}

    # Helper to get English-only morph score (MorphScore is only for English)
    def get_english_morph(method, vocab_size):
        res = results.get("finewiki_en_1gb", {}).get(method, {}).get(vocab_size)
        if res:
            return res.get("morph_recall")
        return None

    lines = []
    lines.append(r"\begin{table}[pt]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Voc} & \textbf{FW Tok.} & \textbf{300MB Tok.} & \textbf{Morph.} \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\textit{\textbf{Train on FineWiki, Evaluate on 300MB}}} \\")
    lines.append(r"\addlinespace[2pt]")

    # Default (baseline - no relchange)
    for vocab_size in VOCAB_SIZES:
        means_fw = get_means("Default", vocab_size, "finewiki", ["tokens"])
        means_300mb = get_means("Default", vocab_size, "300mb", ["tokens"])
        
        tok_fw = f"{means_fw['tokens']/1e6:.1f}\\phantom{{\\relchange{{+0.0}}}}" if means_fw['tokens'] else "---"
        tok_300mb = f"{means_300mb['tokens']/1e6:.1f}\\phantom{{\\relchange{{+0.0}}}}" if means_300mb['tokens'] else "---"
        morph_val = get_english_morph("Default", vocab_size)
        morph = f"{morph_val:.3f}" if morph_val else "---"
        
        prefix = "Default" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_fw} & {tok_300mb} & {morph} \\\\")
    
    lines.append(r"\addlinespace[1pt]")

    # FSP
    for vocab_size in VOCAB_SIZES:
        means_fw = get_means("FSP", vocab_size, "finewiki", ["tokens"])
        means_300mb = get_means("FSP", vocab_size, "300mb", ["tokens"])
        base_fw = get_means("Default", vocab_size, "finewiki", ["tokens"])
        base_300mb = get_means("Default", vocab_size, "300mb", ["tokens"])
        
        tok_fw = format_tokens_millions(means_fw['tokens'], base_fw['tokens']) if means_fw['tokens'] and base_fw['tokens'] else "---"
        tok_300mb = format_tokens_millions(means_300mb['tokens'], base_300mb['tokens']) if means_300mb['tokens'] and base_300mb['tokens'] else "---"
        morph_val = get_english_morph("FSP", vocab_size)
        morph = f"{morph_val:.3f}" if morph_val else "---"
        
        prefix = "FSP" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_fw} & {tok_300mb} & {morph} \\\\")

    lines.append(r"\addlinespace[1pt]")

    # BPE
    for vocab_size in VOCAB_SIZES:
        means_fw = get_means("BPE", vocab_size, "finewiki", ["tokens"])
        means_300mb = get_means("BPE", vocab_size, "300mb", ["tokens"])
        base_fw = get_means("Default", vocab_size, "finewiki", ["tokens"])
        base_300mb = get_means("Default", vocab_size, "300mb", ["tokens"])
        
        tok_fw = format_tokens_millions(means_fw['tokens'], base_fw['tokens']) if means_fw['tokens'] and base_fw['tokens'] else "---"
        tok_300mb = format_tokens_millions(means_300mb['tokens'], base_300mb['tokens']) if means_300mb['tokens'] and base_300mb['tokens'] else "---"
        morph_val = get_english_morph("BPE", vocab_size)
        morph = f"{morph_val:.3f}" if morph_val else "---"
        
        prefix = "BPE" if vocab_size == VOCAB_SIZES[0] else ""
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_fw} & {tok_300mb} & {morph} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Compression and morphological alignment across methods and vocabulary sizes. "
    )
    lines.append(
        r"Models trained on FineWiki (1\,GB), evaluated on held-out 300\,MB corpora. "
    )
    lines.append(
        r"FSP and BPE both improve compression vs default Unigram, with BPE achieving the best compression especially on held-out data. "
    )
    lines.append(
        r"\textbf{Morph.} is MorphScore boundary recall on English (see \autoref{app:morphscore}): Unigram methods substantially outperform BPE on morphological alignment, suggesting Unigram's probabilistic framework better captures linguistic structure.}"
    )
    lines.append(r"\label{tab:fsp_bpe_val}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    """Generate all main tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Table 1: Init Algorithms
    table1 = generate_init_algorithms_table()
    if table1:
        output_path = RESULTS_DIR / "table_init_algorithms.tex"
        output_path.write_text(table1)
        print(f"\n✓ Written: {output_path}")
        print("\n" + "-" * 40)
        print(table1)
        print("-" * 40)

    # Table 2: Generalization (FSP/BPE/Morph)
    table2 = generate_generalization_table()
    if table2:
        output_path = RESULTS_DIR / "table_fsp_bpe_val.tex"
        output_path.write_text(table2)
        print(f"\n✓ Written: {output_path}")
        print("\n" + "-" * 40)
        print(table2)
        print("-" * 40)


if __name__ == "__main__":
    main()
