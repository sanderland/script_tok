#!/usr/bin/env python3
"""Generate LaTeX tables for unigram paper: init algorithms and final style pruning."""


import pandas as pd

from script_bpe.analysis import format_with_relchange, format_tokens_millions, MorphScore
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    FINEWIKI_CORPUS_NAMES,
    DEFAULTS,
    RESULTS_DIR,
    load_experiment_results,
    load_vocab_from_model_file,
)
from paper_utils.unigram.utils import evaluate_on_corpus_cached, evaluate_morphscore_cached

SMOL_CORPUS_NAMES = ["smol_" + corpus for corpus in CORPUS_NAMES]

# Table 1: Seed Vocabulary Algorithms
INIT_ALGO_ROWS = [
    ("Pretokens (Ours)", "corpus_long"),
    ("Pretok., Recovery", "corpus_fallback"),
    ("Full-text (SP)", "corpus_long_no_pt"),
    ("Full-text, Recov.", "corpus_fallback_no_pt"),
]

# Table 2: Generalization (Vocab sizes and Methods)
# Match sweep output types: 0.5*32768=16384.0 (float), 1*32768=32768 (int), 2*32768=65536 (int)
VOCAB_SIZES = [16384.0, 32768, 65536]
VOCAB_LABELS = {16384.0: "16K", 32768: "32K", 65536: "64K"}
METHODS = ["Baseline", "FSP", "BPE"]


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
    _normal_status = check_data_completeness(
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
    lines.append(r"\textbf{Method} & \textbf{Loss} & \textbf{\#Tokens} & \textbf{Overlap} \\")
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
            tokens_str = f"{metrics['tokens'] / 1e6:.2f}\\phantom{{\\relchange{{+0.00}}}}"
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
    lines.append(
        r"\caption{Seed vocabulary initialization algorithms. Comparing pretoken-based (Ours) and full-text (SentencePiece-style) initialization, with and without Maximal Valid Prefix Recovery (Algorithm~\ref{alg:init:repair}). Superscripts show \% change vs baseline; `Overlap' is vocabulary overlap with baseline. Pretoken-based initialization achieves better loss and compression on both corpus sizes. The recovery variation has minimal effect, suggesting that missing prefixes are recovered naturally in sufficiently large corpora.}"
    )
    lines.append(r"\label{tab:seed_init}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def evaluate_model_on_300mb(model, model_path, corpus_300mb, ms):
    """Evaluate model on 300MB corpus (used for both in-domain and holdout evaluation)."""
    # Check if we have cached results in metadata (only if model was trained on this corpus)
    tok = None
    if model.metadata.get("corpus") == corpus_300mb:
        perf = model.metadata.get("performance", {})
        tok = perf.get("total_tokens_len")

    if tok is None:
        # Evaluate on 300MB corpus (cached)
        res = evaluate_on_corpus_cached(model, corpus_300mb, str(model_path))
        tok = res["tokens"]

    results = {"tokens": tok}

    # MorphScore evaluation - on separate data, uses language code
    lang_code = corpus_300mb[:8]  # e.g., "eng_latn" from "eng_latn_300mb"
    if lang_code == "eng_latn":
        ms_results = evaluate_morphscore_cached(ms, model, lang_code, str(model_path))
        results["morph_recall"] = ms_results["morphscore_recall"]

    return results


def generate_generalization_table() -> str:
    """Generate Table 2: Generalization comparing in-domain vs holdout training.

    Both columns evaluate on 300MB corpora:
    - Column 1: Train on 300MB, eval on 300MB (in-domain)
    - Column 2: Train on FineWiki, eval on 300MB (holdout/out-of-domain)
    """
    print("\n" + "=" * 70)
    print("GENERATING TABLE: Generalization (Table 2)")
    print("=" * 70)

    # 1. Load MorphScore
    print("Loading MorphScore...")
    ms = MorphScore()

    from paper_utils.unigram.train_hyperparameters import load_model_if_cached, get_model_path

    # 2. Gather data for BOTH training scenarios
    # Structure: results_300mb[corpus][method][vocab_size] = {tokens, morph_recall}
    #            results_fw[corpus][method][vocab_size] = {tokens, morph_recall}
    results_300mb = {}  # Models trained on 300MB, evaluated on 300MB
    results_fw = {}  # Models trained on FineWiki, evaluated on 300MB

    for i, corpus_300mb in enumerate(CORPUS_NAMES):
        fw_corpus = FINEWIKI_CORPUS_NAMES[i]
        results_300mb[corpus_300mb] = {m: {} for m in METHODS}
        results_fw[corpus_300mb] = {m: {} for m in METHODS}  # keyed by eval corpus for consistency

        print(f"\nProcessing {corpus_300mb}...")

        # ========== Models trained on 300MB ==========
        print("  [300MB-trained models]")
        for vocab_size in VOCAB_SIZES:
            # Baseline
            params = {**DEFAULTS, "additional_vocab_size": vocab_size}
            model, path = load_model_if_cached(corpus_300mb, params)
            if model:
                print(f"    Baseline {vocab_size}")
                results_300mb[corpus_300mb]["Baseline"][vocab_size] = evaluate_model_on_300mb(
                    model, path, corpus_300mb, ms
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
                print(f"    FSP {vocab_size}")
                model = UnigramModel.load(str(path_fsp))
                results_300mb[corpus_300mb]["FSP"][vocab_size] = evaluate_model_on_300mb(
                    model, path_fsp, corpus_300mb, ms
                )

            # BPE
            path_bpe = RESULTS_DIR / corpus_300mb / f"bpe_n{vocab_size}.model.json.gz"
            if path_bpe.exists():
                print(f"    BPE {vocab_size}")
                model = BPETokenizer.load(str(path_bpe))
                results_300mb[corpus_300mb]["BPE"][vocab_size] = evaluate_model_on_300mb(
                    model, path_bpe, corpus_300mb, ms
                )

        # ========== Models trained on FineWiki, evaluated on 300MB ==========
        print("  [FineWiki-trained models]")
        for vocab_size in VOCAB_SIZES:
            # Baseline
            params = {**DEFAULTS, "additional_vocab_size": vocab_size}
            model, path = load_model_if_cached(fw_corpus, params)
            if model:
                print(f"    Baseline {vocab_size}")
                results_fw[corpus_300mb]["Baseline"][vocab_size] = evaluate_model_on_300mb(
                    model, path, corpus_300mb, ms
                )

            # FSP
            params_fsp = {
                **DEFAULTS,
                "additional_vocab_size": vocab_size,
                "final_style_prune": True,
                "pre_final_vocab_factor": 1.0,
                "pruning_shrinking_factor": 0.75,
            }
            path_fsp = get_model_path(fw_corpus, params_fsp)
            if path_fsp.exists():
                print(f"    FSP {vocab_size}")
                model = UnigramModel.load(str(path_fsp))
                results_fw[corpus_300mb]["FSP"][vocab_size] = evaluate_model_on_300mb(model, path_fsp, corpus_300mb, ms)

            # BPE
            path_bpe = RESULTS_DIR / fw_corpus / f"bpe_n{vocab_size}.model.json.gz"
            if path_bpe.exists():
                print(f"    BPE {vocab_size}")
                model = BPETokenizer.load(str(path_bpe))
                results_fw[corpus_300mb]["BPE"][vocab_size] = evaluate_model_on_300mb(model, path_bpe, corpus_300mb, ms)

    # Helper to compute means across corpora
    def get_means(results_dict, method, vocab_size, metric_keys):
        values = {k: [] for k in metric_keys}
        for corpus in CORPUS_NAMES:
            res = results_dict.get(corpus, {}).get(method, {}).get(vocab_size)
            if res:
                for k in metric_keys:
                    val = res.get(k)
                    if val is not None:
                        values[k].append(val)
        return {k: (sum(v) / len(v) if v else None) for k, v in values.items()}

    # Helper to get English-only morph score
    def get_english_morph(results_dict, method, vocab_size):
        res = results_dict.get("eng_latn_300mb", {}).get(method, {}).get(vocab_size)
        if res:
            return res.get("morph_recall")
        return None

    lines = []
    lines.append(r"\newcommand{\tabtwosep}{\addlinespace[4pt]\midrule\addlinespace[4pt]}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & $\vocabsize$ & \textbf{Train Tok.} & \textbf{FW Tok.} & \textbf{Morph.} \\")
    lines.append(r"\midrule")
    lines.append(r"\addlinespace[3.3pt]")

    # Baseline (no relchange)
    for vocab_size in VOCAB_SIZES:
        means_300mb = get_means(results_300mb, "Baseline", vocab_size, ["tokens"])
        means_fw = get_means(results_fw, "Baseline", vocab_size, ["tokens"])

        tok_300mb = (
            f"{means_300mb['tokens'] / 1e6:.2f}\\phantom{{\\relchange{{+0.00}}}}" if means_300mb["tokens"] else "---"
        )
        tok_fw = f"{means_fw['tokens'] / 1e6:.2f}\\phantom{{\\relchange{{+0.00}}}}" if means_fw["tokens"] else "---"
        morph_val = get_english_morph(results_fw, "Baseline", vocab_size)
        morph = f"{morph_val:.3f}" if morph_val else "---"

        prefix = "Baseline" if vocab_size == VOCAB_SIZES[0] else "        "
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_300mb} & {tok_fw} & {morph} \\\\")

    lines.append(r"\tabtwosep")

    # FSP
    for vocab_size in VOCAB_SIZES:
        means_300mb = get_means(results_300mb, "FSP", vocab_size, ["tokens"])
        means_fw = get_means(results_fw, "FSP", vocab_size, ["tokens"])
        base_300mb = get_means(results_300mb, "Baseline", vocab_size, ["tokens"])
        base_fw = get_means(results_fw, "Baseline", vocab_size, ["tokens"])

        tok_300mb = (
            format_tokens_millions(means_300mb["tokens"], base_300mb["tokens"], decimals=2)
            if means_300mb["tokens"] and base_300mb["tokens"]
            else "---"
        )
        tok_fw = (
            format_tokens_millions(means_fw["tokens"], base_fw["tokens"], decimals=2)
            if means_fw["tokens"] and base_fw["tokens"]
            else "---"
        )
        morph_val = get_english_morph(results_fw, "FSP", vocab_size)
        morph = f"{morph_val:.3f}" if morph_val else "---"

        prefix = "FSP" if vocab_size == VOCAB_SIZES[0] else "        "
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_300mb} & {tok_fw} & {morph} \\\\")

    lines.append(r"\tabtwosep")

    # BPE
    for vocab_size in VOCAB_SIZES:
        means_300mb = get_means(results_300mb, "BPE", vocab_size, ["tokens"])
        means_fw = get_means(results_fw, "BPE", vocab_size, ["tokens"])
        base_300mb = get_means(results_300mb, "Baseline", vocab_size, ["tokens"])
        base_fw = get_means(results_fw, "Baseline", vocab_size, ["tokens"])

        tok_300mb = (
            format_tokens_millions(means_300mb["tokens"], base_300mb["tokens"], decimals=2)
            if means_300mb["tokens"] and base_300mb["tokens"]
            else "---"
        )
        tok_fw = (
            format_tokens_millions(means_fw["tokens"], base_fw["tokens"], decimals=2)
            if means_fw["tokens"] and base_fw["tokens"]
            else "---"
        )
        morph_val = get_english_morph(results_fw, "BPE", vocab_size)
        morph = f"{morph_val:.3f}" if morph_val else "---"

        prefix = "BPE" if vocab_size == VOCAB_SIZES[0] else "        "
        lines.append(f"{prefix} & {VOCAB_LABELS[vocab_size]} & {tok_300mb} & {tok_fw} & {morph} \\\\")

    lines.append(r"\addlinespace[1.1pt]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Compression (M tokens, lower is better, superscripts show \% change vs baseline) and morphological alignment across vocabulary sizes $\vocabsize$. Baseline is Unigram with our default settings; FSP uses Final-Style Pruning."
    )
    lines.append(
        r"\emph{Train Tok.}: trained and evaluated on same 300\,MB corpora. \emph{FW Tok.}: trained on FineWiki 1\,GB, evaluated on 300\,MB corpora. Note that the gap with Baseline widens, while BPE gains a small edge over FSP."
    )
    lines.append(r"\emph{Morph.}: MorphScore boundary recall on English (higher is better).}")
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
