#!/usr/bin/env python3
"""Generate detailed appendix LaTeX tables for unigram hyperparameter experiments."""


import pandas as pd

from script_bpe.analysis.formatting import (
    format_latex_value,
    format_vocab_value,
    mark_biggest_in_group,
)
from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    RESULTS_DIR,
    load_experiment_results,
    load_baseline_model,
    compute_relative_performance,
    compute_vocab_overlap,
)

SMOL_CORPUS_NAMES = ["smol_" + corpus for corpus in CORPUS_NAMES]

# Option to include iterations row in tables
INCLUDE_ITERATIONS = False

# Number of rows per corpus (Loss, Tokens, Vocab, optionally Iterations)
N_ROWS_PER_CORPUS = 4 if INCLUDE_ITERATIONS else 3

# Display names for corpora (order matters for table rows)
CORPUS_ORDER = [
    "arb_arab_300mb",
    "deu_latn_300mb",
    "eng_latn_300mb",
    "hin_deva_300mb",
    "kor_hang_300mb",
    "zho_hans_300mb",
]

CORPUS_DISPLAY = {
    "arb_arab_300mb": "Arabic",
    "deu_latn_300mb": "German",
    "eng_latn_300mb": "English",
    "hin_deva_300mb": "Hindi",
    "kor_hang_300mb": "Korean",
    "zho_hans_300mb": "Chinese",
    "smol_arb_arab_300mb": "Arabic",
    "smol_deu_latn_300mb": "German",
    "smol_eng_latn_300mb": "English",
    "smol_hin_deva_300mb": "Hindi",
    "smol_kor_hang_300mb": "Korean",
    "smol_zho_hans_300mb": "Chinese",
}

# Corpus order for init/FSP tables (English first, then alphabetical by language name)
CORPUS_ORDER_INIT = [
    "eng_latn_300mb",
    "deu_latn_300mb",
    "arb_arab_300mb",
    "hin_deva_300mb",
    "kor_hang_300mb",
    "zho_hans_300mb",
]

SMOL_CORPUS_ORDER = ["smol_" + c for c in CORPUS_ORDER_INIT]

# Init algorithm mapping (algo_key, display_name)
INIT_ALGO_ORDER = [
    ("corpus_long_no_pt", "Full text"),
    ("corpus_fallback_no_pt", "Full text, Recovery"),
    ("corpus_fallback", "Pretokens, Recovery"),
]
INIT_ALGO_BASELINE = "corpus_long"  # Pretokens without fallback

# FSP alpha values
FSP_ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]

# Parameter groups for subtables
# Each group: (param_name, display_name, values_to_show, default_value)
SUBTABLE_1_PARAMS = [
    ("initial_vocab_factor", r"$\vocabsizeseedfac$", [3, 25, 50, 100], 10),
    ("pre_final_vocab_factor", r"$\overshootfactor$", [1.0, 1.25, 1.5, 2.0], 1.1),
]

SUBTABLE_2_PARAMS = [
    ("m_step_digamma", "digamma", [False], True),
    ("m_step_low_count_threshold", r"$\mprunecount$", [0.0, 2.0, 10.0], 0.5),
    ("num_sub_iterations", r"$\nem$", [1, 3, 5], 2),
    ("pruning_shrinking_factor", r"$\alphaprune$", [0.5, 0.9, 0.95], 0.75),
]


def load_param_data(param_name: str, values: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load relative performance and vocab overlap data for a parameter sweep.

    Returns (rel_df, vocab_df, raw_df).
    """
    df = load_experiment_results(param_name, corpus_names=CORPUS_NAMES)

    if df.empty:
        print(f"  ✗ No data for {param_name}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Filter to requested values
    df = df[df[param_name].isin(values)].copy()

    # Compute relative performance vs baseline
    rel_df = compute_relative_performance(df, load_baseline_model)

    # Compute vocab overlap
    vocab_df = compute_vocab_overlap(df, load_baseline_model)

    return rel_df, vocab_df, df


def get_metric_values(
    rel_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    param_name: str,
    param_values: list,
    corpus: str,
    raw_df: pd.DataFrame | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Extract loss, tokens, vocab, iterations values for a corpus across parameter values."""
    loss_vals = []
    token_vals = []
    vocab_vals = []
    iter_vals = []

    for val in param_values:
        # Get relative metrics
        rel_row = rel_df[(rel_df["corpus"] == corpus) & (rel_df[param_name] == val)]
        if len(rel_row) > 0:
            loss_vals.append(rel_row["rel_objective"].iloc[0])
            token_vals.append(rel_row["rel_tokens"].iloc[0])
        else:
            loss_vals.append(float("nan"))
            token_vals.append(float("nan"))

        # Get vocab overlap (convert diff to overlap)
        vocab_row = vocab_df[(vocab_df["corpus"] == corpus) & (vocab_df[param_name] == val)]
        if len(vocab_row) > 0:
            vocab_vals.append(100 - vocab_row["vocab_diff_pct"].iloc[0])
        else:
            vocab_vals.append(float("nan"))

        # Get iterations from raw df
        if raw_df is not None and "num_iterations" in raw_df.columns:
            raw_row = raw_df[(raw_df["corpus"] == corpus) & (raw_df[param_name] == val)]
            if len(raw_row) > 0:
                iter_vals.append(raw_row["num_iterations"].iloc[0])
            else:
                iter_vals.append(float("nan"))
        else:
            iter_vals.append(float("nan"))

    return loss_vals, token_vals, vocab_vals, iter_vals


def compute_mean_values(all_corpus_values: list[list[float]]) -> list[float]:
    """Compute mean across corpora for each parameter value."""
    n_values = len(all_corpus_values[0]) if all_corpus_values else 0
    means = []
    for val_idx in range(n_values):
        vals = [corpus_vals[val_idx] for corpus_vals in all_corpus_values if not pd.isna(corpus_vals[val_idx])]
        means.append(sum(vals) / len(vals) if vals else float("nan"))
    return means


def format_and_mark_biggest(
    all_raw: list[list[float]],
    formatter: callable,
) -> list[list[str]]:
    """Format values and mark the biggest outlier."""
    all_formatted = [[formatter(v) for v in row] for row in all_raw]
    return mark_biggest_in_group(all_formatted, all_raw)


def format_iter_value(val: float, is_mean: bool = False) -> str:
    """Format iteration count as integer, or 1 decimal place for means."""
    if pd.isna(val):
        return "---"
    if is_mean:
        return f"{val:.1f}"
    return str(int(val))


def build_tabular(
    params: list[tuple[str, str, list, any]],
    mean_only: bool = False,
) -> str:
    """Build a tabular with multiple parameters grouped in columns (no subtable wrapper).

    Args:
        params: List of (param_name, display_name, values, default) tuples.
        mean_only: If True, only include the Mean row (no per-corpus rows).
    """

    # Load data for all parameters
    param_data = {}
    print("\nLoading data for tabular...")
    for param_name, display_name, values, default in params:
        print(f"  Loading {param_name}...")
        rel_df, vocab_df, raw_df = load_param_data(param_name, values)
        param_data[param_name] = (rel_df, vocab_df, raw_df, values, default)

    # Build column structure
    total_value_cols = sum(len(values) for _, _, values, _ in params)

    # Build tabular
    lines = []

    # Column spec
    col_spec = "ll|" + "|".join("r" * len(values) for _, _, values, _ in params)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Parameter name header row with defaults in parentheses
    header1_parts = [" ", " "]
    col_idx = 3  # Start after Corpus and Metric columns
    cmidrule_parts = []
    for param_name, display_name, values, default in params:
        n_cols = len(values)
        # Format default for display
        if isinstance(default, bool):
            default_str = "on" if default else "off"
        else:
            default_str = str(default)
        header_text = f"{display_name} ({default_str})"
        if n_cols == 1:
            header1_parts.append(f"\\multicolumn{{1}}{{c}}{{{header_text}}}")
        else:
            header1_parts.append(f"\\multicolumn{{{n_cols}}}{{c}}{{{header_text}}}")
        cmidrule_parts.append(f"\\cmidrule(lr){{{col_idx}-{col_idx + n_cols - 1}}}")
        col_idx += n_cols
    lines.append(" & ".join(header1_parts) + r" \\")
    lines.append(" ".join(cmidrule_parts))

    # Parameter value header row
    header2_parts = ["Corpus", "Metric"]
    for param_name, _, values, _ in params:
        for val in values:
            if isinstance(val, bool):
                header2_parts.append("Off" if not val else "On")
            else:
                header2_parts.append(str(val))
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    # Collect all data per parameter (for biggest marking)
    all_loss_raw_by_param = {p[0]: [] for p in params}
    all_token_raw_by_param = {p[0]: [] for p in params}
    all_vocab_raw_by_param = {p[0]: [] for p in params}
    all_iter_raw_by_param = {p[0]: [] for p in params}

    # First pass: collect all raw values
    for corpus in CORPUS_ORDER:
        for param_name, _, values, _ in params:
            rel_df, vocab_df, raw_df, _, _ = param_data[param_name]
            loss_vals, token_vals, vocab_vals, iter_vals = get_metric_values(
                rel_df, vocab_df, param_name, values, corpus, raw_df
            )
            all_loss_raw_by_param[param_name].append(loss_vals)
            all_token_raw_by_param[param_name].append(token_vals)
            all_vocab_raw_by_param[param_name].append(vocab_vals)
            all_iter_raw_by_param[param_name].append(iter_vals)

    # Format and mark biggest per parameter
    all_loss_formatted_by_param = {}
    all_token_formatted_by_param = {}
    all_vocab_formatted_by_param = {}
    all_iter_formatted_by_param = {}

    for param_name, _, _, _ in params:
        all_loss_formatted_by_param[param_name] = format_and_mark_biggest(
            all_loss_raw_by_param[param_name], format_latex_value
        )
        all_token_formatted_by_param[param_name] = format_and_mark_biggest(
            all_token_raw_by_param[param_name], format_latex_value
        )
        # For vocab, we mark biggest difference (100 - overlap)
        vocab_diff_raw = [
            [100 - v if not pd.isna(v) else float("nan") for v in row] for row in all_vocab_raw_by_param[param_name]
        ]
        vocab_formatted = format_and_mark_biggest(
            vocab_diff_raw, lambda v: format_vocab_value(100 - v) if not pd.isna(v) else "---"
        )
        all_vocab_formatted_by_param[param_name] = vocab_formatted
        # Iterations - just format, no biggest marking
        all_iter_formatted_by_param[param_name] = [
            [format_iter_value(v) for v in row] for row in all_iter_raw_by_param[param_name]
        ]

    # Build corpus rows (unless mean_only)
    if not mean_only:
        for corpus_idx, corpus in enumerate(CORPUS_ORDER):
            display_name = CORPUS_DISPLAY[corpus]

            # Loss row
            loss_parts = [f"\\multirow{{{N_ROWS_PER_CORPUS}}}{{*}}{{{display_name}}}", r"\textit{Loss}"]
            for param_name, _, _, _ in params:
                loss_parts.extend(all_loss_formatted_by_param[param_name][corpus_idx])
            lines.append(" & ".join(loss_parts) + r" \\")

            # Tokens row
            token_parts = [" ", r"\textit{Tokens}"]
            for param_name, _, _, _ in params:
                token_parts.extend(all_token_formatted_by_param[param_name][corpus_idx])
            lines.append(" & ".join(token_parts) + r" \\")

            # Vocab row
            vocab_parts = [" ", r"\textit{Vocabulary Overlap}"]
            for param_name, _, _, _ in params:
                vocab_parts.extend(all_vocab_formatted_by_param[param_name][corpus_idx])
            lines.append(" & ".join(vocab_parts) + r" \\")

            # Iterations row (optional)
            if INCLUDE_ITERATIONS:
                iter_parts = [" ", r"\textit{Iterations}"]
                for param_name, _, _, _ in params:
                    iter_parts.extend(all_iter_formatted_by_param[param_name][corpus_idx])
                lines.append(" & ".join(iter_parts) + r" \\")

            # Separator (full width before Mean, partial for others)
            if corpus_idx == len(CORPUS_ORDER) - 1:
                lines.append(f"\\cmidrule{{1-{2 + total_value_cols}}}")
            else:
                lines.append(f"\\cmidrule{{2-{2 + total_value_cols}}}")

    # Mean row
    mean_loss_parts = [f"\\multirow{{{N_ROWS_PER_CORPUS}}}{{*}}{{Mean}}", r"\textit{Loss}"]
    mean_token_parts = [" ", r"\textit{Tokens}"]
    mean_vocab_parts = [" ", r"\textit{Vocabulary Overlap}"]
    mean_iter_parts = [" ", r"\textit{Iterations}"]

    for param_name, _, values, _ in params:
        loss_means = compute_mean_values(all_loss_raw_by_param[param_name])
        token_means = compute_mean_values(all_token_raw_by_param[param_name])
        vocab_means = compute_mean_values(all_vocab_raw_by_param[param_name])
        iter_means = compute_mean_values(all_iter_raw_by_param[param_name])

        mean_loss_parts.extend([format_latex_value(v) for v in loss_means])
        mean_token_parts.extend([format_latex_value(v) for v in token_means])
        mean_vocab_parts.extend([format_vocab_value(v) for v in vocab_means])
        mean_iter_parts.extend([format_iter_value(v, is_mean=True) for v in iter_means])

    lines.append(" & ".join(mean_loss_parts) + r" \\")
    lines.append(" & ".join(mean_token_parts) + r" \\")
    lines.append(" & ".join(mean_vocab_parts) + r" \\")
    if INCLUDE_ITERATIONS:
        lines.append(" & ".join(mean_iter_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def generate_hyperparams_table() -> str:
    """Generate the full hyperparameters appendix table."""
    print("=" * 70)
    print("GENERATING TABLE: Unigram Hyperparameter Effects (Appendix)")
    print("=" * 70)

    lines = []

    # Table (no section header - this table comes second)
    lines.append(r"\begin{table}[H]")
    lines.append(r"%\renewcommand{\arraystretch}{1.0}")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Detailed Unigram hyperparameter results. For each corpus, we report relative change (\%) in Loss and Token Count (lower is better) and absolute Vocabulary Overlap (100\% = identical to baseline). Significant deviations ($>0.5\%$) are highlighted (\goodoutlier{good} and \badoutlier{bad}), with the \biggestoutlier{largest} absolute change per metric marked. Results are shown for relative seed size (default $\vocabsizeseedfac=10$), pruning overshoot (default $\overshootfactor=1.1$), M-step digamma transformation (default on), pruning threshold (default $\mprunecount=0.5$), EM sub-iterations (default $\nem=2$), and pruning shrinking factor (default $\alphaprune=0.75$).}"
    )
    lines.append(r"\label{tab:unigram_hyperparams}")
    lines.append("")

    # Seed Vocabulary Parameters
    lines.append(r"\textbf{Seed Vocabulary Parameters}")
    tabular1 = build_tabular(SUBTABLE_1_PARAMS)
    lines.append(tabular1)
    lines.append(r"\\")
    lines.append(r"\vspace{0.5cm}")

    # Pruning Loop Parameters (mean only, with explanatory text)
    lines.append(r"\textbf{Pruning Loop Parameters}")
    lines.append(r"\\[0.2cm]")
    lines.append(
        r"\noindent Per-language results for pruning loop parameters show similarly negligible effects across all corpora, with no single variation exceeding $0.5\%$ change in loss or token count. The mean results below summarize these findings:"
    )
    lines.append(r"\\[0.3cm]")
    tabular2 = build_tabular(SUBTABLE_2_PARAMS, mean_only=True)
    lines.append(tabular2)
    lines.append("")

    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_init_algo_combined_subtable() -> tuple[str, bool]:
    """Build init algorithm comparison subtable with 30MB and 300MB as column groups.

    Returns (latex_string, has_data).
    """
    print("\nLoading init_algo data for combined 30MB/300MB table...")

    # Load data for both corpus sizes
    smol_df = load_experiment_results("init_algo", corpus_names=SMOL_CORPUS_NAMES)
    normal_df = load_experiment_results("init_algo", corpus_names=CORPUS_NAMES)

    if smol_df.empty and normal_df.empty:
        print("  ✗ No init_algo data found")
        return "", False

    # Check data availability
    print("\n  30 MB data:")
    for corpus in SMOL_CORPUS_ORDER:
        if not smol_df.empty:
            corpus_df = smol_df[smol_df["corpus"] == corpus]
            algos = corpus_df["init_vocab_algo"].unique().tolist() if len(corpus_df) > 0 else []
            print(f"    {CORPUS_DISPLAY.get(corpus, corpus)}: {algos}")
        else:
            print(f"    {CORPUS_DISPLAY.get(corpus, corpus)}: no data")

    print("\n  300 MB data:")
    for corpus in CORPUS_ORDER_INIT:
        if not normal_df.empty:
            corpus_df = normal_df[normal_df["corpus"] == corpus]
            algos = corpus_df["init_vocab_algo"].unique().tolist() if len(corpus_df) > 0 else []
            print(f"    {CORPUS_DISPLAY.get(corpus, corpus)}: {algos}")
        else:
            print(f"    {CORPUS_DISPLAY.get(corpus, corpus)}: no data")

    # Define baseline functions
    def smol_baseline(corpus):
        if smol_df.empty:
            return None
        row = smol_df[(smol_df["corpus"] == corpus) & (smol_df["init_vocab_algo"] == INIT_ALGO_BASELINE)]
        return row.iloc[0] if len(row) > 0 else None

    def normal_baseline(corpus):
        if normal_df.empty:
            return None
        row = normal_df[(normal_df["corpus"] == corpus) & (normal_df["init_vocab_algo"] == INIT_ALGO_BASELINE)]
        return row.iloc[0] if len(row) > 0 else None

    # Compute relative performance and vocab overlap
    smol_rel_df = compute_relative_performance(smol_df, smol_baseline) if not smol_df.empty else pd.DataFrame()
    smol_vocab_df = compute_vocab_overlap(smol_df, smol_baseline) if not smol_df.empty else pd.DataFrame()
    normal_rel_df = compute_relative_performance(normal_df, normal_baseline) if not normal_df.empty else pd.DataFrame()
    normal_vocab_df = compute_vocab_overlap(normal_df, normal_baseline) if not normal_df.empty else pd.DataFrame()

    algo_values = [a[0] for a in INIT_ALGO_ORDER]
    n_algos = len(INIT_ALGO_ORDER)

    # Collect data: for each corpus, get [30MB algos..., 300MB algos...]
    all_loss_raw = []
    all_token_raw = []
    all_vocab_raw = []

    for lang_idx, corpus in enumerate(CORPUS_ORDER_INIT):
        smol_corpus = SMOL_CORPUS_ORDER[lang_idx]

        # Get 30MB values
        smol_loss, smol_token, smol_vocab, _ = get_metric_values(
            smol_rel_df, smol_vocab_df, "init_vocab_algo", algo_values, smol_corpus
        )
        # Get 300MB values
        normal_loss, normal_token, normal_vocab, _ = get_metric_values(
            normal_rel_df, normal_vocab_df, "init_vocab_algo", algo_values, corpus
        )

        # Combine: [30MB values..., 300MB values...]
        all_loss_raw.append(smol_loss + normal_loss)
        all_token_raw.append(smol_token + normal_token)
        all_vocab_raw.append(smol_vocab + normal_vocab)

    # Format and mark biggest (separately for 30MB and 300MB columns)
    # Split into two groups for biggest marking
    smol_loss_raw = [row[:n_algos] for row in all_loss_raw]
    normal_loss_raw = [row[n_algos:] for row in all_loss_raw]
    smol_token_raw = [row[:n_algos] for row in all_token_raw]
    normal_token_raw = [row[n_algos:] for row in all_token_raw]
    smol_vocab_raw = [row[:n_algos] for row in all_vocab_raw]
    normal_vocab_raw = [row[n_algos:] for row in all_vocab_raw]

    smol_loss_fmt = format_and_mark_biggest(smol_loss_raw, format_latex_value)
    normal_loss_fmt = format_and_mark_biggest(normal_loss_raw, format_latex_value)
    smol_token_fmt = format_and_mark_biggest(smol_token_raw, format_latex_value)
    normal_token_fmt = format_and_mark_biggest(normal_token_raw, format_latex_value)

    smol_vocab_diff = [[100 - v if not pd.isna(v) else float("nan") for v in row] for row in smol_vocab_raw]
    normal_vocab_diff = [[100 - v if not pd.isna(v) else float("nan") for v in row] for row in normal_vocab_raw]
    smol_vocab_fmt = format_and_mark_biggest(
        smol_vocab_diff, lambda v: format_vocab_value(100 - v) if not pd.isna(v) else "---"
    )
    normal_vocab_fmt = format_and_mark_biggest(
        normal_vocab_diff, lambda v: format_vocab_value(100 - v) if not pd.isna(v) else "---"
    )

    # Recombine formatted values
    all_loss_fmt = [smol_loss_fmt[i] + normal_loss_fmt[i] for i in range(len(CORPUS_ORDER_INIT))]
    all_token_fmt = [smol_token_fmt[i] + normal_token_fmt[i] for i in range(len(CORPUS_ORDER_INIT))]
    all_vocab_fmt = [smol_vocab_fmt[i] + normal_vocab_fmt[i] for i in range(len(CORPUS_ORDER_INIT))]

    # Build table
    total_cols = 2 * n_algos  # 30MB + 300MB
    lines = []

    # Corpus rows
    for corpus_idx, corpus in enumerate(CORPUS_ORDER_INIT):
        display_name = CORPUS_DISPLAY.get(corpus, corpus)

        loss_parts = [f"\\multirow{{3}}{{*}}{{{display_name}}}", r"\textit{Loss}"]
        loss_parts.extend(all_loss_fmt[corpus_idx])
        lines.append(" & ".join(loss_parts) + r" \\")

        token_parts = [" ", r"\textit{Tokens}"]
        token_parts.extend(all_token_fmt[corpus_idx])
        lines.append(" & ".join(token_parts) + r" \\")

        vocab_parts = [" ", r"\textit{Vocabulary Overlap}"]
        vocab_parts.extend(all_vocab_fmt[corpus_idx])
        lines.append(" & ".join(vocab_parts) + r" \\")

        lines.append(f"\\cmidrule{{2-{2 + total_cols}}}")

    # Mean row
    smol_loss_means = compute_mean_values(smol_loss_raw)
    normal_loss_means = compute_mean_values(normal_loss_raw)
    smol_token_means = compute_mean_values(smol_token_raw)
    normal_token_means = compute_mean_values(normal_token_raw)
    smol_vocab_means = compute_mean_values(smol_vocab_raw)
    normal_vocab_means = compute_mean_values(normal_vocab_raw)

    mean_loss_parts = [r"\multirow{3}{*}{Mean}", r"\textit{Loss}"]
    mean_loss_parts.extend([format_latex_value(v) for v in smol_loss_means])
    mean_loss_parts.extend([format_latex_value(v) for v in normal_loss_means])
    lines.append(" & ".join(mean_loss_parts) + r" \\")

    mean_token_parts = [" ", r"\textit{Tokens}"]
    mean_token_parts.extend([format_latex_value(v) for v in smol_token_means])
    mean_token_parts.extend([format_latex_value(v) for v in normal_token_means])
    lines.append(" & ".join(mean_token_parts) + r" \\")

    mean_vocab_parts = [" ", r"\textit{Vocabulary Overlap}"]
    mean_vocab_parts.extend([format_vocab_value(v) for v in smol_vocab_means])
    mean_vocab_parts.extend([format_vocab_value(v) for v in normal_vocab_means])
    lines.append(" & ".join(mean_vocab_parts) + r" \\")

    return "\n".join(lines), True


def build_fsp_subtable() -> tuple[str, bool]:
    """Build FSP comparison subtable.

    Returns (latex_string, has_data).
    """
    print("\nLoading FSP data...")

    df = load_experiment_results("fsp", corpus_names=CORPUS_NAMES)

    if df.empty:
        print("  ✗ No FSP data found")
        return "", False

    # Check which corpora have data
    available_corpora = []
    for corpus in CORPUS_ORDER_INIT:
        corpus_df = df[df["corpus"] == corpus]
        if len(corpus_df) > 0:
            available_corpora.append(corpus)
            alphas = sorted(corpus_df["pruning_shrinking_factor"].unique())
            print(f"  ✓ {corpus}: α={alphas}")
        else:
            print(f"  ✗ {corpus}: no data")

    if not available_corpora:
        return "", False

    # Compute relative performance and vocab overlap vs baseline
    rel_df = compute_relative_performance(df, load_baseline_model)
    vocab_df = compute_vocab_overlap(df, load_baseline_model)

    # Collect data
    all_loss_raw = []
    all_token_raw = []
    all_vocab_raw = []

    for corpus in available_corpora:
        loss_vals, token_vals, vocab_vals, _ = get_metric_values(
            rel_df, vocab_df, "pruning_shrinking_factor", FSP_ALPHA_VALUES, corpus
        )
        all_loss_raw.append(loss_vals)
        all_token_raw.append(token_vals)
        all_vocab_raw.append(vocab_vals)

    # Format and mark biggest
    all_loss_formatted = format_and_mark_biggest(all_loss_raw, format_latex_value)
    all_token_formatted = format_and_mark_biggest(all_token_raw, format_latex_value)
    vocab_diff_raw = [[100 - v if not pd.isna(v) else float("nan") for v in row] for row in all_vocab_raw]
    all_vocab_formatted = format_and_mark_biggest(
        vocab_diff_raw, lambda v: format_vocab_value(100 - v) if not pd.isna(v) else "---"
    )

    # Build table rows
    n_cols = len(FSP_ALPHA_VALUES)
    lines = []

    # Corpus rows
    for corpus_idx, corpus in enumerate(available_corpora):
        display_name = CORPUS_DISPLAY.get(corpus, corpus)

        loss_parts = [f"\\multirow{{3}}{{*}}{{{display_name}}}", r"\textit{Loss}"]
        loss_parts.extend(all_loss_formatted[corpus_idx])
        lines.append(" & ".join(loss_parts) + r" \\")

        token_parts = [" ", r"\textit{Tokens}"]
        token_parts.extend(all_token_formatted[corpus_idx])
        lines.append(" & ".join(token_parts) + r" \\")

        vocab_parts = [" ", r"\textit{Vocabulary Overlap}"]
        vocab_parts.extend(all_vocab_formatted[corpus_idx])
        lines.append(" & ".join(vocab_parts) + r" \\")

        lines.append(f"\\cmidrule{{2-{2 + n_cols}}}")

    # Mean row
    loss_means = compute_mean_values(all_loss_raw)
    token_means = compute_mean_values(all_token_raw)
    vocab_means = compute_mean_values(all_vocab_raw)

    mean_loss_parts = [r"\multirow{3}{*}{Mean}", r"\textit{Loss}"]
    mean_loss_parts.extend([format_latex_value(v) for v in loss_means])
    lines.append(" & ".join(mean_loss_parts) + r" \\")

    mean_token_parts = [" ", r"\textit{Tokens}"]
    mean_token_parts.extend([format_latex_value(v) for v in token_means])
    lines.append(" & ".join(mean_token_parts) + r" \\")

    mean_vocab_parts = [" ", r"\textit{Vocabulary Overlap}"]
    mean_vocab_parts.extend([format_vocab_value(v) for v in vocab_means])
    lines.append(" & ".join(mean_vocab_parts) + r" \\")

    return "\n".join(lines), True


def generate_init_and_fsp_table() -> str:
    """Generate the seed vocabulary algorithm and FSP appendix table."""
    print("\n" + "=" * 70)
    print("GENERATING TABLE: Seed Vocabulary Algorithm and Final Style Prune (Appendix)")
    print("=" * 70)

    lines = []
    # Section header (this table comes first in the appendix)
    lines.append(r"\section{Detailed results of Unigram hyperparameter variation\label{app:hyperparams}}")
    lines.append("")
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Seed Vocabulary Algorithm and Final Style Prune Results: Relative Change (\%) vs Baseline, lower is better. Each corpus shows loss (relative change), token count (relative change), and vocabulary overlap (absolute \%, 100\% = identical to baseline), with \goodoutlier{good} and \badoutlier{bad} changes above 0.5\% highlighted along with \biggestoutlier{the largest} absolute change per parameter.}"
    )
    lines.append(r"\label{tab:app_seed_and_final}")
    lines.append("")

    # Seed Vocabulary Algorithm Subtable with 30MB and 300MB as column groups
    lines.append(r"% Seed Vocabulary Algorithm Comparison: Relative performance. Baseline: Long (pretoken-based)")
    lines.append(r"\begin{subtable}{\textwidth}")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Seed Vocabulary Algorithm Comparison: Relative performance compared to pretoken-based without prefix recovery.}"
    )

    # Column headers for init algo: 2 groups (30MB, 300MB) x 3 algos
    n_algos = len(INIT_ALGO_ORDER)
    total_cols = 2 * n_algos
    lines.append(f"\\begin{{tabular}}{{ll|{'r' * n_algos}|{'r' * n_algos}}}")
    lines.append(r"\toprule")

    # First header row: corpus size groups
    lines.append(
        f" &  & \\multicolumn{{{n_algos}}}{{c|}}{{30\\,MB}} & \\multicolumn{{{n_algos}}}{{c}}{{300\\,MB}} \\\\"
    )
    lines.append(f"\\cmidrule(lr){{3-{2 + n_algos}}} \\cmidrule(lr){{{3 + n_algos}-{2 + total_cols}}}")

    # Second header row: algorithm names (top line)
    header_top = [" ", " ", "Full text", "Full text", "Pretokens", "Full text", "Full text", "Pretokens"]
    lines.append(" & ".join(header_top) + r" \\")

    # Third header row: "recovery" where applicable
    header_bot = ["Corpus", "Metric", "", "recovery", "recovery", "", "recovery", "recovery"]
    lines.append(" & ".join(header_bot) + r" \\")
    lines.append(r"\midrule")

    # Combined content
    content, has_data = build_init_algo_combined_subtable()
    if has_data:
        lines.append(content)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{subtable}")
    lines.append("")

    # FSP Subtable
    lines.append(
        r"% Final Style Prune (FSP): Relative performance. Standard configuration with final_style_prune=True, pre_final_vocab_factor=1.0"
    )
    lines.append(r"\begin{subtable}{\textwidth}")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Final Style Prune: Relative performance compared to default settings, using varying pruning shrinking factor $\alphaprune$. We use $\overshootfactor=1$ for these experiments, disabling the final pruning step, as it would be identical to normal pruning steps.}"
    )

    n_fsp_cols = len(FSP_ALPHA_VALUES)
    lines.append(f"\\begin{{tabular}}{{ll|{'r' * n_fsp_cols}}}")
    lines.append(r"\toprule")

    # Header
    header = ["Corpus", "Metric"] + [str(v) for v in FSP_ALPHA_VALUES]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    fsp_content, fsp_has_data = build_fsp_subtable()
    if fsp_has_data:
        lines.append(fsp_content)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{subtable}")
    lines.append("")

    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    """Generate appendix tables."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "table_hyperparams_appendix.tex"
    # Lead with the interesting tables (seed algo & FSP), then hyperparameter variation
    table1 = generate_init_and_fsp_table()
    table2 = generate_hyperparams_table()
    output_path.write_text(table1 + "\n\n" + table2)
    print(f"\n✓ Written: {output_path}")


if __name__ == "__main__":
    main()
