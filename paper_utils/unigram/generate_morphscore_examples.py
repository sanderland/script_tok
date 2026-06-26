#!/usr/bin/env python3
"""Generate MorphScore appendix table comparing tokenizers at 32K vocab size."""

from script_bpe.analysis import MorphScore
from paper_utils.unigram.train_hyperparameters import (
    DEFAULTS,
    ADDITIONAL_VOCAB_SIZE,
    RESULTS_DIR,
    get_model_path,
)
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel

RESULTS_BASE = RESULTS_DIR
CORPUS = "eng_latn_300mb"
MORPHSCORE_LANG = "eng_latn"

# FSP params
FSP_PARAMS = {**DEFAULTS, "flat_score_prune": True, "pre_final_vocab_factor": 1.0}

# Variant names for display (Baseline instead of Default for paper consistency)
VARIANTS = ["Baseline", "FSP", "BPE"]


def load_tokenizers() -> dict[str, object]:
    """Load the three tokenizers at 32K vocab size."""
    tokenizers = {}

    # Baseline (default unigram)
    baseline_path = get_model_path(CORPUS, DEFAULTS)
    print(f"Loading Baseline from {baseline_path.name}")
    tokenizers["Baseline"] = UnigramModel.load(str(baseline_path))

    # FSP
    fsp_path = get_model_path(CORPUS, FSP_PARAMS)
    print(f"Loading FSP from {fsp_path.name}")
    tokenizers["FSP"] = UnigramModel.load(str(fsp_path))

    # BPE
    bpe_path = RESULTS_BASE / CORPUS / f"bpe_n{ADDITIONAL_VOCAB_SIZE}.model.json.gz"
    print(f"Loading BPE from {bpe_path.name}")
    tokenizers["BPE"] = BPETokenizer.load(str(bpe_path))

    return tokenizers


def escape_latex(s: str) -> str:
    """Escape LaTeX special characters."""
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")


def fmt_tokenization(tokens: list[str]) -> str:
    """Format tokenization as 'tok1 | tok2 | tok3'."""
    return escape_latex(" | ".join(tokens))


def generate_latex(tokenizers: dict[str, object]) -> str:
    """Generate the complete LaTeX output."""
    ms = MorphScore(language_subset=[MORPHSCORE_LANG], stem_eq_lemma=True, exclude_single_tok=False)

    # Get word frequencies
    word_freqs = ms.get_word_frequencies()

    # Analyze each tokenizer
    analyses = {name: ms.analyze_tokenizer(tok) for name, tok in tokenizers.items()}

    # Build per-word comparison (index by word)
    words_data = {}
    for word_result in analyses["Baseline"]:
        word = word_result["word"]
        words_data[word] = {
            "gold": word_result["gold"],
            "freq": word_freqs.get(word, 0),
            "tokenizations": {"Baseline": word_result["predicted"]},
            "recalls": {"Baseline": word_result["recall"]},
            "is_single": {"Baseline": word_result["is_single"]},
        }

    for variant in ["FSP", "BPE"]:
        for word_result in analyses[variant]:
            word = word_result["word"]
            if word in words_data:
                words_data[word]["tokenizations"][variant] = word_result["predicted"]
                words_data[word]["recalls"][variant] = word_result["recall"]
                words_data[word]["is_single"][variant] = word_result["is_single"]

    # Filter to words present in all tokenizers with valid recalls
    complete_words = []
    for word, data in words_data.items():
        if len(data["tokenizations"]) == 3 and all(data["recalls"].get(v) is not None for v in VARIANTS):
            complete_words.append((word, data))

    # Sort by frequency descending
    complete_words.sort(key=lambda x: x[1]["freq"], reverse=True)

    # Compute summary statistics
    total_freq_sum = sum(d["freq"] for _, d in complete_words)

    scores = {}
    for variant in VARIANTS:
        recalls = [d["recalls"][variant] for _, d in complete_words]
        freqs = [d["freq"] for _, d in complete_words]
        scores[variant] = {
            "mean": sum(recalls) / len(recalls),
            "weighted": sum(r * f for r, f in zip(recalls, freqs)) / total_freq_sum,
        }

    # Categorize words
    all_correct = []
    all_incorrect = []
    unique_wins = {v: [] for v in VARIANTS}

    for word, data in complete_words:
        recalls = data["recalls"]
        max_recall = max(recalls.values())
        winners = [v for v in VARIANTS if recalls[v] == max_recall]

        # Skip single-token words for examples (not interesting)
        all_single = all(data["is_single"].get(v, False) for v in VARIANTS)

        # For "all correct", require actual match with gold (not just recall=1.0 from single-token)
        all_match_gold = all(data["tokenizations"][v] == data["gold"] for v in VARIANTS)

        if all_match_gold:
            all_correct.append((word, data))
        elif all(recalls[v] == 0.0 for v in VARIANTS):
            if not all_single:
                all_incorrect.append((word, data))
        elif len(winners) == 1:
            # Unique win - only one method matches best
            # Skip if the winner is single-token (not a real morphological win)
            winner = winners[0]
            if not all_single and not data["is_single"].get(winner, False):
                unique_wins[winner].append((word, data))
        # else: shared wins (multiple methods tie) - not shown in table

    # Start LaTeX output
    latex = r"""\section{MorphScore Evaluation}

\label{app:morphscore}

Table~\ref{tab:morphscore-examples} shows MorphScore evaluation results comparing our Baseline unigram tokenizer, the FSP (Flat-Score Prune) variant, and BPE on English morphological segmentation.

\begin{table}[H]
\centering
\small
\begin{tabular}{@{}lrp{2.0cm}p{2.0cm}p{2.0cm}p{2.0cm}@{}}
\toprule
\textbf{Word} & \textbf{Freq} & \textbf{Gold} & \textbf{Baseline} & \textbf{FSP} & \textbf{BPE} \\
\midrule
"""

    def add_examples(
        category_name: str, examples: list, num: int = 3, color_winner: str | None = None, all_red: bool = False
    ):
        """Add example rows for a category."""
        nonlocal latex
        # Compute stats for this category
        cat_count = len(examples)
        cat_freq_pct = 100 * sum(d["freq"] for _, d in examples) / total_freq_sum if total_freq_sum > 0 else 0

        latex += (
            rf"\multicolumn{{6}}{{l}}{{\textbf{{{category_name}}} ({cat_count} words covering {cat_freq_pct:.1f}\% of corpus)}} \\"
            + "\n"
        )
        latex += r"\midrule" + "\n"

        for word, data in examples[:num]:
            freq_pct = 100 * data["freq"] / total_freq_sum
            gold_str = fmt_tokenization(data["gold"])

            tok_strs = []
            for v in VARIANTS:
                tok_str = fmt_tokenization(data["tokenizations"][v])
                if all_red:
                    tok_str = rf"\nomatchcolor{{{tok_str}}}"
                elif color_winner and v == color_winner:
                    tok_str = rf"\matchcolor{{{tok_str}}}"
                elif color_winner is None and data["recalls"][v] == 1.0:
                    # All correct case
                    tok_str = rf"\matchcolor{{{tok_str}}}"
                tok_strs.append(tok_str)

            latex += f"{escape_latex(word)} & {freq_pct:.1f}\\% & {gold_str} & {' & '.join(tok_strs)} \\\\\n"

        latex += r"\midrule" + "\n"

    # Add categories
    add_examples("All Methods Match Gold", all_correct, num=3)
    add_examples("Only Baseline Matches", unique_wins["Baseline"], num=3, color_winner="Baseline")
    add_examples("Only FSP Matches", unique_wins["FSP"], num=3, color_winner="FSP")
    add_examples("Only BPE Matches", unique_wins["BPE"], num=3, color_winner="BPE")
    add_examples("No Method Matches", all_incorrect, num=3, all_red=True)

    # Remove last \midrule and close table
    latex = latex.rsplit(r"\midrule", 1)[0]

    latex += r"""\bottomrule
\end{tabular}
\caption{MorphScore examples for English, grouped by which methods match gold morpheme boundaries. Each category shows the three most frequent words meeting that criterion. Freq is the word's share of total evaluation frequency.}
\label{tab:morphscore-examples}
\end{table}
"""

    return latex


def main():
    print(f"Generating MorphScore appendix table for vocab size {ADDITIONAL_VOCAB_SIZE}")

    tokenizers = load_tokenizers()
    latex = generate_latex(tokenizers)

    output_path = RESULTS_BASE / "table_morphscore_appendix.tex"
    output_path.write_text(latex)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
