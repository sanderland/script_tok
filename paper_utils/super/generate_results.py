#!/usr/bin/env python3
"""Generate results for supertoken experiments."""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from script_bpe import get_pretokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from script_bpe.train import train_tokenizer

from paper_utils.super.utils import PRETOKENIZER_NAME, RESULTS_DIR

# ========== CONFIG ==========

CORPUS = "eng_latn_300mb"
VB = 32000


def is_supertoken(pretokenizer, atomic_tokens: tuple[int, ...]) -> bool:
    """A supertoken spans more than 1 pretoken when decoded and re-pretokenized."""
    text = pretokenizer.decode(atomic_tokens)
    pretokens = pretokenizer.pretokenize(text)
    return len(pretokens) > 1


def load_all_models() -> list[tuple[Path, UnigramModel, dict]]:
    """Load all supertoken models and their configs."""
    models = []
    results_dir = RESULTS_DIR / CORPUS / f"vb{VB}" / PRETOKENIZER_NAME

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return []

    for path in sorted(results_dir.glob("super_*.json.gz")):
        model = UnigramModel.load(str(path))
        config = model.metadata.get("supertoken_config", {})
        models.append((path, model, config))

    return models


def load_baseline() -> UnigramModel | None:
    """Load baseline unigram model (no supertokens)."""
    baseline = train_tokenizer(
        pretokenizer_name=PRETOKENIZER_NAME,
        model_name="unigram",
        corpus_name=CORPUS,
        additional_vocab_size=VB,
        n_cpus=4,
        retrain=False,
    )
    return baseline


def count_supertokens(model: UnigramModel, word_pretokenizer) -> int:
    """Count supertokens in a model using word pretokenizer definition."""
    count = 0
    for token in model.tokens.values():
        if len(token.atomic_tokens) > 1:  # Skip base tokens
            if is_supertoken(word_pretokenizer, tuple(token.atomic_tokens)):
                count += 1
    return count


def collect_model_data(models: list, baseline: UnigramModel, word_pretokenizer) -> pd.DataFrame:
    """Collect data from all models."""
    baseline_objective = baseline.metadata["objective"]
    baseline_tokens = baseline.metadata["total_tokens"]

    data = []
    for path, model, config in models:
        objective = model.metadata.get("objective", 0)
        tokens = model.metadata.get("total_tokens", 0)

        rel_objective = (objective - baseline_objective) / baseline_objective * 100
        rel_tokens = (tokens - baseline_tokens) / baseline_tokens * 100

        va = config.get("vocab_a", 0)
        max_ngram = config.get("max_ngram", 0)
        filter_name = config.get("filter_name", "")
        fsp = config.get("fsp", False)

        # Count supertokens
        n_supertokens = count_supertokens(model, word_pretokenizer)

        data.append({
            "path": str(path),
            "vocab_a": va,
            "max_ngram": max_ngram,
            "filter": filter_name,
            "fsp": fsp,
            "objective": objective,
            "tokens": tokens,
            "rel_objective": rel_objective,
            "rel_tokens": rel_tokens,
            "n_supertokens": n_supertokens,
            "tokens_per_pretoken": model.metadata.get("tokens/pretoken", 0),
        })

    return pd.DataFrame(data)


def generate_supertoken_scatter(df: pd.DataFrame):
    """Generate scatter plot: supertoken count vs compression."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color by filter - main filters get distinct colors, patterns get muted
    colors = {
        # Main filters
        "all": "#1f77b4",
        "semantic": "#d62728",
        "all_patterns": "#2ca02c",
        # Individual patterns (muted)
        "space_phrase": "#aec7e8",
        "punct_trans": "#ffbb78",
        "contraction": "#98df8a",
        "suffix": "#ff9896",
        "abbrev": "#c5b0d5",
        "hyphen": "#c49c94",
        "domain": "#f7b6d2",
        # Legacy
        "len_8c": "#9467bd",
    }
    markers = {4: "s", 8: "^"}

    for _, row in df.iterrows():
        color = colors.get(row["filter"], "#333333")
        marker = markers.get(row["max_ngram"], "o")
        edge = "red" if row["fsp"] else "none"

        ax.scatter(
            row["n_supertokens"],
            row["rel_tokens"],
            c=color,
            marker=marker,
            s=120,
            edgecolors=edge,
            linewidths=2 if row["fsp"] else 0,
            alpha=0.8,
        )

    ax.set_xlabel("Supertoken Count", fontsize=12)
    ax.set_ylabel("Relative Tokens (%)", fontsize=12)
    ax.set_title(f"Supertokens vs Compression: {CORPUS}", fontsize=14)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    # Legend - only show filters that appear in data
    from matplotlib.lines import Line2D
    legend_elements = []
    used_filters = set(df["filter"].unique())
    for filter_name, color in colors.items():
        if filter_name in used_filters:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=filter_name)
            )
    for n, marker in markers.items():
        legend_elements.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray", markersize=10, label=f"N={n}")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="red", markeredgewidth=2, markersize=10, label="FSP")
    )

    ax.legend(handles=legend_elements, loc="best", fontsize=9)

    output_path = RESULTS_DIR / "supertoken_scatter.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved scatter plot to {output_path}")


def generate_comparison_table(df: pd.DataFrame, baseline: UnigramModel):
    """Generate markdown comparison table."""
    baseline_objective = baseline.metadata["objective"]
    baseline_tokens = baseline.metadata["total_tokens"]

    lines = [
        "# Supertoken Experiment Results",
        "",
        f"Baseline: objective={baseline_objective:.4f}, tokens={baseline_tokens/1e6:.2f}M",
        "",
        "| VA | N | Filter | FSP | Supertokens | Objective | Δ Obj (%) | Tokens (M) | Δ Tok (%) |",
        "|---|---|--------|-----|-------------|-----------|-----------|------------|-----------|",
    ]

    df_sorted = df.sort_values("rel_objective")
    for _, row in df_sorted.iterrows():
        fsp = "✓" if row["fsp"] else ""
        lines.append(
            f"| {row['vocab_a']} | {row['max_ngram']} | {row['filter']} | {fsp} | "
            f"{row['n_supertokens']:,} | {row['objective']:.4f} | {row['rel_objective']:+.2f} | "
            f"{row['tokens']/1e6:.2f} | {row['rel_tokens']:+.2f} |"
        )

    output_path = RESULTS_DIR / "comparison_table.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved comparison table to {output_path}")


def generate_filter_bar_chart(df: pd.DataFrame):
    """Generate bar chart comparing filters (supertoken count + compression)."""
    # Focus on main experiment (VA=2*VB, N=4, no FSP)
    df_main = df[(df["max_ngram"] == 4) & (~df["fsp"])]
    if df_main.empty:
        print("No data for filter bar chart")
        return

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Sort by supertoken count
    df_sorted = df_main.sort_values("n_supertokens", ascending=False)

    # Left: Supertoken count
    ax1 = axes[0]
    colors = ["#d62728" if f in ("semantic", "all", "all_patterns") else "#aec7e8" for f in df_sorted["filter"]]
    ax1.barh(df_sorted["filter"], df_sorted["n_supertokens"], color=colors)
    ax1.set_xlabel("Supertoken Count")
    ax1.set_title("Surviving Supertokens by Filter")
    ax1.invert_yaxis()

    # Right: Relative tokens (compression)
    ax2 = axes[1]
    colors = ["green" if t < 0 else "red" for t in df_sorted["rel_tokens"]]
    ax2.barh(df_sorted["filter"], df_sorted["rel_tokens"], color=colors, alpha=0.7)
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Δ Tokens (%)")
    ax2.set_title("Compression Change by Filter")
    ax2.invert_yaxis()

    plt.tight_layout()
    output_path = RESULTS_DIR / "filter_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved filter comparison to {output_path}")


def generate_top_supertokens(models: list, word_pretokenizer, n: int = 50):
    """Generate markdown table of top N most common supertokens."""
    supertoken_counts: Counter[tuple[int, ...]] = Counter()

    for path, model, config in models:
        for token in model.tokens.values():
            seq = tuple(token.atomic_tokens)
            if len(seq) > 1 and is_supertoken(word_pretokenizer, seq):
                supertoken_counts[seq] += 1

    top_supertokens = supertoken_counts.most_common(n)

    lines = [
        f"# Top {n} Most Common Supertokens",
        "",
        f"Across {len(models)} supertoken models on {CORPUS}.",
        "",
        "| Rank | Token | #PT | Models |",
        "|------|-------|-----|--------|",
    ]

    for rank, (seq, count) in enumerate(top_supertokens, 1):
        text = word_pretokenizer.decode(seq)
        n_pretokens = len(word_pretokenizer.pretokenize(text))
        text_escaped = repr(text).replace("|", "\\|")
        lines.append(f"| {rank} | {text_escaped} | {n_pretokens} | {count}/{len(models)} |")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- Total unique supertokens: {len(supertoken_counts):,}",
        f"- Supertokens in all models: {sum(1 for c in supertoken_counts.values() if c == len(models)):,}",
    ])

    output_path = RESULTS_DIR / "top_supertokens.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved top supertokens to {output_path}")


if __name__ == "__main__":
    print("Generating supertoken results...")
    print()

    word_pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    models = load_all_models()
    baseline = load_baseline()

    if not models:
        print("No models found!")
        exit(1)

    if baseline is None:
        print("Baseline not found!")
        exit(1)

    print("=== Collecting Data ===")
    df = collect_model_data(models, baseline, word_pretokenizer)

    # Save CSV
    csv_path = RESULTS_DIR / "results_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV to {csv_path}")
    print()

    print("=== Supertoken Scatter ===")
    generate_supertoken_scatter(df)
    print()

    print("=== Filter Comparison ===")
    generate_filter_bar_chart(df)
    print()

    print("=== Comparison Table ===")
    generate_comparison_table(df, baseline)
    print()

    print("=== Top Supertokens ===")
    generate_top_supertokens(models, word_pretokenizer, 50)
    print()

    print("Done!")
