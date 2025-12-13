#!/usr/bin/env python3
"""Generate scatter plot showing hyperparameter trade-offs."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    SWEEP_CONFIGS,
    ADDITIONAL_VOCAB_SIZE,
    RESULTS_DIR,
    load_baseline_model,
    load_experiment_results,
)
from script_bpe.tokenizers.bpe import BPETokenizer

# Fixed axis limits
XLIM = (-2, 2)
YLIM = (-1, 1)

# Minimum difference threshold (%) - points with smaller difference will be skipped
MIN_DIFF_THRESHOLD = 0.05

# Experiment display configuration
EXPERIMENT_DISPLAY = {
    "initial_vocab_factor": {
        "color": "#1f77b4",
        "marker": "o",
        "name": r"Seed Vocab Factor ($\beta_{\mathrm{seed}}$)",
    },
    "init_algo": {"color": "#17becf", "marker": "s", "name": "Initialization Algorithm"},
    "m_step_digamma": {"color": "#d62728", "marker": "v", "name": "M-Step Digamma"},
    "m_step_low_count_threshold": {
        "color": "#9467bd",
        "marker": "D",
        "name": r"M-Step Low Count Threshold ($\tau_{\mathrm{mp}}$)",
    },
    "num_sub_iterations": {"color": "#8c564b", "marker": "P", "name": r"Number of Sub-Iterations ($N_{\mathrm{em}}$)"},
    "pre_final_vocab_factor": {
        "color": "#ff7f0e",
        "marker": "p",
        "name": r"Pre-Final Vocab Factor ($\alpha_{\mathrm{inter}}$)",
    },
    "pruning_shrinking_factor": {
        "color": "#2ca02c",
        "marker": "h",
        "name": r"Pruning Shrinking Factor ($\alpha_{\mathrm{prune}}$)",
    },
    "fsp": {"color": "#e377c2", "marker": "*", "name": "Final Style Prune"},
}
PLOT_ALL_POINTS = False


def create_scatter_plot_all_points() -> tuple[list[dict], list[dict], list[dict]]:
    """Create scatter plot with ALL individual points plus means.

    Returns (all_points, out_of_bounds, skipped_experiments) for reporting.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Load baselines for both regular and smol corpora
    baselines = {}
    for corpus_name in CORPUS_NAMES:
        baseline = load_baseline_model(corpus_name)
        if baseline is not None:
            baselines[corpus_name] = baseline

    for corpus_name in CORPUS_NAMES:
        smol_corpus_name = "smol_" + corpus_name
        baseline = load_baseline_model(smol_corpus_name)
        if baseline is not None:
            baselines[smol_corpus_name] = baseline

    all_points = []  # Track all points for distance reporting
    experiment_points = {}  # Track points per experiment for threshold check

    # FIRST PASS: Collect all points and calculate their distances
    print("\nFirst pass: Collecting all data points...")
    for experiment_name, display_config in EXPERIMENT_DISPLAY.items():
        print(f"  Collecting {experiment_name}...")

        # Determine corpus names to use
        if experiment_name == "init_no_pt":
            corpus_names_for_exp = ["smol_" + c for c in CORPUS_NAMES]
        else:
            corpus_names_for_exp = CORPUS_NAMES

        # Load this experiment's results
        df = load_experiment_results(experiment_name, corpus_names=corpus_names_for_exp)

        if df.empty:
            print(f"    No data for {experiment_name}")
            continue

        # Determine which parameter is being swept
        if experiment_name in SWEEP_CONFIGS:
            param_col = experiment_name
        elif experiment_name == "fsp":
            param_col = "pruning_shrinking_factor"
        elif experiment_name in ["init_algo", "init_no_pt"]:
            param_col = "init_vocab_algo"
        elif experiment_name in ["bpe_init", "bpe_init_fsp"]:
            param_col = "bpe_init_factor"
        elif experiment_name == "token_bias_fsp":
            param_col = "token_bias"
        else:
            print(f"    Unknown sweep parameter for {experiment_name}")
            continue

        # Get unique parameter values
        param_values = df[param_col].unique()

        experiment_points[experiment_name] = []

        # For each unique parameter value, compute normalized metrics
        for param_val in param_values:
            param_data = df[df[param_col] == param_val]

            points_for_this_param = []
            for corpus_name in corpus_names_for_exp:
                if corpus_name not in baselines:
                    continue

                baseline = baselines[corpus_name]
                corpus_data = param_data[param_data["corpus"] == corpus_name]

                for _, row in corpus_data.iterrows():
                    rel_objective = (row["objective"] - baseline["objective"]) / baseline["objective"] * 100
                    rel_tokens = (row["tokens"] - baseline["tokens"]) / baseline["tokens"] * 100
                    points_for_this_param.append(
                        {
                            "corpus": corpus_name,
                            "rel_objective": rel_objective,
                            "rel_tokens": rel_tokens,
                        }
                    )

            if not points_for_this_param:
                continue

            param_points_df = pd.DataFrame(points_for_this_param)

            # Compute mean across corpora for this parameter value
            mean_obj = param_points_df["rel_objective"].mean()
            mean_tokens = param_points_df["rel_tokens"].mean()

            # Track point
            point_info = {
                "experiment": experiment_name,
                "param_value": param_val,
                "x": mean_obj,
                "y": mean_tokens,
                "dist": abs(mean_obj) + abs(mean_tokens),
                "individual_points": points_for_this_param,
            }
            all_points.append(point_info)
            experiment_points[experiment_name].append(point_info)

    # Add BPE comparison to the collection
    print("  Collecting BPE baseline...")
    bpe_individual = []
    for corpus_name in CORPUS_NAMES:
        bpe_file = RESULTS_DIR / corpus_name / f"bpe_n{ADDITIONAL_VOCAB_SIZE}.model.json.gz"
        if not bpe_file.exists():
            continue
        bpe_model = BPETokenizer.load(str(bpe_file))
        baseline = load_baseline_model(corpus_name)
        if baseline is not None:
            bpe_tokens = bpe_model.metadata["performance"]["total_tokens_len"]
            rel_bpe_tokens = (bpe_tokens - baseline["tokens"]) / baseline["tokens"] * 100
            bpe_individual.append(rel_bpe_tokens)

    if bpe_individual:
        bpe_mean = np.mean(bpe_individual)
        bpe_point = {
            "experiment": "BPE",
            "param_value": "baseline",
            "x": 0.0,
            "y": bpe_mean,
            "dist": abs(bpe_mean),
            "individual_points": bpe_individual,
        }
        all_points.append(bpe_point)
        experiment_points["BPE"] = [bpe_point]

    # SECOND PASS: Determine which experiments to include based on max distance
    print("\nSecond pass: Determining which experiments to plot...")
    experiments_to_plot = []
    skipped_experiments = []
    
    for experiment_name in list(EXPERIMENT_DISPLAY.keys()) + ["BPE"]:
        if experiment_name not in experiment_points or not experiment_points[experiment_name]:
            continue
        
        # Find max distance for this experiment
        max_dist = max(pt["dist"] for pt in experiment_points[experiment_name])
        
        if max_dist >= MIN_DIFF_THRESHOLD:
            experiments_to_plot.append(experiment_name)
            print(f"  ✓ Including {experiment_name} (max dist: {max_dist:.3f}%)")
        else:
            skipped_experiments.append(experiment_name)
            print(f"  ⊘ Skipping {experiment_name} (max dist: {max_dist:.3f}% < {MIN_DIFF_THRESHOLD}%)")

    # THIRD PASS: Create the plot with only included experiments
    print("\nThird pass: Creating plot...")
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    plotted_experiments = []
    out_of_bounds = []

    for experiment_name in experiments_to_plot:
        if experiment_name == "BPE":
            # Handle BPE specially
            bpe_point = experiment_points["BPE"][0]
            bpe_individual = bpe_point["individual_points"]
            bpe_mean = bpe_point["y"]
            
            # Plot individual BPE points (small)
            if PLOT_ALL_POINTS:
                ax.scatter(
                    [0] * len(bpe_individual),
                    bpe_individual,
                    c="black",
                    marker="x",
                    s=50,
                    linewidths=2,
                    alpha=0.3,
                    zorder=9,
                )
            # Plot mean BPE point (large)
            ax.scatter([0], [bpe_mean], c="black", marker="x", s=150, linewidths=4, alpha=1.0, zorder=10)
            print(f"  ✓ Plotted {len(bpe_individual)} BPE points")
            
            # Check for out-of-bounds
            if not (XLIM[0] <= 0 <= XLIM[1] and YLIM[0] <= bpe_mean <= YLIM[1]):
                out_of_bounds.append(bpe_point)
        else:
            # Handle regular experiments
            display_config = EXPERIMENT_DISPLAY[experiment_name]
            points = experiment_points[experiment_name]
            
            for point_info in points:
                # Plot individual corpus points if enabled
                if PLOT_ALL_POINTS and "individual_points" in point_info:
                    individual_df = pd.DataFrame(point_info["individual_points"])
                    ax.scatter(
                        individual_df["rel_objective"],
                        individual_df["rel_tokens"],
                        c=display_config["color"],
                        marker=display_config["marker"],
                        alpha=0.5,
                        s=60,
                        edgecolors="none",
                    )
                
                # Plot mean point
                ax.scatter(
                    [point_info["x"]],
                    [point_info["y"]],
                    c=display_config["color"],
                    marker=display_config["marker"],
                    alpha=1.0,
                    s=150,
                    edgecolors="black",
                    linewidths=1.5,
                )
                
                # Check for out-of-bounds
                if not (XLIM[0] <= point_info["x"] <= XLIM[1] and YLIM[0] <= point_info["y"] <= YLIM[1]):
                    out_of_bounds.append(point_info)
            
            plotted_experiments.append(experiment_name)
            print(f"  ✓ Plotted {len(points)} parameter values for {experiment_name}")

    # Reference lines and annotations
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    # Quadrant labels (all 4 corners)
    fontsize = 11
    ax.text(
        0.02,
        0.02,
        "✓ Better objective\n✓ Fewer tokens",
        transform=ax.transAxes,
        fontsize=fontsize,
        va="bottom",
        ha="left",
        color="green",
        style="italic",
        weight="bold",
    )
    ax.text(
        0.98,
        0.02,
        "✗ Worse objective\n✓ Fewer tokens",
        transform=ax.transAxes,
        fontsize=fontsize,
        va="bottom",
        ha="right",
        color="orange",
        style="italic",
        weight="bold",
    )
    ax.text(
        0.02,
        0.98,
        "✓ Better objective\n✗ More tokens",
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="left",
        color="orange",
        style="italic",
        weight="bold",
    )
    ax.text(
        0.98,
        0.98,
        "✗ Worse objective\n✗ More tokens",
        transform=ax.transAxes,
        fontsize=fontsize,
        va="top",
        ha="right",
        color="red",
        style="italic",
        weight="bold",
    )

    ax.set_xlabel("Change in Objective (%)\n← Better | Worse →", fontsize=14)
    ax.set_ylabel("Change in Token Count (%)\n← Better | Worse →", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Fixed axis limits
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_aspect("equal", adjustable="box")

    # Legend - simplified to just show parameter names with large markers
    handles = []
    for exp_name in plotted_experiments:
        display_config = EXPERIMENT_DISPLAY[exp_name]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=display_config["marker"],
                linestyle="None",
                markerfacecolor=display_config["color"],
                markeredgecolor="black",
                markeredgewidth=1.5,
                markersize=10,
                label=display_config["name"],
            )
        )

    # Only add BPE to legend if it was plotted
    if "BPE" in experiments_to_plot:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                color="black",
                markersize=12,
                markeredgewidth=3.0,
                label="BPE (objective n/a)",
            )
        )

    # Add note about small vs large markers
    if PLOT_ALL_POINTS:
        handles.insert(
            0,
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="gray",
                alpha=0.3,
                markersize=5,
                markeredgecolor="none",
                label="Individual corpus result",
            ),
        )
        handles.insert(
            1,
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="gray",
                alpha=1.0,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label="Mean across corpora",
            ),
        )

    ax.legend(
        handles=handles,
        fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        framealpha=0.95,
        title="Parameters",
        title_fontsize=12,
    )

    plt.tight_layout()
    output_path = RESULTS_DIR / "scatter.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"\n✓ Saved scatter plot to {output_path}")
    plt.show()

    return all_points, out_of_bounds, skipped_experiments


def main():
    """Main visualization workflow."""
    print("=" * 70)
    print("GENERATING: Unigram Hyperparameter Trade-off Scatter Plot")
    print(f"  Minimum difference threshold: {MIN_DIFF_THRESHOLD}%")
    print("=" * 70)
    all_points, out_of_bounds, skipped_experiments = create_scatter_plot_all_points()
    print("\n✓ Visualization complete")

    # Show max distance from origin per experiment
    print("\n" + "-" * 70)
    print("MAX DISTANCE FROM ORIGIN (|x| + |y|) PER EXPERIMENT:")
    print("-" * 70)
    exp_max_dist = {}
    for pt in all_points:
        exp = pt["experiment"]
        if exp not in exp_max_dist or pt["dist"] > exp_max_dist[exp]["dist"]:
            exp_max_dist[exp] = pt
    
    for pt in sorted(exp_max_dist.values(), key=lambda p: p["dist"], reverse=True):
        exp = pt["experiment"]
        skipped_str = " (SKIPPED)" if exp in skipped_experiments else ""
        print(f"  {pt['dist']:.3f}  {pt['experiment']:30s} (max at {pt['param_value']}){skipped_str}")

    # Report out-of-bounds points
    if out_of_bounds:
        print("\n" + "!" * 70)
        print("WARNING: DATA POINTS OUTSIDE PLOT BOUNDS")
        print(f"  xlim={XLIM}, ylim={YLIM}")
        print("!" * 70)
        for pt in out_of_bounds:
            print(f"  {pt['experiment']} ({pt['param_value']}): x={pt['x']:.3f}, y={pt['y']:.3f}")
        print("!" * 70)


if __name__ == "__main__":
    main()
