#%%
import glob
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.lines import Line2D


# Configuration
RESULTS_DIRS = {
    "pfvf": "results/mce",
    "psf": "results/mce", 
    "defend": "results/mce_defend",
    "dp_smoothing": "results/mce_dp_smoothing",
    "low_count_threshold": "results/mce_low_count_threshold", 
    "num_sub_iterations": "results/mce_num_sub_iterations",
    "final_style_prune": "results/mce_final_style_prune",
    "corpus_init": "results/mce",
    "bpe_ref": "results/bpe_ref",
}

# Default baselines for relative comparison
DEFAULTS = {
    "pfvf": 1.1,  # Default pre_final_vocab_factor
    "psf": 0.75,  # Default pruning_shrinking_factor
    "defend": False,  # Default defensive_prune
    "dp_smoothing": True,  # Default m_step_dp_smoothing
    "low_count_threshold": 0.0,  # Default m_step_low_count_threshold
    "num_sub_iterations": 2,  # Default num_sub_iterations
    "final_style_prune": False,  # Default final_style_prune (use Viterbi-based pruning)
    # Reference default for corpus_init comparisons
    "corpus_init": "corpus_repair",
}


def _load_parameter_results(param_type: str, results_dir: str) -> pd.DataFrame:
    """Load results for a specific parameter type."""
    files = glob.glob(f"{results_dir}/**/*.json", recursive=True)
    if not files:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
            
            # Filter by parameter type
            if param_type == "pfvf" and "pre_final_vocab_factor" in data:
                data["param_value"] = data["pre_final_vocab_factor"]
                data["param_type"] = "pfvf"
                rows.append(data)
            elif param_type == "psf" and "pruning_shrinking_factor" in data:
                data["param_value"] = data["pruning_shrinking_factor"]
                data["param_type"] = "psf"
                rows.append(data)
            elif param_type == "defend" and "defensive_prune" in data:
                data["param_value"] = data["defensive_prune"]
                data["param_type"] = "defend"
                rows.append(data)
            elif param_type == "dp_smoothing" and "m_step_dp_smoothing" in data:
                data["param_value"] = data["m_step_dp_smoothing"]
                data["param_type"] = "dp_smoothing"
                rows.append(data)
            elif param_type == "low_count_threshold" and "m_step_low_count_threshold" in data:
                data["param_value"] = data["m_step_low_count_threshold"]
                data["param_type"] = "low_count_threshold"
                rows.append(data)
            elif param_type == "num_sub_iterations" and "num_sub_iterations" in data:
                data["param_value"] = data["num_sub_iterations"]
                data["param_type"] = "num_sub_iterations"
                rows.append(data)
            elif param_type == "final_style_prune" and "final_style_prune" in data:
                data["param_value"] = data["final_style_prune"]
                data["param_type"] = "final_style_prune"
                rows.append(data)
            elif param_type == "corpus_init" and "algo" in data:
                # Include default + all corpus_* algos; keep specific repair modifiers separate for sanity checks
                algo = data["algo"]
                if algo.startswith("corpus_"):
                    # Allow: corpus_repair(_short/_few/_many), corpus_long, corpus_intermediate
                    allowed = {
                        "corpus_repair",
                        "corpus_repair_short",
                        "corpus_repair_few",
                        "corpus_repair_many",
                        "corpus_long",
                        "corpus_intermediate",
                    }
                    if algo in allowed:
                        data["param_value"] = algo
                        data["param_type"] = "corpus_init"
                        rows.append(data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={"corpus": "dataset", "algo": "algorithm"})
    
    # Filter to corpus_repair only for parameter sweeps that assume fixed algorithm
    if param_type not in {"corpus_init"}:
        df = df[df["algorithm"] == "corpus_repair"].copy()
    df = df[df['dataset'] != 'smol_eng_latn_300mb']
    return df


def _display_parameter_grid(
    df: pd.DataFrame,
    param_type: str,
    values_col: str,
    caption: str,
    *,
    higher_is_better: bool,
):
    """Display colored grid for a parameter."""
    if df.empty:
        return
    
    # For final_style_prune, use num_sub_iterations as columns (like num_sub_iterations does)
    if param_type == "final_style_prune":
        if "num_sub_iterations" in df.columns:
            pivot_df = df.pivot_table(index="dataset", columns="num_sub_iterations", values=values_col, aggfunc="mean")
        else:
            # If no num_sub_iterations, just return without displaying
            return
    else:
        pivot_df = df.pivot_table(index="dataset", columns="param_value", values=values_col, aggfunc="mean")
    
    # Sort columns appropriately
    try:
        if param_type in ["defend", "dp_smoothing"]:
            # Boolean: False, True
            pivot_df = pivot_df.reindex([False, True], axis=1)
        elif param_type == "corpus_init":
            # Custom order for algorithms
            desired = [
                "corpus_repair",
                "corpus_long",
                "corpus_intermediate",
                "corpus_repair_short",
                "corpus_repair_few",
                "corpus_repair_many",
            ]
            existing = [c for c in desired if c in pivot_df.columns]
            pivot_df = pivot_df.reindex(existing, axis=1)
        else:
            # Numeric: sort ascending
            pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    except Exception:
        pass

    cmap = "RdYlGn" if higher_is_better else "RdYlGn_r"
    styled = (
        pivot_df.style.background_gradient(cmap=cmap, axis=1)
        .format("{:.5f}", na_rep="-")
        .set_caption(caption)
    )
    display(styled)


def _compute_relative_performance(all_data: pd.DataFrame) -> pd.DataFrame:
    """Compute relative performance vs defaults for scatterplot."""
    relative_rows = []
    
    for dataset in all_data["dataset"].unique():
        dataset_data = all_data[all_data["dataset"] == dataset]
        
        # Find baseline (all defaults)
        baseline = dataset_data[
            ((dataset_data["param_type"] == "pfvf") & (dataset_data["param_value"] == DEFAULTS["pfvf"])) |
            ((dataset_data["param_type"] == "psf") & (dataset_data["param_value"] == DEFAULTS["psf"])) |
            ((dataset_data["param_type"] == "defend") & (dataset_data["param_value"] == DEFAULTS["defend"])) |
            ((dataset_data["param_type"] == "dp_smoothing") & (dataset_data["param_value"] == DEFAULTS["dp_smoothing"])) |
            ((dataset_data["param_type"] == "low_count_threshold") & (dataset_data["param_value"] == DEFAULTS["low_count_threshold"])) |
            ((dataset_data["param_type"] == "num_sub_iterations") & (dataset_data["param_value"] == DEFAULTS["num_sub_iterations"])) |
            ((dataset_data["param_type"] == "final_style_prune") & (dataset_data["param_value"] == DEFAULTS["final_style_prune"])) |
            ((dataset_data["param_type"] == "corpus_init") & (dataset_data["param_value"] == DEFAULTS["corpus_init"]))
        ]
        
        if baseline.empty:
            continue
            
        # Use first baseline found (should be similar across parameter types)
        baseline_obj = baseline["objective"].iloc[0]
        baseline_bpt = baseline["bytes_per_token"].iloc[0]
        
        # Compute relatives for all non-baseline points
        for _, row in dataset_data.iterrows():
            is_baseline = (
                (row["param_type"] == "pfvf" and row["param_value"] == DEFAULTS["pfvf"]) or
                (row["param_type"] == "psf" and row["param_value"] == DEFAULTS["psf"]) or
                (row["param_type"] == "defend" and row["param_value"] == DEFAULTS["defend"]) or
                (row["param_type"] == "dp_smoothing" and row["param_value"] == DEFAULTS["dp_smoothing"]) or
                (row["param_type"] == "low_count_threshold" and row["param_value"] == DEFAULTS["low_count_threshold"]) or
                (row["param_type"] == "num_sub_iterations" and row["param_value"] == DEFAULTS["num_sub_iterations"]) or
                (row["param_type"] == "final_style_prune" and row["param_value"] == DEFAULTS["final_style_prune"]) or
                (row["param_type"] == "corpus_init" and row["param_value"] == DEFAULTS["corpus_init"])
            )
            
            if is_baseline:
                continue
                
            rel_obj = (row["objective"] - baseline_obj) / baseline_obj
            rel_bpt = (row["bytes_per_token"] - baseline_bpt) / baseline_bpt
            
            relative_rows.append({
                "dataset": dataset,
                "param_type": row["param_type"],
                "param_value": row["param_value"],
                "rel_objective": rel_obj,
                "rel_bytes_per_token": rel_bpt,
            })
    
    return pd.DataFrame(relative_rows)


def _load_bpe_reference(results_dir: str) -> pd.DataFrame:
    """Load BPE reference results."""
    files = glob.glob(f"{results_dir}/**/*.json", recursive=True)
    if not files:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
            if "algo" in data and data["algo"] == "bpe":
                rows.append(data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={"corpus": "dataset", "algo": "algorithm"})
    return df


def main():
    # Load all parameter data
    all_data = []
    for param_type, results_dir in RESULTS_DIRS.items():
        if param_type == "bpe_ref":
            continue  # Handle separately
        param_df = _load_parameter_results(param_type, results_dir)
        if not param_df.empty:
            all_data.append(param_df)
    
    if not all_data:
        print("No parameter experiment data found")
        return
        
    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure we have visibility for corpus_init across datasets by forward-filling from base variants
    # This keeps one row per dataset×corpus_init value (repair/long/intermediate) if present
    if not combined_df.empty:
        # No-op other than ensuring types are consistent
        pass
    
    # Load BPE reference data
    bpe_df = _load_bpe_reference(RESULTS_DIRS["bpe_ref"])
    
    # Data summary
    display(HTML("<h2>Combined Parameter Analysis</h2>"))
    summary_data = []
    for param_type in ["pfvf", "psf", "defend", "dp_smoothing", "low_count_threshold", "num_sub_iterations", "final_style_prune", "corpus_init"]:
        param_data = combined_df[combined_df["param_type"] == param_type]
        if not param_data.empty:
            summary_data.append({
                "parameter": param_type,
                "runs": len(param_data),
                "datasets": param_data["dataset"].nunique(),
                "values": param_data["param_value"].nunique(),
                "default": DEFAULTS[param_type],
            })
    
    display(pd.DataFrame(summary_data))

    # Unified scatterplot relative to defaults
    display(HTML("<h2>Relative Performance vs Defaults</h2>"))
    rel_df = _compute_relative_performance(combined_df)
    
    if not rel_df.empty:
        plt.figure(figsize=(12, 8))
        
        # Color by parameter type
        param_colors = {
            "pfvf": "blue", 
            "psf": "red", 
            "defend": "green",
            "dp_smoothing": "orange",
            "low_count_threshold": "purple",
            "num_sub_iterations": "brown",
            "final_style_prune": "cyan",
            "corpus_init": "pink",
        }
        
        for param_type in ["pfvf", "psf", "defend", "dp_smoothing", "low_count_threshold", "num_sub_iterations", "final_style_prune", "corpus_init"]:
            subset = rel_df[rel_df["param_type"] == param_type]
            if not subset.empty:
                # Determine if each point is above or below reference
                for _, row in subset.iterrows():
                    # Marker logic
                    if param_type == "defend":
                        # Only show the non-baseline: defend=True
                        if row["param_value"] is not True:
                            continue
                        marker = "^"
                    elif param_type == "dp_smoothing":
                        # Only show the non-baseline: dp_smoothing=False
                        if row["param_value"] is not False:
                            continue
                        marker = "x"
                    elif param_type == "final_style_prune":
                        # Only show the non-baseline: final_style_prune=True
                        if row["param_value"] is not True:
                            continue
                        marker = "s"
                    elif param_type == "corpus_init":
                        # Show only: long and intermediate (exclude baseline and short)
                        if row["param_value"] not in ["corpus_long", "corpus_intermediate"]:
                            continue
                        marker = "^" if row["param_value"] == "corpus_long" else "o"
                    else:
                        above_ref = row["param_value"] > DEFAULTS[param_type]
                        marker = "^" if above_ref else "v"
                    
                    plt.scatter(
                        row["rel_objective"],
                        row["rel_bytes_per_token"],
                        c=param_colors[param_type],
                        marker=marker,
                        alpha=0.7,
                        s=80,
                    )
        
        # Create legend manually
        legend_elements = []
        for param_type in ["pfvf", "psf", "defend", "dp_smoothing", "low_count_threshold", "num_sub_iterations", "final_style_prune", "corpus_init"]:
            if not rel_df[rel_df["param_type"] == param_type].empty:
                if param_type == "corpus_init":
                    legend_elements.extend([
                        Line2D([0], [0], marker='^', color='w', markerfacecolor=param_colors[param_type], markersize=8, label="corpus_long"),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=param_colors[param_type], markersize=8, label="corpus_intermediate"),
                    ])
                elif param_type == "defend":
                    legend_elements.extend([
                        Line2D([0], [0], marker='^', color='w', markerfacecolor=param_colors[param_type], markersize=8, label="defend=True"),
                    ])
                elif param_type == "dp_smoothing":
                    legend_elements.extend([
                        Line2D([0], [0], marker='x', color='w', markerfacecolor=param_colors[param_type], markersize=8, label="dp_smoothing=False"),
                    ])
                elif param_type == "final_style_prune":
                    legend_elements.extend([
                        Line2D([0], [0], marker='s', color='w', markerfacecolor=param_colors[param_type], markersize=8, label="final_style_prune=True"),
                    ])
                else:
                    legend_elements.extend([
                        Line2D([0], [0], marker='^', color='w', markerfacecolor=param_colors[param_type], 
                               markersize=8, label=f"{param_type.upper()} above ref"),
                        Line2D([0], [0], marker='v', color='w', markerfacecolor=param_colors[param_type], 
                               markersize=8, label=f"{param_type.upper()} below ref"),
                    ])
        
        # Add BPE reference markers at objective=0
        if not bpe_df.empty and "bytes_per_token" in bpe_df.columns:
            for _, bpe_row in bpe_df.iterrows():
                dataset = bpe_row["dataset"]
                
                # Find baseline for this dataset
                dataset_baseline = combined_df[
                    (combined_df["dataset"] == dataset) &
                    (
                        ((combined_df["param_type"] == "pfvf") & (combined_df["param_value"] == DEFAULTS["pfvf"])) |
                        ((combined_df["param_type"] == "psf") & (combined_df["param_value"] == DEFAULTS["psf"])) |
                        ((combined_df["param_type"] == "defend") & (combined_df["param_value"] == DEFAULTS["defend"])) |
                        ((combined_df["param_type"] == "dp_smoothing") & (combined_df["param_value"] == DEFAULTS["dp_smoothing"])) |
                        ((combined_df["param_type"] == "low_count_threshold") & (combined_df["param_value"] == DEFAULTS["low_count_threshold"])) |
                        ((combined_df["param_type"] == "num_sub_iterations") & (combined_df["param_value"] == DEFAULTS["num_sub_iterations"])) |
                        ((combined_df["param_type"] == "final_style_prune") & (combined_df["param_value"] == DEFAULTS["final_style_prune"])) |
                        ((combined_df["param_type"] == "corpus_init") & (combined_df["param_value"] == DEFAULTS["corpus_init"]))
                    )
                ]
                
                if not dataset_baseline.empty:
                    baseline_bpt = dataset_baseline["bytes_per_token"].iloc[0]
                    bpe_rel_bpt = (bpe_row["bytes_per_token"] - baseline_bpt) / baseline_bpt
                    
                    # Add marker at objective=0 (baseline) showing BPE compression
                    plt.scatter(0, bpe_rel_bpt, c='black', marker='x', s=100, alpha=0.8)
            
            # Add BPE to legend
            legend_elements.append(
                Line2D([0], [0], marker='x', color='w', markerfacecolor='black', 
                       markersize=8, label="BPE reference")
            )
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='best')
        
        # Reference lines
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel("Relative objective change vs defaults (%) (lower is better)")
        plt.ylabel("Relative bytes_per_token change vs defaults (%) (higher is better)")
        
        # Format axes as percentages
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.1f}%'))
        plt.title(f"Parameter Effects vs Defaults (PFVF={DEFAULTS['pfvf']}, PSF={DEFAULTS['psf']}, Defend={DEFAULTS['defend']})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Compression summary grid: dataset x method
    display(HTML("<h2>Compression Summary (bytes_per_token)</h2>"))
    
    compression_data = []
    
    # Add BPE reference data
    if not bpe_df.empty:
        for _, bpe_row in bpe_df.iterrows():
            compression_data.append({
                "dataset": bpe_row["dataset"],
                "method": "BPE",
                "bytes_per_token": bpe_row["bytes_per_token"],
            })
    
    # Add unigram baseline data (defaults)
    baseline_data = combined_df[
        ((combined_df["param_type"] == "pfvf") & (combined_df["param_value"] == DEFAULTS["pfvf"])) |
        ((combined_df["param_type"] == "psf") & (combined_df["param_value"] == DEFAULTS["psf"])) |
        ((combined_df["param_type"] == "defend") & (combined_df["param_value"] == DEFAULTS["defend"])) |
        ((combined_df["param_type"] == "dp_smoothing") & (combined_df["param_value"] == DEFAULTS["dp_smoothing"])) |
        ((combined_df["param_type"] == "low_count_threshold") & (combined_df["param_value"] == DEFAULTS["low_count_threshold"])) |
        ((combined_df["param_type"] == "num_sub_iterations") & (combined_df["param_value"] == DEFAULTS["num_sub_iterations"])) |
        ((combined_df["param_type"] == "final_style_prune") & (combined_df["param_value"] == DEFAULTS["final_style_prune"])) |
        ((combined_df["param_type"] == "corpus_init") & (combined_df["param_value"] == DEFAULTS["corpus_init"]))
    ]
    
    for _, row in baseline_data.iterrows():
        compression_data.append({
            "dataset": row["dataset"],
            "method": "Unigram (default)",
            "bytes_per_token": row["bytes_per_token"],
        })
    
    # Add specific parameter values
    # PFVF=1.0
    pfvf_1_data = combined_df[
        (combined_df["param_type"] == "pfvf") & (combined_df["param_value"] == 1.0)
    ]
    for _, row in pfvf_1_data.iterrows():
        compression_data.append({
            "dataset": row["dataset"],
            "method": "Unigram (PFVF=1.0)",
            "bytes_per_token": row["bytes_per_token"],
        })
    
    # PSF=0.9
    psf_09_data = combined_df[
        (combined_df["param_type"] == "psf") & (combined_df["param_value"] == 0.9)
    ]
    for _, row in psf_09_data.iterrows():
        compression_data.append({
            "dataset": row["dataset"],
            "method": "Unigram (PSF=0.9)",
            "bytes_per_token": row["bytes_per_token"],
        })
    
    # Add corpus_init variants as methods
    corpus_init_df = combined_df[combined_df["param_type"] == "corpus_init"]
    for _, row in corpus_init_df.iterrows():
        label = f"Unigram ({row['param_value']})"
        compression_data.append({
            "dataset": row["dataset"],
            "method": label,
            "bytes_per_token": row.get("bytes_per_token", 0.0),
        })

    if compression_data:
        compression_df = pd.DataFrame(compression_data)
        
        # Create pivot table: dataset x method
        compression_pivot = compression_df.pivot_table(
            index="dataset", 
            columns="method", 
            values="bytes_per_token", 
            aggfunc="mean"
        )
        
        # Sort columns: BPE, Unigram (default), specific parameters (including corpus init variants)
        column_order = []
        if "BPE" in compression_pivot.columns:
            column_order.append("BPE")
        if "Unigram (default)" in compression_pivot.columns:
            column_order.append("Unigram (default)")
        if "Unigram (PFVF=1.0)" in compression_pivot.columns:
            column_order.append("Unigram (PFVF=1.0)")
        if "Unigram (PSF=0.9)" in compression_pivot.columns:
            column_order.append("Unigram (PSF=0.9)")
        # Add corpus_init variants if present (as methods for comparison)
        corpus_cols = [
            ("Unigram (corpus_repair)", "corpus_repair"),
            ("Unigram (corpus_long)", "corpus_long"),
            ("Unigram (corpus_intermediate)", "corpus_intermediate"),
            ("Unigram (corpus_repair_short)", "corpus_repair_short"),
            ("Unigram (corpus_repair_few)", "corpus_repair_few"),
            ("Unigram (corpus_repair_many)", "corpus_repair_many"),
        ]
        # If these columns exist in the pivot (due to earlier labelling), include them in order
        for col_label, _ in corpus_cols:
            if col_label in compression_pivot.columns and col_label not in column_order:
                column_order.append(col_label)
        
        compression_pivot = compression_pivot.reindex(column_order, axis=1)
        
        styled_compression = (
            compression_pivot.style
            .format("{:.4f}", na_rep="-")
            .background_gradient(cmap="RdYlGn", axis=1)  # Row-wise comparison
            .set_caption("Compression performance by dataset and method (higher is better)")
        )
        display(styled_compression)

    # Parameter grids
    display(HTML("<h2>Parameter Effect Grids</h2>"))
    
    for param_type in ["pfvf", "psf", "defend", "dp_smoothing", "low_count_threshold", "num_sub_iterations", "final_style_prune", "corpus_init"]:
        param_data = combined_df[combined_df["param_type"] == param_type]
        if param_data.empty:
            continue
            
        display(HTML(f"<h3>{param_type.upper()} Effects</h3>"))
        
        # Objective grid (lower is better)
        if "objective" in param_data.columns:
            x_axis_label = "num_sub_iterations" if param_type == "final_style_prune" else param_type
            _display_parameter_grid(
                param_data,
                param_type,
                "objective",
                f"Objective by dataset × {x_axis_label} (lower is better)",
                higher_is_better=False,
            )
        
        # Compression grid (higher is better)
        if "bytes_per_token" in param_data.columns:
            x_axis_label = "num_sub_iterations" if param_type == "final_style_prune" else param_type
            _display_parameter_grid(
                param_data,
                param_type,
                "bytes_per_token",
                f"Bytes per token by dataset × {x_axis_label} (higher is better)",
                higher_is_better=True,
            )


if __name__ == "__main__":
    main()

# %%
