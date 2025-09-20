# %%
import glob
import json
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from IPython.display import display, HTML
from pandas.io.formats.style import Styler


# Configuration
RESULTS_DIR = "results/mce"
METRIC = "objective"  # Metric to analyze, e.g., "objective", "bytes_per_token"

# Color schemes
ALGORITHM_COLORS = {
    "corpus_repair": "#1f77b4",
    "corpus_repair_many": "#ff7f0e",
    "corpus_repair_short": "#2ca02c",
    "corpus_repair_few": "#d62728",
    "simple": "#9467bd",
    "simple_many": "#8c564b",
    "simple_short": "#e377c2",
    "simple_few": "#7f7f7f",
}


def _load_compression_results(results_dir: str) -> pd.DataFrame:
    """Load monolingual compression stats and transform into a long-form DataFrame."""
    files = glob.glob(f"{results_dir}/**/*.json", recursive=True)
    if not files:
        raise FileNotFoundError(f"No JSON files found in {results_dir}")

    rows = []
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
            rows.append(data)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No data found in results directory")
    df = df.rename(columns={"corpus": "dataset", "algo": "algorithm"})

    def _extract_features(algo_name):
        features = {"modifier": "default", "base_algo": algo_name}
        if algo_name.endswith("_many"):
            features["modifier"] = "many"
            features["base_algo"] = algo_name.removesuffix("_many")
        elif algo_name.endswith("_few"):
            features["modifier"] = "few"
            features["base_algo"] = algo_name.removesuffix("_few")
        elif algo_name.endswith("_short"):
            features["modifier"] = "short"
            features["base_algo"] = algo_name.removesuffix("_short")

        if features["base_algo"].startswith("corpus_"):
            features["family"] = "corpus"
            features["subtype"] = features["base_algo"].split("_", 1)[1]
        else:
            features["family"] = "simple"
            features["subtype"] = "simple"
        return pd.Series(features)

    algo_features = df["algorithm"].apply(_extract_features)
    df = pd.concat([df, algo_features], axis=1)
    return df


def _fit_glm(df: pd.DataFrame) -> tuple[Any, str]:
    """Fit GLM for the chosen metric."""
    df_glm = df.copy()
    for col in ["family", "subtype", "modifier", "dataset"]:
        df_glm[col] = df_glm[col].astype("category")

    df_glm = df_glm.dropna(subset=[METRIC, "chars"])
    if df_glm.empty:
        raise ValueError(f"No valid data to fit model after dropping NaNs in '{METRIC}' or 'chars'")

    result = smf.glm(
        f"{METRIC} ~ C(family, Sum) + C(subtype, Sum) + C(modifier, Sum) + C(dataset, Sum)",
        data=df_glm,
        family=sm.families.Gaussian(),
        freq_weights=df_glm["chars"],
    ).fit()

    result._jam_levels = {
        col: df_glm[col].cat.categories.tolist() for col in ["family", "subtype", "modifier", "dataset"]
    }
    return result, METRIC


def _extract_effects(result, var: str) -> tuple[list[str], np.ndarray]:
    """Extract coefficients including reconstructed omitted level for sum coding."""
    mask = result.params.index.str.contains(rf"^C\({re.escape(var)}(?:, Sum)?\)\[(?:T|S)\.")
    sub = result.params[mask]
    if sub.empty:
        return [], np.array([])

    labels = [re.search(r"\[(?:T|S)\.(.*?)\]", idx).group(1) for idx in sub.index]
    values = sub.values

    # Reconstruct omitted level for sum coding
    if hasattr(result, "_jam_levels") and var in result._jam_levels:
        all_levels = result._jam_levels[var]
        missing = [lvl for lvl in all_levels if lvl not in labels]
        if len(missing) == 1:
            labels.append(missing[0])
            values = np.append(values, -values.sum())

    return labels, values


def _plot_bars(labels: list[str], values: np.ndarray, title: str, colors: list[str] | None = None):
    """Create horizontal bar plot with optional colors."""
    if not len(values):
        return

    order = np.argsort(values)
    plt.figure(figsize=(10, max(3, len(values) * 0.25)))
    plt.barh(
        [labels[i] for i in order], values[order], color=[colors[i] for i in order] if colors else None
    )
    plt.xlabel("Coefficient")
    plt.title(title)
    plt.tight_layout()


def _show_plots(result, metric_label: str):
    """Generate all coefficient plots."""
    family_labels, family_values = _extract_effects(result, "family")
    subtype_labels, subtype_values = _extract_effects(result, "subtype")
    modifier_labels, modifier_values = _extract_effects(result, "modifier")
    dataset_labels, dataset_values = _extract_effects(result, "dataset")

    if len(family_values):
        _plot_bars(family_labels, family_values, f"Family effects on {metric_label}")
        plt.show()
    if len(subtype_values):
        _plot_bars(subtype_labels, subtype_values, f"Subtype effects on {metric_label}")
        plt.show()
    if len(modifier_values):
        _plot_bars(modifier_labels, modifier_values, f"Modifier effects on {metric_label}")
        plt.show()

    # Dataset plot
    if len(dataset_values):
        _plot_bars(dataset_labels, dataset_values, f"Dataset effects on {metric_label}")
        plt.show()


def _make_table(result, var: str, label_name: str) -> pd.DataFrame:
    """Create coefficient table with stats."""
    mask = result.params.index.str.contains(rf"^C\({re.escape(var)}(?:, Sum)?\)\[(?:T|S)\.")
    sub = result.params[mask]
    if sub.empty:
        return pd.DataFrame(columns=[label_name, "coef", "std_err", "z", "p", "ci_low", "ci_high"])

    labels = [re.search(r"\[(?:T|S)\.(.*?)\]", idx).group(1) for idx in sub.index]
    ci = result.conf_int()

    data = [
        {
            label_name: label,
            "coef": result.params[idx],
            "std_err": result.bse[idx],
            "z": result.tvalues[idx],
            "p": result.pvalues[idx],
            "ci_low": ci.loc[idx, 0],
            "ci_high": ci.loc[idx, 1],
        }
        for label, idx in zip(labels, sub.index)
    ]

    # Add reconstructed level with NaN stats
    if hasattr(result, "_jam_levels") and var in result._jam_levels:
        all_levels = result._jam_levels[var]
        missing = [lvl for lvl in all_levels if lvl not in labels]
        if len(missing) == 1:
            data.append(
                {
                    label_name: missing[0],
                    "coef": -sub.sum(),
                    **{k: float("nan") for k in ["std_err", "z", "p", "ci_low", "ci_high"]},
                }
            )

    return pd.DataFrame(data).sort_values("coef", ascending=False).reset_index(drop=True)


def _style_df(df: pd.DataFrame) -> Styler:
    """Apply gradient styling to numeric columns."""
    if df.empty:
        return pd.DataFrame({"info": ["No data"]}).style
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not len(numeric_cols):
        return df.style
    return df.style.format({col: "{:.3f}" for col in numeric_cols}).background_gradient(
        cmap="RdYlGn_r", axis=0, subset=list(numeric_cols)
    )


def main():
    """Main analysis function."""
    # Load and process data
    try:
        df = _load_compression_results(RESULTS_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")
        return

    # Calculate key metric
    df_metric = df.dropna(subset=[METRIC, "chars"])
    weighted_metric = (df_metric[METRIC] * df_metric["chars"]).sum()
    total_chars = df_metric["chars"].sum()
    key_metric = weighted_metric / total_chars if total_chars > 0 else float("nan")
    key_label = f"{METRIC} (weighted by chars)"

    # Display results
    display(HTML("<h2>Key Metric</h2>"))
    display(pd.DataFrame({key_label: [key_metric]}))

    display(HTML("<h2>Data Summary</h2>"))
    display(
        pd.DataFrame(
            {
                "runs": [len(df)],
                "datasets": [df.dataset.nunique()],
                "algorithms": [df.algorithm.nunique()],
                "families": [df.family.nunique()],
                "subtypes": [df.subtype.nunique()],
                "modifiers": [df.modifier.nunique()],
            }
        )
    )

    # Fit model and show results
    try:
        result, metric_label = _fit_glm(df)
        display(HTML("<h2>Model Summary</h2>"))
        print(result.summary())

        # Generate tables
        for name, var, col in [
            ("Family", "family", "family"),
            ("Subtype", "subtype", "subtype"),
            ("Modifier", "modifier", "modifier"),
            ("Dataset", "dataset", "dataset"),
        ]:
            display(HTML(f"<h2>{name} Coefficients</h2>"))
            display(_style_df(_make_table(result, var, col)))

        # Show plots
        _show_plots(result, metric_label)
    except (ValueError, Exception) as e:
        print(f"Could not fit model or generate plots: {e}")


if __name__ == "__main__":
    main()
