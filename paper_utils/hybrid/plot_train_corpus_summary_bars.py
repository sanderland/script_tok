#!/usr/bin/env python3
"""Generate compact bar summaries for FineWeb-trained evaluation robustness.

The appendix table keeps all language-by-setup cells. These candidates collapse
one axis at a time, always using compression improvement over Unigram
(positive is better). Training is FineWeb-only; robustness varies evaluation
between in-domain FineWeb and held-out Goldfish corpora.

* mean by train/eval setup, averaging over languages
* mean by language, averaging over training setups
* mean by method, averaging over all setup-language cells
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from paper_utils.hybrid.utils import paper_figure_path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
GRID_JSON = RESULTS_DIR / "compression_train_eval_grid.json"

OUT_BY_SETUP = paper_figure_path("train_corpus_summary_by_setup.png", extra=True)
OUT_BY_LANGUAGE = paper_figure_path("train_corpus_summary_by_language.png", extra=True)
OUT_BY_METHOD = paper_figure_path("train_corpus_summary_by_method.png", extra=True)
OUT_BY_LANGUAGE_HATCH = paper_figure_path("train_corpus_summary_by_language_hatch_print_balanced.png", extra=True)

PAIRINGS = [
    ("fineweb->fineweb", "1-lang Web\nEval Web"),
    ("fineweb->fishfood", "1-lang Web\nEval Goldfish"),
    ("fineweb_h6->fineweb", "6-lang Web\nEval Web"),
    ("fineweb_h6->fishfood", "6-lang Web\nEval Goldfish"),
]

LANG_ORDER = ["eng", "deu", "fin", "rus", "arb", "kor"]
LANG_LABEL = {
    "eng": "English",
    "deu": "German",
    "fin": "Finnish",
    "rus": "Russian",
    "arb": "Arabic",
    "kor": "Korean",
}

METHOD_ORDER = ["bpe_init", "fsp", "bpe_init_fsp", "mingram", "mingram_pp", "bpe", "pathpiece_bpe", "convextok"]
METHOD_LABEL = {
    "bpe": "BPE",
    "fsp": "FSP",
    "bpe_init": "Unigram-BPE-Init",
    "bpe_init_fsp": "FSP-BPE-Init",
    "mingram": "MinGram",
    "mingram_pp": "MinGram-PP",
    "pathpiece_bpe": "PathPiece-BPE",
    "convextok": "ConvexTok",
}
METHOD_TICK_LABEL = {
    "bpe": "BPE",
    "fsp": "FSP",
    "bpe_init": "Unigram-\nBPE-Init",
    "bpe_init_fsp": "FSP-\nBPE-Init",
    "mingram": "MinGram",
    "mingram_pp": "MinGram-\nPP",
    "pathpiece_bpe": "PathPiece-\nBPE",
    "convextok": "ConvexTok",
}
METHOD_STYLE = {
    "bpe": {"facecolor": "#D55E00", "edgecolor": "#C15000", "hatch": None, "alpha": 0.88},
    "bpe_init": {"facecolor": "#6F6F6F", "edgecolor": "#555555", "hatch": None, "alpha": 0.84},
    "fsp": {"facecolor": "#7E57C2", "edgecolor": "#6C46B3", "hatch": None, "alpha": 0.84},
    "bpe_init_fsp": {"facecolor": "#B18BE8", "edgecolor": "#7E57C2", "hatch": None, "alpha": 0.9},
    "mingram": {"facecolor": "#0072B2", "edgecolor": "#00649D", "hatch": None, "alpha": 0.92},
    "mingram_pp": {"facecolor": "#009E73", "edgecolor": "#007F5F", "hatch": None, "alpha": 0.9},
    "pathpiece_bpe": {"facecolor": "#CC79A7", "edgecolor": "#AA5F8C", "hatch": None, "alpha": 0.9},
    "convextok": {"facecolor": "#009E73", "edgecolor": "#007F5F", "hatch": None, "alpha": 0.9},
}
BALANCED_HATCH = {
    "bpe": "//",
    "bpe_init": "..",
    "fsp": "\\\\",
    "bpe_init_fsp": "xx",
    "mingram": "--",
    "mingram_pp": "oo",
    "pathpiece_bpe": "++",
    "convextok": "**",
}


def _improvement(value: float) -> float:
    return -float(value)


def _cell(grid: dict, pair_key: str, lang: str, method: str) -> float | None:
    value = grid[pair_key]["series"][lang][method]
    return None if value is None else _improvement(value)


def _mean_present(values: list[float | None]) -> float:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else float("nan")


def _available_pairings(grid: dict) -> list[tuple[str, str]]:
    return [
        (pair_key, label)
        for pair_key, label in PAIRINGS
        if any(lang in grid.get(pair_key, {}).get("series", {}) for lang in LANG_ORDER)
    ]


def _style_axis(ax) -> None:
    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.55, zorder=1)
    ax.grid(True, axis="y", color="#E2E2E2", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)


def _style_for(
    method: str,
    hatches: dict[str, str] | None = None,
) -> dict:
    style = METHOD_STYLE[method].copy()
    if hatches is not None:
        style["hatch"] = hatches[method]
    return style


def _legend_handles(
    hatches: dict[str, str] | None = None,
) -> list[Patch]:
    return [
        Patch(
            facecolor=_style_for(method, hatches)["facecolor"],
            edgecolor=_style_for(method, hatches)["edgecolor"],
            hatch=_style_for(method, hatches)["hatch"],
            alpha=_style_for(method, hatches)["alpha"],
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]


def _grouped_bars(
    ax,
    x: np.ndarray,
    values: dict[str, list[float]],
    width: float,
    hatches: dict[str, str] | None = None,
) -> None:
    offsets = np.linspace(
        -(len(METHOD_ORDER) - 1) / 2 * width,
        (len(METHOD_ORDER) - 1) / 2 * width,
        len(METHOD_ORDER),
    )
    for method_idx, method in enumerate(METHOD_ORDER):
        style = _style_for(method, hatches)
        ax.bar(
            x + offsets[method_idx],
            values[method],
            width=width,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            alpha=style["alpha"],
            linewidth=0.8,
            zorder=3,
        )


def plot_by_setup(grid: dict) -> None:
    pairings = _available_pairings(grid)
    x = np.arange(len(pairings))
    values = {
        method: [
            _mean_present([_cell(grid, pair_key, lang, method) for lang in LANG_ORDER])
            for pair_key, _ in pairings
        ]
        for method in METHOD_ORDER
    }

    fig, ax = plt.subplots(figsize=(6.6, 2.7), dpi=220)
    _grouped_bars(ax, x, values, width=0.105)
    _style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in pairings])
    ax.set_ylabel("Compression improvement (%)", fontsize=9)
    ax.set_xlabel("FineWeb-trained tokenizers; evaluation corpus varies", fontsize=8.5)
    ax.legend(
        handles=_legend_handles(),
        ncol=len(METHOD_ORDER),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=6.4,
    )
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_BY_SETUP, bbox_inches="tight")
    plt.close(fig)


def plot_by_language(
    grid: dict,
    out_path: Path = OUT_BY_LANGUAGE,
    hatches: dict[str, str] | None = None,
    hatch_linewidth: float = 0.45,
) -> None:
    pairings = _available_pairings(grid)
    x = np.arange(len(LANG_ORDER))
    values = {
        method: [
            _mean_present([_cell(grid, pair_key, lang, method) for pair_key, _ in pairings])
            for lang in LANG_ORDER
        ]
        for method in METHOD_ORDER
    }

    with plt.rc_context({"hatch.linewidth": hatch_linewidth}):
        fig, ax = plt.subplots(figsize=(7.2, 2.85), dpi=220)
        _grouped_bars(ax, x, values, width=0.105, hatches=hatches)
        _style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABEL[lang] for lang in LANG_ORDER], rotation=20, ha="right")
        ax.set_ylabel("Compression improvement (%)", fontsize=9)
        ax.set_xlabel("Mean over available FineWeb train/eval robustness settings", fontsize=8.5)
        ax.legend(
            handles=_legend_handles(hatches),
            ncol=len(METHOD_ORDER),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
            fontsize=6.4,
        )
        fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.88))
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def plot_by_method(grid: dict) -> None:
    pairings = _available_pairings(grid)
    methods = METHOD_ORDER
    means = []
    mins = []
    maxs = []
    for method in methods:
        vals = [
            _cell(grid, pair_key, lang, method)
            for pair_key, _ in pairings
            for lang in LANG_ORDER
        ]
        present = [value for value in vals if value is not None]
        means.append(float(np.mean(present)) if present else float("nan"))
        mins.append(float(np.min(present)) if present else float("nan"))
        maxs.append(float(np.max(present)) if present else float("nan"))

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(5.1, 2.85), dpi=220)
    for idx, method in enumerate(methods):
        style = METHOD_STYLE[method]
        ax.bar(
            x[idx],
            means[idx],
            width=0.64,
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            alpha=style["alpha"],
            linewidth=0.85,
            zorder=3,
        )
        ax.vlines(x[idx], mins[idx], maxs[idx], color="#444444", linewidth=1.0, alpha=0.55, zorder=4)
        ax.scatter([x[idx]], [means[idx]], s=16, color="#222222", zorder=5)

    _style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_TICK_LABEL[method] for method in methods])
    ax.set_ylabel("Compression improvement (%)", fontsize=9)
    ax.set_xlabel("Mean over available FineWeb train/eval cells; whisker shows min--max", fontsize=8.5)
    fig.tight_layout(pad=0.8)
    fig.savefig(OUT_BY_METHOD, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    grid = json.loads(GRID_JSON.read_text())
    plot_by_setup(grid)
    plot_by_language(grid)
    plot_by_language(grid, OUT_BY_LANGUAGE_HATCH, BALANCED_HATCH)
    plot_by_method(grid)
    for path in [OUT_BY_SETUP, OUT_BY_LANGUAGE, OUT_BY_LANGUAGE_HATCH, OUT_BY_METHOD]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
