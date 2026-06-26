#!/usr/bin/env python3
"""Generate the FineWeb-trained evaluation-domain figure (Appendix F3).

For each language, plot a heatmap showing compression improvement over
Default under each train/eval condition. Training is FineWeb-only; robustness
varies evaluation between in-domain FineWeb, held-out Goldfish corpora, and
FLORES+.

Reads `results/hybrid/compression_train_eval_grid.json` and writes
the paper PDF under `results/mingram_paper/figures/` plus a PNG under
`results/mingram_paper/extra/`.

Layout: 3 rows × 2 cols = 6 language panels, each showing FineWeb-trained
evaluation settings as rows and methods as columns. Colour and cell labels show
compression improvement over Unigram.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from paper_utils.hybrid.utils import paper_figure_path

plt.rcParams["hatch.linewidth"] = 0.45

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
GRID_JSON = RESULTS_DIR / "compression_train_eval_grid.json"
OUT_PNG = paper_figure_path("domain_shift.png", appendix=True, extra=True)
OUT_PDF = paper_figure_path("domain_shift.pdf", appendix=True, extra=True)

PAIRINGS = [
    ("fineweb->fineweb", "1-lang FW\nEval Web"),
    ("fineweb->fishfood", "1-lang FW\nEval Goldfish"),
    ("fineweb->flores_plus", "1-lang FW\nEval FLORES+"),
    ("fineweb_h6->fineweb", "6-lang FW\nEval Web"),
    ("fineweb_h6->fishfood", "6-lang FW\nEval Goldfish"),
    ("fineweb_h6->flores_plus", "6-lang FW\nEval FLORES+"),
]

METHOD_ORDER = ["bpe", "bpe_init", "fsp", "bpe_init_fsp", "mingram", "mingram_mi", "pathpiece_bpe", "convextok"]
METHOD_LABEL = {
    "bpe": "BPE",
    "bpe_init": "U-BPE",
    "fsp": "FSP",
    "bpe_init_fsp": "FSP-BPE-Init",
    "mingram": "MinGram",
    "mingram_mi": "MinGram-MI",
    "pathpiece_bpe": "PathPiece-BPE",
    "convextok": "ConvexTok",
}

LANG_ORDER = ["eng", "deu", "fin", "rus", "arb", "kor"]
LANG_LABEL = {
    "eng": "English",
    "deu": "German",
    "fin": "Finnish",
    "rus": "Russian",
    "arb": "Arabic",
    "kor": "Korean",
}


def collect_values(grid: dict, lang: str) -> dict[str, list[float | None]]:
    """Return {method: [value per pairing]} for a given language."""
    pairings = _available_pairings(grid)
    out: dict[str, list[float | None]] = {method: [] for method in METHOD_ORDER}
    for pair_key, _ in pairings:
        row = grid.get(pair_key, {}).get("series", {}).get(lang)
        for method in METHOD_ORDER:
            value = None if row is None else row.get(method)
            out[method].append(None if value is None else -value)
    return out


def collect_matrix(grid: dict, lang: str) -> np.ndarray:
    """Return rows=train/eval conditions, columns=methods for one language."""
    by_method = collect_values(grid, lang)
    matrix = np.empty((len(_available_pairings(grid)), len(METHOD_ORDER)), dtype=float)
    matrix.fill(np.nan)
    for method_idx, method in enumerate(METHOD_ORDER):
        for pairing_idx, value in enumerate(by_method[method]):
            if value is not None:
                matrix[pairing_idx, method_idx] = float(value)
    return matrix


def _available_pairings(grid: dict) -> list[tuple[str, str]]:
    return [
        (pair_key, label)
        for pair_key, label in PAIRINGS
        if any(lang in grid.get(pair_key, {}).get("series", {}) for lang in LANG_ORDER)
    ]


def plot(grid: dict, out_png: Path, out_pdf: Path) -> None:
    pairings = _available_pairings(grid)
    all_values = [
        value
        for lang in LANG_ORDER
        for value in collect_matrix(grid, lang).ravel()
        if not np.isnan(value)
    ]
    abs_max = max(abs(min(all_values)), abs(max(all_values)))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    cmap = plt.colormaps["RdYlGn"].copy()
    cmap.set_bad("#F2F2F2")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(9.2, 10.2),
        dpi=180,
        sharey=True,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.88,
        bottom=0.08,
        top=0.96,
        wspace=0.12,
        hspace=0.36,
    )

    for ax, lang in zip(axes.flat, LANG_ORDER, strict=True):
        matrix = collect_matrix(grid, lang)
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(LANG_LABEL[lang], fontsize=13, pad=6)
        ax.set_xticks(np.arange(len(METHOD_ORDER)))
        ax.set_xticklabels([METHOD_LABEL[method] for method in METHOD_ORDER], rotation=35, ha="right", fontsize=8.4)
        ax.set_yticks(np.arange(len(pairings)))
        ax.set_yticklabels([label for _, label in pairings], fontsize=8.4)
        ax.tick_params(axis="both", length=0)

        ax.set_xticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pairings), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    ax.text(col_idx, row_idx, "NA", ha="center", va="center", fontsize=7.0, color="#666666")
                else:
                    ax.text(col_idx, row_idx, f"{value:+.1f}", ha="center", va="center", fontsize=7.0, color="#222222")

    cbar_ax = fig.add_axes((0.91, 0.18, 0.018, 0.64))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Compression improvement over Unigram (%)", fontsize=9)
    fig.supxlabel("Method (U-BPE = Unigram-BPE-Init)", fontsize=11)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    grid = json.loads(GRID_JSON.read_text())
    plot(grid, OUT_PNG, OUT_PDF)


if __name__ == "__main__":
    main()
