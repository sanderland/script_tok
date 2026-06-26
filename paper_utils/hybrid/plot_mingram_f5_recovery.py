"""MinGram-score compression vs f, with the f=5 pruning-schedule variants.

The default MinGram (one-shot score prune, em=2) peaks at f~=1.15 and drops
to -1.05% by f=5. This overlays the f=5 ablation variants (varying EM
iterations and pruning_shrinking_factor) as separate points at f=5 to show
that *gradual* pruning -- not more EM -- recovers the high-f drop.
"""

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from paper_utils.hybrid.utils import paper_figure_path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
GRID_CACHE = RESULTS_DIR / "cache_train_eval_compression_grid.json"
ABL_DIR = REPO_ROOT / "results/mingram/ablations"
OUT_PNG = paper_figure_path("mingram_f5_recovery.png", extra=True)
UNIGRAM_BASELINE_ENG = 69_729_749


def _default_ftrace(cache: dict) -> list[tuple[float, float]]:
    out = []
    for k, v in cache.items():
        if "mingram" not in k or "fineweb_en_5gb/eng_latn_fishfood" not in k:
            continue
        m = re.search(r"mingram_f([\d.]+)_em(\d+)_p([\d.]+)", k)
        if not m:
            continue
        f, em, p = float(m.group(1)), int(m.group(2)), float(m.group(3))
        if em != 2 or p != 0.0:
            continue
        out.append((f, (int(v) - UNIGRAM_BASELINE_ENG) / UNIGRAM_BASELINE_ENG * 100))
    return sorted(out)


def main() -> None:
    cache = json.loads(GRID_CACHE.read_text())
    ftrace = _default_ftrace(cache)

    # Pruning-schedule variants (any f) from ablation summaries.
    variants = []  # (f, em, psf, delta)
    for sp in sorted(glob.glob(str(ABL_DIR / "mgf*_em*_psf*.summary.json"))):
        d = json.loads(Path(sp).read_text())
        variants.append((float(d["overshoot_factor"]), d["num_em_iterations"],
                         d["pruning_shrinking_factor"], d["delta_vs_unigram_pct"]))

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)

    xs, ys = zip(*ftrace)
    ax.plot(xs, ys, "-o", color="#1f77b4", lw=1.8, ms=6, zorder=3,
            label="MinGram default (em=2, one-shot prune)")

    # Group variants by f; jitter same-f points horizontally so labels don't overlap.
    from collections import defaultdict
    by_f: dict[float, list] = defaultdict(list)
    for f, em, psf, delta in variants:
        by_f[f].append((em, psf, delta))
    for f, pts in by_f.items():
        jit = 0.0
        for em, psf, delta in sorted(pts, key=lambda v: (v[1], v[0])):
            x = f * (1.0 + jit)
            color = "#ff7f0e" if psf == 0.0 else "#2ca02c"
            ax.scatter([x], [delta], s=60, color=color, zorder=5, edgecolors="white", linewidths=0.8)
            ax.annotate(f"em{em},psf{psf:g}", xy=(x, delta), xytext=(6, 0),
                        textcoords="offset points", fontsize=6.5, va="center")
            jit += 0.05

    ax.axhline(-1.88, color="#888", lw=0.7, ls="--", alpha=0.6)
    ax.text(1.0, -1.88, " f=1.15 peak (-1.88%)", fontsize=8, color="#666", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks([1.0, 1.25, 1.5, 2, 3, 5])
    ax.set_xticklabels(["1.0", "1.25", "1.5", "2", "3", "5"])
    ax.set_xlabel("Overshoot factor $f$")
    ax.set_ylabel("Compression $\\Delta$ vs Unigram (\\%, English)")
    ax.set_title("MinGram f=5 recovery: gradual pruning (green) helps, more EM (orange) doesn't")
    ax.invert_yaxis()
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", marker="o", lw=1.8, label="MinGram default f-trace"),
        plt.Line2D([0], [0], color="#ff7f0e", marker="o", lw=0, label="f=5, more EM (psf=0)"),
        plt.Line2D([0], [0], color="#2ca02c", marker="o", lw=0, label="f=5, gradual prune (psf>0)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    print("variants (f, em, psf, delta):")
    for f, em, psf, delta in sorted(variants):
        print(f"  f={f:<5} em={em} psf={psf:<4} -> {delta:+.2f}%")


if __name__ == "__main__":
    main()
