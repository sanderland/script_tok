"""Plot the overshoot factor sweep for all pruning-method variants on English.

Compares:
  * MinGram-score, no L cap     -- existing native MinGram
  * MinGram-MI hybrid, no L cap -- new (L=1024 used as effective unbounded)
  * MinGram-MI hybrid, L=32     -- new (paper-effective L for our pretokenizer)
  * PathPiece-B, L=32           -- canonical PathPiece-V at paper-effective L

All values are compression Δ vs Unigram on the English Goldfish corpus.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from paper_utils.hybrid.utils import paper_figure_path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
GRID_CACHE = RESULTS_DIR / "cache_train_eval_compression_grid.json"
MG_ABL_DIR = REPO_ROOT / "results/mingram/ablations"
PP_ABL_DIR = REPO_ROOT / "results/pathpiece/ablations"
OUT_PNG = paper_figure_path("pruning_method_fsweep.png", extra=True)

UNIGRAM_BASELINE_ENG = 69_729_749
BPE_DELTA_ENG = -0.73  # for horizontal reference line


def _mg_score_noL_trace(cache: dict) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for k, v in cache.items():
        if "mingram" not in k or "fineweb_en_5gb/eng_latn_fishfood" not in k:
            continue
        m = re.search(r"mingram_f([\d.]+)_em(\d+)_p([\d.]+)", k)
        if not m:
            continue
        f, em, p = float(m.group(1)), int(m.group(2)), float(m.group(3))
        if em != 2 or p != 0.0:
            continue
        delta = (int(v) - UNIGRAM_BASELINE_ENG) / UNIGRAM_BASELINE_ENG * 100
        out.append((f, delta))
    return sorted(out)


def _read_ablation_summaries(dir_path: Path, tag_pattern: str) -> list[tuple[float, float]]:
    rx = re.compile(tag_pattern)
    out: list[tuple[float, float]] = []
    for summary_path in sorted(dir_path.glob("*.summary.json")):
        m = rx.match(summary_path.stem.replace(".summary", ""))
        if not m:
            continue
        d = json.loads(summary_path.read_text())
        out.append((float(d["overshoot_factor"]), float(d["delta_vs_unigram_pct"])))
    return sorted(out)


def main() -> None:
    cache = json.loads(GRID_CACHE.read_text())

    mg_score_noL = _mg_score_noL_trace(cache)
    # PathPiece ablation summaries: collect L=32 and L=1000 (effective no-L) traces.
    # f ≈ (init_vocab − atomic) / 32768 ; atomic = 1916 for scriptenc_cb
    pp_L32: list[tuple[float, float]] = []
    pp_noL: list[tuple[float, float]] = []
    for sp in sorted(PP_ABL_DIR.glob("*.summary.json")):
        d = json.loads(sp.read_text())
        if d.get("init") != "bpe":
            continue
        iv = int(d["init_vocab_size"])
        f = round((iv - 1916) / 32768, 2)
        L = d.get("max_token_width")
        if L == 32:
            pp_L32.append((f, float(d["delta_vs_unigram_pct"])))
        elif L >= 64:  # treat L=1000 as "no L"
            pp_noL.append((f, float(d["delta_vs_unigram_pct"])))
    # Also include the headline main-table PP-B L=32 (f≈8) from the grid cache.
    pp_main_keys = [k for k in cache if "/pathpiece_bpe/pathpiece_bpe_iv262144_L32_" in k
                    and "fineweb_en_5gb/eng_latn_fishfood" in k]
    for k in pp_main_keys:
        tokens = int(cache[k])
        delta = (tokens - UNIGRAM_BASELINE_ENG) / UNIGRAM_BASELINE_ENG * 100
        pp_L32.append((8.0, delta))
    pp_L32 = sorted(set(pp_L32))
    pp_noL = sorted(set(pp_noL))

    series = [
        ("MinGram score (no L)",      mg_score_noL, {"color": "#1f77b4", "marker": "o", "linestyle": "-"}),
        ("PathPiece-B L=32",          pp_L32,       {"color": "#2ca02c", "marker": "^", "linestyle": "-"}),
        ("PathPiece-B no L (L=1000)", pp_noL,       {"color": "#2ca02c", "marker": "v", "linestyle": ":"}),
    ]

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)

    for label, points, style in series:
        if not points:
            print(f"  [skip empty] {label}")
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, label=label, linewidth=1.8, markersize=7, alpha=0.9, **style)
        print(f"  {label}: {len(points)} points  best={min(ys):.2f}% at f={xs[ys.index(min(ys))]:g}")

    ax.axhline(0.0, color="#444", lw=0.7, linestyle="--", alpha=0.55, zorder=0)
    ax.text(1.02, 0.05, "Unigram", color="#444", fontsize=8, ha="right", va="bottom")
    ax.axhline(BPE_DELTA_ENG, color="#888", lw=0.7, linestyle=":", alpha=0.6, zorder=0)
    ax.text(1.02, BPE_DELTA_ENG - 0.07, "BPE 32K", color="#888", fontsize=8, ha="right", va="top")

    ax.set_xscale("log")
    ax.set_xticks([1.0, 1.25, 1.5, 2, 3, 4, 5, 8])
    ax.set_xticklabels(["1.0", "1.25", "1.5", "2", "3", "4", "5", "8"])
    ax.set_xlabel("Overshoot factor $f$ (=init_vocab/target)")
    ax.set_ylabel("Compression $\\Delta$ vs Unigram (\\%, English Goldfish)")
    ax.set_title("Pruning-method × seed-budget × token-length-cap, on English")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # lower delta = better, put it visually higher

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUT_PNG}")


if __name__ == "__main__":
    main()
