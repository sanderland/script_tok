#!/usr/bin/env python3
"""Plot BPE-Init vs FSP-BPE-Init compression delta as a function of overshoot factor f."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from paper_utils.hybrid.utils import paper_figure_path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
DATA = RESULTS_DIR / "bpe_init_fsp_fsweep.json"
OUT_PNG = paper_figure_path("bpe_init_fsp_fsweep.png", extra=True)

data = json.loads(DATA.read_text())
factors = data["factors"]
langs = data["languages"]

fig, ax = plt.subplots(figsize=(7, 4.3))
lang_order = ["eng", "deu", "fin", "rus", "arb", "kor"]
colors = {
    "eng": "#1f77b4", "deu": "#ff7f0e", "fin": "#2ca02c",
    "rus": "#d62728", "arb": "#9467bd", "kor": "#8c564b",
}
xpos = list(range(len(factors)))
mean_bi = [
    sum(langs[lang]["by_f"][str(f)]["bpe_init"] for lang in lang_order) / len(lang_order)
    for f in factors
]
mean_bf = [
    sum(langs[lang]["by_f"][str(f)]["bpe_init_fsp"] for lang in lang_order) / len(lang_order)
    for f in factors
]

ax.plot(
    xpos,
    mean_bi,
    "--o",
    color="black",
    markersize=5,
    linewidth=2.2,
    markerfacecolor="white",
    markeredgewidth=1.1,
    zorder=6,
    label="Mean Default",
)
ax.plot(
    xpos,
    mean_bf,
    "-s",
    color="black",
    markersize=5.5,
    linewidth=2.4,
    markerfacecolor="white",
    markeredgewidth=1.1,
    zorder=7,
    label="Mean FSP",
)

for lang in lang_order:
    info = langs[lang]
    bi = [info["by_f"][str(f)]["bpe_init"] for f in factors]
    bf = [info["by_f"][str(f)]["bpe_init_fsp"] for f in factors]
    c = colors[lang]
    ax.plot(
        xpos,
        bi,
        "--o",
        color=c,
        markersize=3,
        linewidth=0.95,
        alpha=0.45,
        zorder=2,
        label=f"{info['label']} Default",
    )
    ax.plot(
        xpos,
        bf,
        "-s",
        color=c,
        markersize=3.5,
        linewidth=1.1,
        alpha=0.55,
        zorder=3,
        label=f"{info['label']} FSP",
    )

ax.set_xticks(xpos)
ax.set_xticklabels([str(f) for f in factors])
ax.set_xlim(xpos[0], xpos[-1])
ax.margins(x=0)
ax.set_xlabel(r"Overshoot factor $f$")
ax.set_ylabel(r"Compression $\Delta$ vs Default Unigram (%)")
ax.set_title("BPE-Init vs FSP-BPE-Init across overshoot factor (Goldfish eval)")
ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
ax.set_axisbelow(True)
ax.grid(True, axis="y", alpha=0.3)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    fontsize=7,
    ncol=1,
    loc="lower left",
    bbox_to_anchor=(1.01, 0.02),
    framealpha=0.9,
    borderaxespad=0.0,
)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")

print()
print("=== Mean across 6 languages ===")
print(f"{'f':>6} {'BPE-Init':>12} {'FSP-BPE-Init':>14} {'gap':>8}")
for f, bi, bf in zip(factors, mean_bi, mean_bf):
    print(f"{f:>6} {bi:+11.2f}% {bf:+13.2f}% {bi - bf:+7.2f}")
