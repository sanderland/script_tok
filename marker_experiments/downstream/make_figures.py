#!/usr/bin/env python3
"""Plot bits-per-byte against compression, on both corpora it can be measured on.

The downstream result and the compression result order the arms differently: \\bnd{w}
compresses worst of the four and reaches the lowest bits-per-byte. A table makes that a
thing to notice; a scatter makes it the shape of the figure.

Compression is measured on ClimbMix, not on the held-out FineWiki slice the tables in this
work report. It is the corpus the models read and are scored on, so it is the only choice
that puts x and y on the same text; the FineWiki figure is out of domain for tokenizers
trained on FineWeb. Both were drafted as side-by-side panels and the panels came out
nearly identical, because \\bnd{w} sets the x-range in both and the plain-to-\\bnd{wpd}
distance that actually differs between the corpora (+2.87% against +0.62%) is invisible
next to it. Two indistinguishable panels argue nothing, so the corpus caveat is left to
prose, where it can be stated as a number.

No proportional reference line, for the same reason it would be the most tempting thing to
add: \\bnd{w} compresses 7.8% worse and still reaches the lowest bits-per-byte, so a line
through plain with slope 1 leaves the panel entirely and takes the y-scale with it. That
bits-per-byte barely responds to compression is the finding, not a defect of the plot.

Reads the same artifacts as make_tex_tables.py and writes beside it:

    manifest.json    chars/token per tokenizer      (train_matched.py)
    results.tsv      bits-per-byte per run          (collect_results.py)
    text_stats.json  bytes/token on ClimbMix        (measure_text_stats.py)

    uv run python marker_experiments/downstream/make_figures.py
"""

import csv
import json
import os
import statistics

import cyclopts
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
GENERATED = os.path.join(REPO, "marker_experiments", "paper", "generated")

ARM_ORDER = ["plain", "bnd_w", "bnd_wp", "bnd_wpd", "bnd_wpd_caps"]
ARM_LABEL = {
    "plain": "plain",
    "bnd_w": "bnd_w",
    "bnd_wp": "bnd_wp",
    "bnd_wpd": "bnd_wpd",
    "bnd_wpd_caps": "bnd_wpd_caps",
}

# Two colours, one job: separate the baseline from the variants. Identity is carried by a
# direct label on every point, so colour is never the only channel. Validated as a
# categorical pair on the light surface (CVD dEE 24.7 protan, 33.6 normal, both >= 3:1).
BASELINE_COLOR = "#eb6834"
VARIANT_COLOR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd6"

app = cyclopts.App()


def _load(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"missing {what}: {path}")
    with open(path) as f:
        return json.load(f)


def _bpb_by_arm(results_path):
    """Mean and sample sd of bits-per-byte per true byte, per arm."""
    if not os.path.exists(results_path):
        raise SystemExit(f"missing results: {results_path}")
    rows = list(csv.DictReader(open(results_path), delimiter="\t"))
    by_arm = {}
    for r in rows:
        val = r.get("val_bpb_true") or ""
        if not val.strip():
            continue
        by_arm.setdefault(r["arm"], []).append(float(val))
    return {
        arm: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))
        for arm, v in by_arm.items()
    }


def _panel(ax, arms, xs, bpb, xlabel):
    for arm in arms:
        x = xs[arm]
        mean, sd, _ = bpb[arm]
        color = BASELINE_COLOR if arm == "plain" else VARIANT_COLOR
        ax.errorbar(
            x, mean, yerr=sd, fmt="o", markersize=6, color=color,
            ecolor=color, elinewidth=1.2, capsize=3, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8,
        )

    # Label every point, so identity never depends on colour. Points right of the panel
    # midpoint label leftwards, so no label runs off the axis.
    lo, hi = min(xs[a] for a in arms), max(xs[a] for a in arms)
    mid = (lo + hi) / 2
    span = (hi - lo) or 1.0
    for arm in arms:
        x, (mean, _, _) = xs[arm], bpb[arm]
        right = x <= mid
        # Arms that land on top of each other in x get their labels pushed apart in y,
        # or the lower one's text sits on the upper one's error bar. bnd_wpd and
        # bnd_wpd_caps are 0.005 bytes/token apart and need this.
        crowd = [a for a in arms if a != arm and abs(xs[a] - x) < 0.03 * span]
        dy = 0.0
        if crowd:
            dy = 6.0 if mean >= max(bpb[a][0] for a in crowd) else -6.0
        ax.annotate(
            ARM_LABEL[arm], (x, mean),
            xytext=(7 if right else -7, dy), textcoords="offset points",
            ha="left" if right else "right", va="center",
            fontsize=7.5, color=INK,
        )

    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7.5, colors=MUTED, length=3)
    pad = (hi - lo) * 0.22 or 0.05
    ax.set_xlim(lo - pad, hi + pad)


@app.default
def main(
    manifest: str = os.path.join(GENERATED, "manifest.json"),
    results: str = os.path.join(GENERATED, "results.tsv"),
    text_stats: str = os.path.join(GENERATED, "text_stats.json"),
    out: str = os.path.join(GENERATED, "bpb_vs_compression"),
) -> None:
    """Write <out>.pdf and <out>.png.

    Args:
        manifest: Merged tokenizer manifest (chars/token on the held-out FineWiki slice).
        results: TSV from collect_results.py (val_bpb_true per run).
        text_stats: JSON from measure_text_stats.py (bytes/token on ClimbMix).
        out: Output path without extension.
    """
    man = {v["arm"]: v for v in _load(manifest, "manifest").values()}
    stats = _load(text_stats, "text stats")["arms"]
    bpb = _bpb_by_arm(results)

    arms = [a for a in ARM_ORDER if a in bpb and a in man and a in stats]
    if len(arms) < 2:
        raise SystemExit(f"need at least two arms with all three artifacts, have {arms}")
    missing = sorted(set(bpb) - set(arms))
    if missing:
        print(f"[figures] no compression artifact for {', '.join(missing)}; omitted")

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(4.4, 3.0))

    _panel(
        ax, arms, {a: stats[a]["true_bytes_per_token"] for a in arms}, bpb,
        "bytes / token on ClimbMix  (higher is better)",
    )
    ax.set_ylabel("val bits per byte  (lower is better)", fontsize=8, color=MUTED)

    n = min(v[2] for v in bpb.values() if v[2])
    fig.text(
        0.5, -0.02,
        f"Error bars: sample standard deviation over {n} paired seeds.",
        ha="center", fontsize=7, color=MUTED,
    )
    fig.tight_layout()

    # Reported here rather than plotted: the same arms on the held-out FineWiki slice,
    # where the compression differences are several times larger.
    print("[figures] chars/token on held-out FineWiki, for the caption:")
    base = man["plain"]["eval_chars_per_token"] if "plain" in man else None
    for arm in arms:
        cpt = man[arm]["eval_chars_per_token"]
        delta = f"{100 * (cpt - base) / base:+.2f}%" if base else ""
        print(f"           {ARM_LABEL[arm]:<14} {cpt:.4f}  {delta}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"[figures] wrote {out}.pdf and {out}.png for arms: {', '.join(arms)}")


if __name__ == "__main__":
    app()
