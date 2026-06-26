#!/usr/bin/env python3
"""Generate the 2D MorphAlign vs Compression figure for the overshoot factor sweep.

Two side-by-side panels share a single axis convention:
  x = compression improvement over Default Unigram (%, larger = better)
  y = MorphAlign score (larger = better)

Left panel -- mean across English, German, Finnish:
  * MinGram, FSP-BPE-Init, Unigram-BPE-Init traces parameterised by f
  * BPE, Default Unigram, FSP baselines as single points
Right panel -- per-language detail:
  * MinGram f-trace per language with baseline glyphs (Default Unigram, BPE, FSP)

Styling matches `generate_lead_scatter.py`: same Okabe-Ito-ish palette,
filled/hollow convention for BPE-initialized variants, white-stroke path effects,
and "better" axis annotation.
"""

import re
from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from paper_utils.hybrid.generate_morphalign_scatter import (
    LANGUAGE_CONFIGS,
    MORPHALIGN_CACHE_PATH,
    PLOT_EM,
    collect_bpe_init_points,
    collect_points,
    get_reference_points,
    load_baseline,
)
from paper_utils.hybrid.utils import geomean, load_cache, morphalign_paper_score, paper_figure_path

OUT_PNG = paper_figure_path("morphalign_2d_paper.png", appendix=True, extra=True)

HIGHLIGHT_F = 1.15
EXCLUDE_F: set[float] = set()

# Palette + markers mirror generate_lead_scatter.py so the two figures read as a pair.
COLOR_MINGRAM = "#0072B2"
COLOR_FSP = "#7E57C2"
COLOR_UNIGRAM = "#333333"
COLOR_BPE = "#D55E00"
COLOR_PATHPIECE = "#CC79A7"
COLOR_CONVEXTOK = "#009E73"

METHOD_STYLES = {
    "mingram": {
        "color": COLOR_MINGRAM,
        "marker": "P",
        "linestyle": (0, (1.2, 2.2)),
        "linewidth": 2.0,
        "label": "MinGram",
        "filled": True,
    },
    "mingram_pp": {
        "color": COLOR_CONVEXTOK,
        "marker": "P",
        "linestyle": "-",
        "linewidth": 1.8,
        "label": "MinGram-PP",
        "filled": True,
    },
    "bpe_init_fsp": {
        "color": COLOR_FSP,
        "marker": "^",
        "linestyle": (0, (5, 2)),
        "linewidth": 1.6,
        "label": "FSP-BPE-Init",
        "filled": False,
    },
    "bpe_init": {
        "color": COLOR_UNIGRAM,
        "marker": "o",
        "linestyle": (0, (2.5, 2.5)),
        "linewidth": 1.6,
        "label": "Unigram-BPE-Init",
        "filled": False,
    },
}

BASELINE_STYLES = {
    "BPE": {"color": COLOR_BPE, "marker": "D", "label": "BPE", "filled": True},
    "Default": {"color": COLOR_UNIGRAM, "marker": "o", "label": "Unigram", "filled": True},
    "FSP": {"color": COLOR_FSP, "marker": "^", "label": "FSP", "filled": True},
    "PathPiece-B": {"color": COLOR_PATHPIECE, "marker": "s", "label": "PathPiece-BPE", "filled": False},
    "ConvexTok": {"color": COLOR_CONVEXTOK, "marker": "*", "label": "ConvexTok", "filled": True},
}

LANG_COLORS = {
    "English": "#6F8FBF",
    "German": "#D99A73",
    "Finnish": "#78A982",
}

LEFT_LEGEND_ORDER = [
    "BPE",
    "Default",
    "bpe_init",
    "FSP",
    "bpe_init_fsp",
    "mingram",
    "mingram_pp",
    "PathPiece-B",
    "ConvexTok",
]
RIGHT_GLYPH_LEGEND_ORDER = ["BPE", "Default", "FSP", "mingram", "PathPiece-B", "ConvexTok"]


def _filter(pts: list[dict]) -> list[dict]:
    return sorted([p for p in pts if round(p["f"], 8) not in EXCLUDE_F], key=lambda p: p["f"])


def _improvement(pts: list[dict]) -> tuple[list[float], list[float]]:
    """Convert rel_tok (delta, negative = better) to improvement (positive = better)."""
    return [-p["rel_tok"] for p in pts], [morphalign_paper_score(p["morphalign"]) for p in pts]


def _scatter_marker(ax, x: float, y: float, style: dict, size: float, zorder: int = 6) -> None:
    """Place a single marker following the filled/hollow lead_scatter convention."""
    facecolor = style["color"] if style["filled"] else "white"
    edgecolor = "white" if style["filled"] else style["color"]
    linewidth = 1.0 if style["filled"] else 1.6
    ax.scatter(
        [x],
        [y],
        facecolors=facecolor,
        edgecolors=edgecolor,
        marker=style["marker"],
        s=size,
        linewidths=linewidth,
        zorder=zorder,
        path_effects=[pe.Stroke(linewidth=linewidth + 1.4, foreground="white", alpha=0.7), pe.Normal()],
    )


def _format_f(value: float) -> str:
    return f"{value:g}"


def _annotate_endpoints(ax, trace_pts: list[dict], color: str) -> None:
    if not trace_pts:
        return
    xs, ys = _improvement(trace_pts)
    endpoints = [
        (xs[0], ys[0], trace_pts[0]["f"], {"xytext": (-4, -7), "ha": "right", "va": "top"}),
        (xs[-1], ys[-1], trace_pts[-1]["f"], {"xytext": (-4, 7), "ha": "right", "va": "bottom"}),
    ]
    for x, y, f_val, style in endpoints:
        ax.annotate(
            f"$f{{=}}{_format_f(f_val)}$",
            xy=(x, y),
            xytext=style["xytext"],
            textcoords="offset points",
            fontsize=7,
            ha=style["ha"],
            va=style["va"],
            color=color,
            alpha=0.92,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )


def _better_arrow(ax, scale: float = 1.0) -> None:
    ax.annotate(
        "better",
        xy=(0.915, 0.92),
        xytext=(0.79, 0.81),
        xycoords="axes fraction",
        fontsize=7.6 * scale,
        color="#666666",
        arrowprops={"arrowstyle": "-|>", "color": "#777777", "alpha": 0.72, "linewidth": 1.1},
        ha="left",
        va="center",
    )


def _plot_mean_panel(ax, mean_traces: dict[str, list[dict]], mean_refs: dict[str, dict]) -> None:
    for method, style in METHOD_STYLES.items():
        pts = _filter(mean_traces.get(method, []))
        if not pts:
            continue
        xs, ys = _improvement(pts)
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            alpha=0.9,
            zorder=3,
            path_effects=[pe.Stroke(linewidth=style["linewidth"] + 1.6, foreground="white", alpha=0.65), pe.Normal()],
        )
        # Hollow open dots along the trace for non-highlight f values.
        ax.scatter(
            xs,
            ys,
            s=14,
            marker="o",
            facecolors="white",
            edgecolors=style["color"],
            linewidths=0.7,
            alpha=0.85,
            zorder=4,
        )
        # Highlighted marker at f = HIGHLIGHT_F (matches the static baseline glyph).
        for p, x, y in zip(pts, xs, ys):
            if abs(p["f"] - HIGHLIGHT_F) < 1e-9:
                _scatter_marker(ax, x, y, style, size=92, zorder=6)
                break

    for name in BASELINE_STYLES:
        ref = mean_refs.get(name)
        if ref is None:
            continue
        style = BASELINE_STYLES[name]
        x = -ref["rel_tok"]
        y = morphalign_paper_score(ref["morphalign"])
        _scatter_marker(ax, x, y, style, size=104, zorder=7)


def _plot_perlang_panel(ax, lang_mingram_pts: dict[str, list[dict]], lang_refs: dict[str, dict]) -> None:
    for cfg in LANGUAGE_CONFIGS:
        lang = cfg["label"]
        color = LANG_COLORS[lang]
        pts = _filter(lang_mingram_pts.get(lang, []))
        if pts:
            xs, ys = _improvement(pts)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.7,
                linestyle=(0, (1.2, 2.2)),
                alpha=0.9,
                zorder=3,
                path_effects=[pe.Stroke(linewidth=3.0, foreground="white", alpha=0.55), pe.Normal()],
            )
            ax.scatter(
                xs,
                ys,
                s=12,
                marker="o",
                facecolors="white",
                edgecolors=color,
                linewidths=0.7,
                alpha=0.85,
                zorder=4,
            )
            for p, x, y in zip(pts, xs, ys):
                if abs(p["f"] - HIGHLIGHT_F) < 1e-9:
                    ax.scatter(
                        [x],
                        [y],
                        marker="P",
                        facecolors=color,
                        edgecolors="white",
                        s=75,
                        linewidths=0.9,
                        zorder=6,
                        path_effects=[pe.Stroke(linewidth=2.2, foreground="white", alpha=0.7), pe.Normal()],
                    )
                    break

        refs = lang_refs.get(lang, {})
        baseline_markers = (
            ("Default", "o", True, 64),
            ("FSP", "^", False, 76),
            ("BPE", "D", True, 64),
            ("PathPiece-B", "s", False, 70),
            ("ConvexTok", "*", True, 100),
        )
        for ref_name, marker, filled, size in baseline_markers:
            if ref_name not in refs:
                continue
            r = refs[ref_name]
            facecolor = color if filled else "white"
            edgecolor = "white" if filled else color
            linewidth = 0.9 if filled else 1.5
            ax.scatter(
                [-r["rel_tok"]],
                [morphalign_paper_score(r["morphalign"])],
                marker=marker,
                facecolors=facecolor,
                edgecolors=edgecolor,
                s=size,
                linewidths=linewidth,
                zorder=7,
                path_effects=[pe.Stroke(linewidth=linewidth + 1.4, foreground="white", alpha=0.7), pe.Normal()],
            )


def _ax_axes(ax, xlabel: bool = True, ylabel: bool = True) -> None:
    if xlabel:
        ax.set_xlabel("Compression Improvement over Unigram (%)", fontsize=9.4)
    if ylabel:
        ax.set_ylabel("MorphAlign Score $\\times 100$", fontsize=9.4)
    ax.grid(True, alpha=0.45, zorder=0)


def _mean_legend(ax, mean_traces: dict, mean_refs: dict) -> None:
    handles = []
    for name in LEFT_LEGEND_ORDER:
        if name in BASELINE_STYLES:
            if name not in mean_refs:
                continue
            style = BASELINE_STYLES[name]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker=style["marker"],
                    markerfacecolor=style["color"],
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    markersize=7,
                    label=style["label"],
                )
            )
            continue
        if name not in METHOD_STYLES or name not in mean_traces:
            continue
        style = METHOD_STYLES[name]
        facecolor = style["color"] if style["filled"] else "white"
        edgecolor = style["color"]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=6,
                markerfacecolor=facecolor,
                markeredgecolor=edgecolor,
                markeredgewidth=1.0,
                label=style["label"],
            )
        )
    ax.legend(
        handles=handles,
        fontsize=7.6,
        framealpha=0.94,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        handletextpad=0.6,
        columnspacing=1.0,
        borderpad=0.45,
        labelspacing=0.4,
    )


def _perlang_legend(ax) -> None:
    lang_handles = [
        Line2D([0], [0], color=LANG_COLORS[cfg["label"]], linewidth=2.0, label=cfg["label"])
        for cfg in LANGUAGE_CONFIGS
    ]
    glyph_by_method = {
        "BPE": Line2D([0], [0], linestyle="none", marker="D", markersize=6,
                      markerfacecolor="#4A4A4A", markeredgecolor="white", markeredgewidth=0.7, label="BPE"),
        "Default": Line2D([0], [0], linestyle="none", marker="o", markersize=6,
                          markerfacecolor="#4A4A4A", markeredgecolor="white", markeredgewidth=0.7, label="Unigram"),
        "FSP": Line2D([0], [0], linestyle="none", marker="^", markersize=6,
                      markerfacecolor="white", markeredgecolor="#4A4A4A", markeredgewidth=1.4, label="FSP"),
        "mingram": Line2D([0], [0], color="#4A4A4A", linestyle=(0, (1.2, 2.2)), linewidth=1.6,
                          marker="P", markersize=7, markerfacecolor="#4A4A4A", markeredgecolor="white",
                          markeredgewidth=0.8, label=r"MinGram trace; marker $f{=}1.15$"),
        "PathPiece-B": Line2D([0], [0], linestyle="none", marker="s", markersize=6,
                              markerfacecolor="white", markeredgecolor="#4A4A4A", markeredgewidth=1.4,
                              label="PathPiece-BPE"),
        "ConvexTok": Line2D([0], [0], linestyle="none", marker="*", markersize=8,
                            markerfacecolor="#4A4A4A", markeredgecolor="white", markeredgewidth=0.7,
                            label="ConvexTok"),
    }
    glyph_handles = [glyph_by_method[name] for name in RIGHT_GLYPH_LEGEND_ORDER]
    ax.legend(
        handles=glyph_handles + lang_handles,
        fontsize=7.4,
        framealpha=0.94,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        handletextpad=0.6,
        columnspacing=1.0,
        borderpad=0.45,
        labelspacing=0.45,
    )


def plot_2d(mean_traces, mean_refs, lang_mingram_pts, lang_refs, out_png) -> None:
    sns.set_style(
        "whitegrid",
        rc={
            "axes.edgecolor": "#333333",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.55,
        },
    )
    sns.set_context("paper", font_scale=1.03)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 5.15), dpi=220)

    _plot_mean_panel(ax1, mean_traces, mean_refs)
    _plot_perlang_panel(ax2, lang_mingram_pts, lang_refs)

    _ax_axes(ax1, ylabel=False)
    _ax_axes(ax2, ylabel=False)
    ax1.set_ylabel("MorphAlign Score $\\times 100$", fontsize=9.4)
    ax2.set_ylabel("MorphAlign Score $\\times 100$", fontsize=9.4)
    ax1.set_title(f"Geomean across {len(LANGUAGE_CONFIGS)} MorphAlign-eval languages", fontsize=9.8)
    ax2.set_title("Per-language detail", fontsize=9.8)

    for ax, ticks in ((ax1, [0.7, 0.8, 0.9, 1.0, 1.2, 1.5]), (ax2, [0.4, 0.6, 0.8, 1.0, 1.5, 2.0])):
        ax.set_yscale("log")
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:.1f}" for tick in ticks])
        ax.autoscale(axis="y")
        ax.margins(y=0.08)

    _better_arrow(ax1)
    _mean_legend(ax1, mean_traces, mean_refs)
    _perlang_legend(ax2)
    sns.despine(ax=ax1)
    sns.despine(ax=ax2)

    fig.tight_layout(rect=(0.0, 0.18, 1.0, 1.0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def _mean_trace(per_lang_lists: list[list[dict]]) -> list[dict]:
    by_f: dict[float, list[dict]] = defaultdict(list)
    for pts in per_lang_lists:
        for p in pts:
            by_f[round(p["f"], 8)].append(p)
    return [
        {
            "f": f,
            "rel_tok": float(np.mean([p["rel_tok"] for p in pts])),
            "morphalign": float(geomean([p["morphalign"] for p in pts])),
        }
        for f, pts in sorted(by_f.items())
    ]


def _cache_only_baseline(eval_corpus_name: str, lang: str, cache: dict) -> dict | None:
    """Fallback baseline when the Unigram model file is missing but the cache has tokens."""
    tokens_key = f"tokens/{eval_corpus_name}/{lang}/ref/Default"
    if tokens_key not in cache:
        return None
    return {"model": None, "tokens": int(cache[tokens_key])}


_MINGRAM_STEM = re.compile(r"^mingram_f([\d.]+)_em(\d+)_p([\d.]+)_n\d+_[0-9a-f]+\.model\.json\.gz$")
_MINGRAM_PP_STEM = re.compile(r"^mingram_f([\d.]+)_em(\d+)_p([\d.]+)_pcmi_n\d+_[0-9a-f]+\.model\.json\.gz$")
_BPE_INIT_F = re.compile(r"/f([\d.]+)/")


def _cache_only_mingram(eval_corpus_name: str, lang: str, baseline_tokens: int, cache: dict) -> list[dict]:
    """Recover MinGram points (at PLOT_EM, p=0) from the cache without loading models."""
    prefix = f"{lang}/mingram/"
    tokens_prefix = f"tokens/{eval_corpus_name}/{prefix}"
    points: list[dict] = []
    for key, score in cache.items():
        if not key.startswith(prefix):
            continue
        stem = key[len(prefix):]
        match = _MINGRAM_STEM.match(stem)
        if match is None:
            continue
        f = float(match.group(1))
        em = int(match.group(2))
        p = float(match.group(3))
        if em != PLOT_EM or p != 0.0:
            continue
        tokens_key = f"{tokens_prefix}{stem}"
        if tokens_key not in cache:
            continue
        tokens = int(cache[tokens_key])
        rel = (tokens - baseline_tokens) / baseline_tokens * 100
        points.append({
            "f": f, "em": em, "p": p, "variant": "mingram",
            "rel_tok": float(rel), "morphalign": float(score),
        })
    return points


def _cache_only_mingram_pp(eval_corpus_name: str, lang: str, baseline_tokens: int, cache: dict) -> list[dict]:
    """Recover MinGram-PP points from the cache without loading models."""
    prefix = f"{lang}/mingram/"
    tokens_prefix = f"tokens/{eval_corpus_name}/{prefix}"
    points: list[dict] = []
    for key, score in cache.items():
        if not key.startswith(prefix):
            continue
        stem = key[len(prefix):]
        match = _MINGRAM_PP_STEM.match(stem)
        if match is None:
            continue
        f = float(match.group(1))
        em = int(match.group(2))
        p = float(match.group(3))
        if em != PLOT_EM or p != 0.9:
            continue
        tokens_key = f"{tokens_prefix}{stem}"
        if tokens_key not in cache:
            continue
        tokens = int(cache[tokens_key])
        rel = (tokens - baseline_tokens) / baseline_tokens * 100
        points.append({
            "f": f, "em": em, "p": p, "variant": "mingram_pp",
            "rel_tok": float(rel), "morphalign": float(score),
        })
    return points


def _cache_only_bpe_init(method: str, eval_corpus_name: str, lang: str, baseline_tokens: int, cache: dict) -> list[dict]:
    """Recover bpe_init / bpe_init_fsp points from the cache without loading models."""
    prefix = f"{lang}/{method}/"
    tokens_prefix = f"tokens/{eval_corpus_name}/{prefix}"
    points: list[dict] = []
    for key, score in cache.items():
        if not key.startswith(prefix):
            continue
        match = _BPE_INIT_F.search(key)
        if match is None:
            continue
        f = float(match.group(1))
        tokens_key = key.replace(prefix, tokens_prefix, 1)
        if tokens_key not in cache:
            continue
        tokens = int(cache[tokens_key])
        rel = (tokens - baseline_tokens) / baseline_tokens * 100
        points.append({
            "f": f, "variant": method,
            "rel_tok": float(rel), "morphalign": float(score),
        })
    return points


def _cache_only_refs(eval_corpus_name: str, lang: str, baseline_tokens: int, cache: dict) -> dict[str, dict]:
    """Recover BPE/Default/FSP reference points from the cache without loading models."""
    out: dict[str, dict] = {}
    for name in ("Default", "BPE", "FSP"):
        score_key = f"{lang}/ref/{name}"
        if score_key not in cache:
            continue
        score = float(cache[score_key])
        if name == "Default":
            out[name] = {"rel_tok": 0.0, "morphalign": score}
            continue
        tokens_key = f"tokens/{eval_corpus_name}/{lang}/ref/{name}"
        if tokens_key not in cache:
            continue
        tokens = int(cache[tokens_key])
        rel = (tokens - baseline_tokens) / baseline_tokens * 100
        out[name] = {"rel_tok": float(rel), "morphalign": score}
    for name, prefix in (("PathPiece-B", f"{lang}/pathpiece_bpe/"), ("ConvexTok", f"{lang}/convextok/")):
        score_keys = [key for key in cache if key.startswith(prefix)]
        if not score_keys:
            continue
        score_key = score_keys[0]
        tokens_key = f"tokens/{eval_corpus_name}/{score_key}"
        if tokens_key not in cache:
            continue
        tokens = int(cache[tokens_key])
        rel = (tokens - baseline_tokens) / baseline_tokens * 100
        out[name] = {"rel_tok": float(rel), "morphalign": float(cache[score_key])}
    return out


def main() -> None:
    cache = load_cache(MORPHALIGN_CACHE_PATH)

    method_lang_pts: dict[str, dict[str, list[dict]]] = {m: {} for m in METHOD_STYLES}
    lang_refs: dict[str, dict] = {}
    refs_for_mean: list[dict] = []

    for cfg in LANGUAGE_CONFIGS:
        print(f"Processing {cfg['label']}...")
        baseline = _cache_only_baseline(cfg["eval_corpus"], cfg["lang"], cache)
        if baseline is None:
            baseline = load_baseline(cfg["train_corpus"], cfg["eval_corpus"], cache, cfg["lang"])
            if baseline is None:
                print("  No baseline (cache + model both missing); skipping.")
                continue
        else:
            print("  Using cache-only baseline.")

        common = {
            "train_corpus": cfg["train_corpus"],
            "eval_corpus_name": cfg["eval_corpus"],
            "gold_file": cfg["gold_file"],
            "baseline": baseline,
            "cache": cache,
            "lang": cfg["lang"],
        }

        # MinGram: collapse multiple prune-factor samples per f to their mean.
        if baseline["model"] is None:
            ming_pts = _cache_only_mingram(cfg["eval_corpus"], cfg["lang"], baseline["tokens"], cache)
        else:
            ming_pts = [p for p in collect_points(**common) if int(p["em"]) == PLOT_EM]
        by_f: dict[float, list[dict]] = defaultdict(list)
        for p in ming_pts:
            by_f[round(p["f"], 8)].append(p)
        method_lang_pts["mingram"][cfg["label"]] = [
            {
                "f": f,
                "rel_tok": float(np.mean([p["rel_tok"] for p in pts])),
                "morphalign": float(np.mean([p["morphalign"] for p in pts])),
            }
            for f, pts in sorted(by_f.items())
        ]

        method_lang_pts["mingram_pp"][cfg["label"]] = _cache_only_mingram_pp(
            cfg["eval_corpus"],
            cfg["lang"],
            baseline["tokens"],
            cache,
        )

        for method in ("bpe_init", "bpe_init_fsp"):
            if baseline["model"] is None:
                method_lang_pts[method][cfg["label"]] = _cache_only_bpe_init(
                    method, cfg["eval_corpus"], cfg["lang"], baseline["tokens"], cache
                )
            else:
                method_lang_pts[method][cfg["label"]] = collect_bpe_init_points(method=method, **common)

        if baseline["model"] is None:
            refs = _cache_only_refs(cfg["eval_corpus"], cfg["lang"], baseline["tokens"], cache)
        else:
            refs = get_reference_points(**common)
        lang_refs[cfg["label"]] = refs
        if refs:
            refs_for_mean.append(refs)

    mean_traces: dict[str, list[dict]] = {}
    for method, by_lang in method_lang_pts.items():
        per_lang_lists = [pts for pts in by_lang.values() if pts]
        if not per_lang_lists:
            continue
        mean_pts = _filter(_mean_trace(per_lang_lists))
        if mean_pts:
            mean_traces[method] = mean_pts

    grouped: dict[str, list[dict]] = {}
    for refs in refs_for_mean:
        for name, pt in refs.items():
            grouped.setdefault(name, []).append(pt)
    mean_refs = {
        name: {
            "rel_tok": float(np.mean([p["rel_tok"] for p in pts])),
            "morphalign": float(geomean([p["morphalign"] for p in pts])),
        }
        for name, pts in grouped.items()
    }

    plot_2d(mean_traces, mean_refs, method_lang_pts["mingram"], lang_refs, OUT_PNG)


if __name__ == "__main__":
    main()
