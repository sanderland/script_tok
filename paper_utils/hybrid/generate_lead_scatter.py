#!/usr/bin/env python3
"""Generate the lead scatter figure: compression improvement vs MorphAlign.

Places all paper methods (BPE, Unigram, FSP, FSP-BPE-Init, MinGram)
as static points in the (compression improvement, MorphAlign) plane,
plus MinGram as a dotted trace parameterized by the overshoot factor f.

Axes are oriented so up-and-to-the-right is better:
  x = compression improvement over Unigram (%) across the six compression
      languages -- sign-flipped from raw token-count delta so larger positive
      = better compression
  y = MorphAlign geomean across 3 morphology-eval languages (log scale)

Compression mean is the arithmetic mean of percentage improvements (already
per-language-normalized). MorphAlign mean is the geometric mean across
eng/deu/fin because per-language MorphAlign spans roughly two orders of
magnitude (English ~1e-3, Finnish ~3e-2), and arithmetic mean would be
Finnish-dominated.

Inputs:
  results/hybrid/compression_train_eval_grid.json
  results/hybrid/cache_train_eval_compression_grid.json
  results/hybrid/cache_morphalign_scatter.json

Output:
  results/mingram_paper/figures/lead_scatter.png
"""

import json
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from paper_utils.hybrid.train_hybrid import get_model_path as get_hybrid_model_path
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import FSP_OVERRIDES, geomean, morphalign_paper_score, paper_figure_path
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as get_unigram_model_path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
GRID_JSON = RESULTS_DIR / "compression_train_eval_grid.json"
COMPRESSION_CACHE_JSON = RESULTS_DIR / "cache_train_eval_compression_grid.json"
MORPHALIGN_JSON = RESULTS_DIR / "cache_morphalign_scatter.json"
OUT_PNG = paper_figure_path("lead_scatter.png")

MAIN_F = 1.15
PLOT_EM = 2
PLOT_P = 0.0
MINGRAM_PP_F = 8.0
MINGRAM_PP_P = 0.9  # MinGram-PP MinGram uses iterative pruning (p=0.9), where MI pays off

COMPRESSION_LANGS = ["eng", "deu", "fin", "rus", "arb", "kor"]
MORPHALIGN_LANGS = ["eng", "deu", "fin"]
MINGRAM_TRACE_FS = [1.05, 1.1, 1.15, 1.25, 1.5, 2.0, 3.0, 5.0]
# MinGram-PP's candidate is f=8 (matches downstream/renyi/glitch); extend its trace to 8.0.
# Stock MinGram has no f=8 data (its sweep maxed at 5.0) and is skipped there by the data guards.
MINGRAM_PP_TRACE_FS = MINGRAM_TRACE_FS + [8.0]

COMPRESSION_CORPORA = {
    "eng": {"train": "fineweb_en_5gb", "eval": "eng_latn_fishfood"},
    "deu": {"train": "fineweb_de_5gb", "eval": "deu_latn_fishfood"},
    "fin": {"train": "fineweb_fi_5gb", "eval": "fin_latn_fishfood"},
    "rus": {"train": "fineweb_ru_5gb", "eval": "rus_cyrl_fishfood"},
    "arb": {"train": "fineweb_ar_5gb", "eval": "arb_arab_fishfood"},
    "kor": {"train": "fineweb_ko_5gb", "eval": "kor_hang_fishfood"},
}

METHOD_ORDER = [
    "bpe",
    "unigram",
    "bpe_init",
    "fsp",
    "fsp_bpe_init",
    "mingram",
    "mingram_pp",
    "pathpiece_bpe",
    "convextok",
]
METHOD_STYLE = {
    "bpe": {"color": "#D55E00", "marker": "D", "label": "BPE", "filled": True},
    "unigram": {"color": "#333333", "marker": "o", "label": "Unigram", "filled": True},
    "bpe_init": {"color": "#333333", "marker": "o", "label": "Unigram-BPE-Init", "filled": False, "size": 88},
    "fsp": {"color": "#7E57C2", "marker": "^", "label": "FSP", "filled": True},
    "fsp_bpe_init": {"color": "#7E57C2", "marker": "^", "label": "FSP-BPE-Init", "filled": False, "size": 88},
    "mingram": {"color": "#0072B2", "marker": "P", "label": "MinGram", "filled": True},
    "mingram_pp": {"color": "#009E73", "marker": "x", "label": "MinGram-PP", "filled": False, "size": 82},
    "pathpiece_ngram": {"color": "#8c564b", "marker": "s", "label": "PathPiece (n-gram)", "filled": True},
    "pathpiece_bpe": {"color": "#CC79A7", "marker": "s", "label": "PathPiece (BPE)", "filled": False, "size": 88},
    "convextok": {"color": "#C44E52", "marker": "*", "label": "ConvexTok", "filled": True, "size": 190},
}

CONVEXTOK_MODEL = "results/convextok_tokenizers/{train}/n32768_cmin50_mp200000_L32_det.json.gz"


def _convextok_point(compression_cache: dict, morph_cache: dict) -> dict | None:
    """Compute (x=compression improvement over Unigram %, y=MorphAlign) for ConvexTok,
    meaned over the MorphAlign langs (eng/deu/fin), the same way the other methods are."""
    import os

    from paper_utils.hybrid.generate_morphalign_scatter import LANGUAGE_CONFIGS, morphalign_score
    from script_bpe import get_pretokenizer
    from script_bpe.corpus.registry import load_corpus_by_name
    from script_bpe.tokenizers import load_tokenizer

    pretok = get_pretokenizer("scriptenc_cb")
    deltas = []
    for lang in COMPRESSION_LANGS:
        cfg = COMPRESSION_CORPORA[lang]
        path = CONVEXTOK_MODEL.format(train=cfg["train"])
        uni_tokens = _compression_default_tokens(compression_cache, lang)
        if not os.path.exists(path) or uni_tokens is None:
            return None
        model = load_tokenizer(path)
        ck = f"tokens/{cfg['train']}/{cfg['eval']}/convextok/{os.path.basename(path)}"
        cvx_tokens = _cache_value(compression_cache, ck)
        if cvx_tokens is None:
            ev = load_corpus_by_name(cfg["eval"], pretok)
            cvx_tokens = float(model.corpus_performance(ev)["total_tokens_len"])
            compression_cache[ck] = cvx_tokens
        deltas.append((cvx_tokens - uni_tokens) / uni_tokens * 100.0)

    morphs = []
    for lang in MORPHALIGN_LANGS:
        cfg = COMPRESSION_CORPORA[lang]
        path = CONVEXTOK_MODEL.format(train=cfg["train"])
        model = load_tokenizer(path)
        gold = next(c["gold_file"] for c in LANGUAGE_CONFIGS if c["lang"] == lang)
        mk = f"{lang}/convextok/{os.path.basename(path)}"
        morphs.append(morphalign_score(model, gold, morph_cache, mk))
    return {"x": -float(np.mean(deltas)), "y": morphalign_paper_score(float(geomean(morphs)))}


def _grid_method_key(method: str) -> str:
    return {
        "bpe": "bpe",
        "unigram": "default",
        "bpe_init": "bpe_init",
        "fsp": "fsp",
        "fsp_bpe_init": "bpe_init_fsp",
        "mingram": "mingram",
        "mingram_pp": "mingram_pp",
        "pathpiece_ngram": "pathpiece_ngram",
        "pathpiece_bpe": "pathpiece_bpe",
    }[method]


def _format_f(value: float) -> str:
    return f"{value:g}"


def _compression_deltas(grid: dict) -> dict[str, dict[str, float]]:
    """Return {method: {lang: delta_pct}} for the fineweb->fishfood pairing."""
    series = grid["fineweb->fishfood"]["series"]
    out: dict[str, dict[str, float]] = {method: {} for method in METHOD_ORDER}
    for lang in COMPRESSION_LANGS:
        row = series[lang]
        out["unigram"][lang] = 0.0
        for method in METHOD_ORDER:
            if method in ("unigram", "convextok"):  # unigram is baseline; convextok injected separately
                continue
            value = row.get(_grid_method_key(method))
            if value is not None:
                out[method][lang] = float(value)
    return out


def _cache_value(cache: dict, key: str) -> float | None:
    if key not in cache:
        return None
    return float(cache[key])


def _compression_default_tokens(cache: dict, lang: str) -> float | None:
    cfg = COMPRESSION_CORPORA[lang]
    model_name = get_unigram_model_path(cfg["train"], DEFAULTS).name
    key = f"tokens/{cfg['train']}/{cfg['eval']}/default/{model_name}"
    return _cache_value(cache, key)


def _compression_mingram_tokens(compression_cache: dict, morph_cache: dict, lang: str, f: float, em: int, p: float, prune_criterion: str = "usage_count") -> float | None:
    cfg = COMPRESSION_CORPORA[lang]
    model_name = get_mingram_model_path(cfg["train"], f, em, p, prune_criterion=prune_criterion).name
    key = f"tokens/{cfg['train']}/{cfg['eval']}/mingram/{model_name}"
    value = _cache_value(compression_cache, key)
    if value is not None:
        return value
    morph_key = f"tokens/{cfg['eval']}/{lang}/mingram/{model_name}"
    return _cache_value(morph_cache, morph_key)


def _compression_mingram_delta(compression_cache: dict, morph_cache: dict, lang: str, f: float, em: int, p: float, prune_criterion: str = "usage_count") -> float | None:
    default_tokens = _compression_default_tokens(compression_cache, lang)
    mingram_tokens = _compression_mingram_tokens(compression_cache, morph_cache, lang, f, em, p, prune_criterion)
    if default_tokens is None or mingram_tokens is None:
        return None
    return (mingram_tokens - default_tokens) / default_tokens * 100.0


def _mean_mingram_delta(compression_cache: dict, morph_cache: dict, langs: list[str], f: float, em: int, p: float) -> float | None:
    values = []
    for lang in langs:
        value = _compression_mingram_delta(compression_cache, morph_cache, lang, f, em, p)
        if value is None:
            return None
        values.append(value)
    return float(np.mean(values))


def _mingram_trace_deltas(compression_cache: dict, morph_cache: dict, langs: list[str]) -> list[tuple[float, float]]:
    points = []
    for f in MINGRAM_TRACE_FS:
        value = _mean_mingram_delta(compression_cache, morph_cache, langs, f, PLOT_EM, PLOT_P)
        if value is not None:
            points.append((f, value))
    return points


def _morphalign_references(cache: dict, lang: str) -> dict[str, float]:
    out = {}
    for ref_name, method in [("BPE", "bpe"), ("Default", "unigram"), ("FSP", "fsp")]:
        key = f"{lang}/ref/{ref_name}"
        if key in cache:
            out[method] = float(cache[key])
    return out


def _compute_fsp_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import LANGUAGE_CONFIGS, morphalign_score
    from script_bpe.tokenizers.unigram import UnigramModel

    cfg = next(cfg for cfg in LANGUAGE_CONFIGS if cfg["lang"] == lang)
    path = get_hybrid_model_path(
        cfg["train_corpus"],
        {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": f},
    )
    if not path.exists():
        return None
    model = UnigramModel.load(str(path))
    cache_key = f"{lang}/bpe_init_fsp/f{f}/{path.name}"
    return morphalign_score(model, cfg["gold_file"], cache, cache_key)


def _compute_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import LANGUAGE_CONFIGS, morphalign_score
    from script_bpe.tokenizers.unigram import UnigramModel

    cfg = next(cfg for cfg in LANGUAGE_CONFIGS if cfg["lang"] == lang)
    path = get_hybrid_model_path(
        cfg["train_corpus"],
        {**DEFAULTS, "overshoot_factor": f},
    )
    if not path.exists():
        return None
    model = UnigramModel.load(str(path))
    cache_key = f"{lang}/bpe_init/f{f}/{path.name}"
    return morphalign_score(model, cfg["gold_file"], cache, cache_key)


def _morphalign_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    train_corpus = COMPRESSION_CORPORA[lang]["train"]
    model_name = get_hybrid_model_path(
        train_corpus,
        {**DEFAULTS, "overshoot_factor": f},
    ).name
    key = f"{lang}/bpe_init/f{f}/{model_name}"
    value = _cache_value(cache, key)
    if value is not None:
        return value
    return _compute_bpe_init(cache, lang, f)


def _morphalign_fsp_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    train_corpus = COMPRESSION_CORPORA[lang]["train"]
    model_name = get_hybrid_model_path(
        train_corpus,
        {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": f},
    ).name
    key = f"{lang}/bpe_init_fsp/f{f}/{model_name}"
    value = _cache_value(cache, key)
    if value is not None:
        return value
    return _compute_fsp_bpe_init(cache, lang, f)


def _morphalign_mingram(cache: dict, lang: str, f: float, em: int, p: float, prune_criterion: str = "usage_count") -> float | None:
    train_corpus = COMPRESSION_CORPORA[lang]["train"]
    model_name = get_mingram_model_path(train_corpus, f, em, p, prune_criterion=prune_criterion).name
    key = f"{lang}/mingram/{model_name}"
    return _cache_value(cache, key)


def _morphalign_pathpiece(cache: dict, lang: str, init: str) -> float | None:
    train_corpus = COMPRESSION_CORPORA[lang]["train"]
    model_name = get_pathpiece_model_path(train_corpus, init=init).name
    key = f"{lang}/pathpiece_{init}/{model_name}"
    return _cache_value(cache, key)


def _morphalign_per_method_lang(cache: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {method: {} for method in METHOD_ORDER}
    for lang in MORPHALIGN_LANGS:
        refs = _morphalign_references(cache, lang)
        for method, score in refs.items():
            out[method][lang] = score
        bpe_init = _morphalign_bpe_init(cache, lang, MAIN_F)
        if bpe_init is not None:
            out["bpe_init"][lang] = bpe_init
        fsp_bpe_init = _morphalign_fsp_bpe_init(cache, lang, MAIN_F)
        if fsp_bpe_init is not None:
            out["fsp_bpe_init"][lang] = fsp_bpe_init
        mingram = _morphalign_mingram(cache, lang, MAIN_F, PLOT_EM, PLOT_P)
        if mingram is not None:
            out["mingram"][lang] = mingram
        mingram_pp = _morphalign_mingram(cache, lang, MINGRAM_PP_F, PLOT_EM, MINGRAM_PP_P, "mi")
        if mingram_pp is not None:
            out["mingram_pp"][lang] = mingram_pp
        for init in ("ngram", "bpe"):
            key = f"pathpiece_{init}"
            if key not in out:
                continue
            score = _morphalign_pathpiece(cache, lang, init)
            if score is not None:
                out[key][lang] = score
    return out


def _morphalign_mingram_trace(cache: dict) -> list[tuple[float, float]]:
    points = []
    for f in MINGRAM_TRACE_FS:
        values = []
        for lang in MORPHALIGN_LANGS:
            value = _morphalign_mingram(cache, lang, f, PLOT_EM, PLOT_P)
            if value is None:
                break
            values.append(value)
        if len(values) == len(MORPHALIGN_LANGS):
            points.append((f, float(geomean(values))))
    return points


def _arith_mean_or_none(values_by_lang: dict[str, float], langs: list[str]) -> float | None:
    values = [values_by_lang[lang] for lang in langs if lang in values_by_lang]
    if len(values) != len(langs):
        return None
    return float(np.mean(values))


def _geomean_or_none(values_by_lang: dict[str, float], langs: list[str]) -> float | None:
    values = [values_by_lang[lang] for lang in langs if lang in values_by_lang]
    if len(values) != len(langs):
        return None
    return float(geomean(values))


def build_points(grid: dict, compression_cache: dict, morph_cache: dict) -> tuple[dict, list[tuple[float, float, float]]]:
    """Return ({method: {'x': compression, 'y': morphalign}}, mingram_trace)."""
    compression = _compression_deltas(grid)
    morphalign = _morphalign_per_method_lang(morph_cache)

    method_points: dict[str, dict[str, float]] = {}
    for method in METHOD_ORDER:
        if method == "mingram":
            compression_mean = _mean_mingram_delta(compression_cache, morph_cache, COMPRESSION_LANGS, MAIN_F, PLOT_EM, PLOT_P)
        elif method == "mingram_pp":
            vals = [
                _compression_mingram_delta(
                    compression_cache,
                    morph_cache,
                    lang,
                    MINGRAM_PP_F,
                    PLOT_EM,
                    MINGRAM_PP_P,
                    "mi",
                )
                for lang in COMPRESSION_LANGS
            ]
            compression_mean = float(np.mean(vals)) if all(v is not None for v in vals) else None
        else:
            compression_mean = _arith_mean_or_none(compression.get(method, {}), COMPRESSION_LANGS)
        morphalign_mean = _geomean_or_none(morphalign.get(method, {}), MORPHALIGN_LANGS)
        if compression_mean is None or morphalign_mean is None:
            continue
        method_points[method] = {"x": -compression_mean, "y": morphalign_paper_score(morphalign_mean)}

    cvx_point = _convextok_point(compression_cache, morph_cache)
    if cvx_point is not None:
        method_points["convextok"] = cvx_point

    compression_trace = dict(_mingram_trace_deltas(compression_cache, morph_cache, COMPRESSION_LANGS))
    morphalign_trace = dict(_morphalign_mingram_trace(morph_cache))
    mingram_trace = [
        (f, -compression_trace[f], morphalign_paper_score(morphalign_trace[f]))
        for f in MINGRAM_TRACE_FS
        if f in compression_trace and f in morphalign_trace
    ]

    # MinGram-PP MinGram f-trace (p=0.9): compression mean over COMPRESSION_LANGS, morphalign
    # geomean over MORPHALIGN_LANGS, both at prune_criterion="mi".
    def _mi_comp(f):
        vals = [_compression_mingram_delta(compression_cache, morph_cache, lang, f, PLOT_EM, MINGRAM_PP_P, "mi") for lang in COMPRESSION_LANGS]
        return float(np.mean(vals)) if all(v is not None for v in vals) else None

    def _mi_morph(f):
        vals = [_morphalign_mingram(morph_cache, lang, f, PLOT_EM, MINGRAM_PP_P, "mi") for lang in MORPHALIGN_LANGS]
        return float(geomean(vals)) if all(v is not None for v in vals) else None

    mingram_pp_trace = []
    for f in MINGRAM_PP_TRACE_FS:
        c, m = _mi_comp(f), _mi_morph(f)
        if c is not None and m is not None:
            mingram_pp_trace.append((f, -c, morphalign_paper_score(m)))

    return method_points, mingram_trace, mingram_pp_trace


def plot(method_points: dict, mingram_trace: list[tuple[float, float, float]], out_png: Path,
         mingram_pp_trace: list[tuple[float, float, float]] | None = None) -> None:
    sns.set_style(
        "whitegrid",
        rc={
            "axes.edgecolor": "#333333",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.55,
        },
    )
    sns.set_context("paper", font_scale=1.03)

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=220)

    if mingram_trace:
        xs = [point[1] for point in mingram_trace]
        ys = [point[2] for point in mingram_trace]
        ax.plot(
            xs,
            ys,
            color=METHOD_STYLE["mingram"]["color"],
            linestyle=(0, (1.2, 2.2)),
            linewidth=1.9,
            alpha=0.82,
            zorder=3,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="white", alpha=0.7), pe.Normal()],
        )
        ax.scatter(
            xs,
            ys,
            s=18,
            marker="o",
            facecolors="white",
            edgecolors=METHOD_STYLE["mingram"]["color"],
            linewidths=0.8,
            alpha=0.85,
            zorder=4,
        )
        endpoint_styles = {
            0: {"xytext": (-3, -7), "ha": "right", "va": "top"},
            -1: {"xytext": (-3, 6), "ha": "right", "va": "bottom"},
        }
        for idx, point in ((0, mingram_trace[0]), (-1, mingram_trace[-1])):
            style = endpoint_styles[idx]
            ax.annotate(
                f"$f{{=}}{_format_f(point[0])}$",
                xy=(point[1], point[2]),
                xytext=style["xytext"],
                textcoords="offset points",
                fontsize=7,
                ha=style["ha"],
                va=style["va"],
                color=METHOD_STYLE["mingram"]["color"],
                alpha=0.9,
                bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )

    if mingram_pp_trace:
        xs = [p[1] for p in mingram_pp_trace]
        ys = [p[2] for p in mingram_pp_trace]
        ax.plot(
            xs, ys, color="#009E73", linestyle=(0, (1.2, 2.2)), linewidth=1.9, alpha=0.85, zorder=3,
            path_effects=[pe.Stroke(linewidth=3.4, foreground="white", alpha=0.7), pe.Normal()],
        )
        ax.scatter(
            xs,
            ys,
            s=18,
            marker="o",
            facecolors="white",
            edgecolors="#009E73",
            linewidths=0.8,
            alpha=0.85,
            zorder=4,
        )
        for idx, anno in ((0, {"xytext": (3, -7), "ha": "left", "va": "top"}), (-1, {"xytext": (3, 6), "ha": "left", "va": "bottom"})):
            pt = mingram_pp_trace[idx]
            ax.annotate(
                f"$f{{=}}{_format_f(pt[0])}$", xy=(pt[1], pt[2]), xytext=anno["xytext"], textcoords="offset points",
                fontsize=7, ha=anno["ha"], va=anno["va"], color="#009E73", alpha=0.9,
                bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            )

    for method in METHOD_ORDER:
        point = method_points.get(method)
        if point is None:
            continue
        style = METHOD_STYLE[method]
        marker_size = style.get("size", 104)
        if style["marker"] == "x":
            ax.scatter(
                [point["x"]],
                [point["y"]],
                c=style["color"],
                marker=style["marker"],
                s=marker_size,
                linewidths=1.5,
                zorder=5,
                label=style["label"],
                path_effects=[pe.Stroke(linewidth=2.8, foreground="white", alpha=0.7), pe.Normal()],
            )
            continue
        facecolor = style["color"] if style["filled"] else "white"
        edgecolor = "white" if style["filled"] else style["color"]
        linewidth = 1.0 if style["filled"] else 1.7
        ax.scatter(
            [point["x"]],
            [point["y"]],
            facecolors=facecolor,
            edgecolors=edgecolor,
            marker=style["marker"],
            s=marker_size,
            linewidths=linewidth,
            zorder=5,
            label=style["label"],
            path_effects=[pe.Stroke(linewidth=linewidth + 1.4, foreground="white", alpha=0.7), pe.Normal()],
        )

    ax.axvline(0.0, color="#7F7F7F", linewidth=0.7, linestyle="--", alpha=0.65, zorder=1)
    ax.set_xlabel("Compression Improvement over Unigram (%)", fontsize=9.4)
    ax.set_ylabel("MorphAlign Score", fontsize=9.4)
    ax.set_yscale("log")
    y_ticks = [0.6, 0.8, 1.0, 1.2, 1.5]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.1f}" for tick in y_ticks])
    ax.grid(True, alpha=0.45, zorder=0)

    ax.legend(
        fontsize=7.6,
        framealpha=0.94,
        loc="upper right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.6,
        borderpad=0.45,
        labelspacing=0.6,
        markerscale=0.62,
    )
    sns.despine(ax=ax)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> None:
    grid = json.loads(GRID_JSON.read_text())
    compression_cache = json.loads(COMPRESSION_CACHE_JSON.read_text())
    morph_cache = json.loads(MORPHALIGN_JSON.read_text())
    method_points, mingram_trace, mingram_pp_trace = build_points(grid, compression_cache, morph_cache)
    MORPHALIGN_JSON.write_text(json.dumps(morph_cache, indent=2))
    COMPRESSION_CACHE_JSON.write_text(json.dumps(compression_cache, indent=2))

    if not method_points:
        raise SystemExit("No method points resolved from caches. Check lookup keys.")
    if "mingram" not in method_points:
        raise SystemExit("MinGram point could not be resolved from caches.")

    print("Static points (x = compression improvement %, y = MorphAlign Score x100):")
    for method in METHOD_ORDER:
        point = method_points.get(method)
        if point is None:
            print(f"  {method:16s}  MISSING")
            continue
        print(f"  {method:16s}  x={point['x']:+.2f}  y={point['y']:.4f}")

    if mingram_trace:
        print(f"MinGram f-trace: {len(mingram_trace)} points")
        for f, x, y in mingram_trace:
            print(f"  f={_format_f(f):>4s}  x={x:+.2f}  y={y:.4f}")
    else:
        print("MinGram f-trace: empty")

    plot(method_points, mingram_trace, OUT_PNG, mingram_pp_trace=mingram_pp_trace)


if __name__ == "__main__":
    main()
