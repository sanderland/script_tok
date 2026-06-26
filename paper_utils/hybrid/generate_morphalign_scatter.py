#!/usr/bin/env python3
"""Generate Appendix A scatter panels in the compression-MorphAlign plane.

One panel is produced for each MorphAlign-evaluated language plus the mean.
Each panel places BPE, Unigram, and FSP as static reference points, and traces
Unigram-BPE-Init, FSP-BPE-Init, and MinGram across the overshoot factor sweep.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from paper_utils.hybrid.train_hybrid import OVERSHOOT_FACTORS, get_model_path as get_hybrid_model_path
from paper_utils.hybrid.utils import (
    FSP_OVERRIDES,
    FSP_PARAMS,
    load_cache,
    morphalign_paper_score,
    paper_figure_path,
    save_cache,
)
from paper_utils.hybrid.train_mingram import ADDITIONAL_VOCAB_SIZE, get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as get_unigram_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.pathpiece import PathPieceModel
from script_bpe.tokenizers.unigram import UnigramModel

RESULTS_DIR = Path("results/hybrid")
MORPHALIGN_CACHE_PATH = RESULTS_DIR / "cache_morphalign_scatter.json"
MORPHALIGN_SEGMENTED_DIR = RESULTS_DIR / "morphalign_segmented"

MORPH_TOK_EVAL_DIR = Path(__file__).parents[2] / "eval/morph-tok-eval"
MORPHALIGN_THRESHOLDS = [0.01]
MORPHALIGN_ITERATIONS = 10
MORPHALIGN_MODEL = "IBM1"
MORPHALIGN_METRIC_NAME = "test-morpho-score-mean-split-0.01-IBM1"
PLOT_EM = 2
PLOT_P = 0.0
DEFAULT_F = 1.15

LANGUAGE_CONFIGS = [
    {
        "lang": "eng",
        "label": "English",
        "train_corpus": "fineweb_en_5gb",
        "eval_corpus": "eng_latn_fishfood",
        "gold_file": MORPH_TOK_EVAL_DIR / "data/morpho/eng-unimorph2uniseg_CELEX.tsv",
    },
    {
        "lang": "deu",
        "label": "German",
        "train_corpus": "fineweb_de_5gb",
        "eval_corpus": "deu_latn_fishfood",
        "gold_file": MORPH_TOK_EVAL_DIR / "data/morpho/deu-unimorph2uniseg_CELEX.tsv",
    },
    {
        "lang": "fin",
        "label": "Finnish",
        "train_corpus": "fineweb_fi_5gb",
        "eval_corpus": "fin_latn_fishfood",
        # MorphyNet→UniSegments morpheme segmentation (matches the eng/deu CELEX
        # methodology and the morph-tok-eval paper). The raw fin-unimorph.tsv dump
        # (1M inflected forms, 75k missing segmentations) does not discriminate
        # between tokenizers — every method captures the same high-frequency case
        # endings — and inflates Finnish ~10x above eng/deu.
        "gold_file": MORPH_TOK_EVAL_DIR / "data/morpho/fin-unimorph2uniseg_morphynet.tsv",
    },
]

TRACE_STYLE = {
    "bpe_init": {"color": "#444444", "marker": "o", "label": "Unigram-BPE-Init", "filled": False},
    "bpe_init_fsp": {"color": "#9467bd", "marker": "^", "label": "FSP-BPE-Init", "filled": False},
    "mingram": {"color": "#1f77b4", "marker": "P", "label": "MinGram", "filled": True},
}

REFERENCE_STYLE = {
    "Default": {"color": "#444444", "marker": "o", "label": "Unigram", "filled": True},
    "BPE": {"color": "#ff7f0e", "marker": "D", "label": "BPE", "filled": True},
    "FSP": {"color": "#9467bd", "marker": "^", "label": "FSP", "filled": True},
    "PathPiece-B": {"color": "#e377c2", "marker": "s", "label": "PathPiece", "filled": True},
    "ConvexTok": {"color": "#17becf", "marker": "*", "label": "ConvexTok", "filled": True},
}

TRACE_ORDER = ["bpe_init", "bpe_init_fsp", "mingram"]


def _load_align_module():
    align_path = MORPH_TOK_EVAL_DIR / "align.py"
    spec = importlib.util.spec_from_file_location("morph_tok_eval_align", align_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALIGN_MODULE = _load_align_module()


def _tokenize_word(model, word: str) -> list[str]:
    token_ids = model.encode(word)
    tokens: list[str] = []
    pending_atomic_tokens: list[int] = []
    for token_id in token_ids:
        pending_atomic_tokens.extend(int(tid) for tid in model.tokens[token_id].atomic_tokens)
        decoded = model.pretokenizer.try_decode_strict(pending_atomic_tokens)
        if decoded is not None:
            tokens.append(decoded)
            pending_atomic_tokens = []
    assert not pending_atomic_tokens, f"Tokenization ended inside a partial character: {word!r}"
    assert "".join(tokens) == word, f"Tokenization mismatch: {word} != {' '.join(tokens)}"
    return tokens


def _load_variant_model(path: Path) -> MinGramModel:
    return MinGramModel.load(str(path))


def _write_segmented_tsv(model, gold_file: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_file, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            word, tag, _segments = line.rstrip("\n").split("\t")
            tokenized = _tokenize_word(model, word)
            print(word, tag, "|".join(tokenized), sep="\t", file=f_out)


def morphalign_score(model, gold_file: Path, cache: dict, cache_key: str) -> float:
    if cache_key in cache:
        return float(cache[cache_key])
    out_tsv = MORPHALIGN_SEGMENTED_DIR / f"{cache_key.replace('/', '__')}.tsv"
    _write_segmented_tsv(model, gold_file, out_tsv)
    results, _model_name = ALIGN_MODULE.evaluate_segmentations(
        str(gold_file),
        str(out_tsv),
        MORPHALIGN_THRESHOLDS,
        MORPHALIGN_ITERATIONS,
        MORPHALIGN_MODEL,
        skip_gold_train=True,
    )
    value = float(results[MORPHALIGN_METRIC_NAME])
    cache[cache_key] = value
    save_cache(cache, MORPHALIGN_CACHE_PATH)
    return value


def load_baseline(train_corpus: str, eval_corpus_name: str, cache: dict, lang: str) -> dict | None:
    baseline_path = get_unigram_model_path(train_corpus, DEFAULTS)
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}")
        return None
    baseline_model = UnigramModel.load(str(baseline_path))
    tokens_key = f"tokens/{eval_corpus_name}/{lang}/ref/Default"
    if tokens_key in cache:
        total_tokens_len = int(cache[tokens_key])
    else:
        eval_corpus = load_corpus_by_name(eval_corpus_name, baseline_model.pretokenizer)
        perf = baseline_model.corpus_performance(eval_corpus)
        total_tokens_len = int(perf["total_tokens_len"])
        cache[tokens_key] = total_tokens_len
        save_cache(cache, MORPHALIGN_CACHE_PATH)
    return {"tokens": total_tokens_len, "model": baseline_model}


def cached_eval_total_tokens(model, eval_corpus, cache: dict, cache_key: str, eval_corpus_name: str) -> int:
    tokens_key = f"tokens/{eval_corpus_name}/{cache_key}"
    if tokens_key in cache:
        return int(cache[tokens_key])
    perf = model.corpus_performance(eval_corpus)
    total_tokens_len = int(perf["total_tokens_len"])
    cache[tokens_key] = total_tokens_len
    save_cache(cache, MORPHALIGN_CACHE_PATH)
    return total_tokens_len


def _parse_mingram_stem(stem: str) -> tuple[float, int, float, int] | None:
    """Parse (f, em, p, n) from filename stem like 'mingram_f1.1_em2_p0.0_n32768_<hash>'."""
    import re
    m = re.match(r"mingram_f([\d.]+)_em(\d+)_p([\d.]+)_n(\d+)_", stem)
    if not m:
        return None
    return float(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4))


def collect_points(
    train_corpus: str,
    eval_corpus_name: str,
    gold_file: Path,
    baseline: dict,
    cache: dict,
    lang: str,
) -> list[dict]:
    # MinGram models live under results/mingram/{train_corpus}/ (resolved via
    # get_mingram_model_path), not results/hybrid/{train_corpus}/. Derive the
    # directory from the canonical path helper so this collector (used by the
    # 2D figure) stays in sync with collect_mingram_points.
    model_dir = get_mingram_model_path(train_corpus, 1.0, PLOT_EM, PLOT_P).parent
    if not model_dir.exists():
        return []
    points = []
    cache_hits = 0
    cache_misses = 0
    eval_corpus = None  # loaded lazily on first cache miss
    for path in sorted(model_dir.glob("mingram_*.model.json.gz")):
        parsed = _parse_mingram_stem(path.stem.replace(".model.json", ""))
        if parsed is None:
            continue
        f, em, p, vocab_size = parsed
        if em != PLOT_EM or vocab_size != ADDITIONAL_VOCAB_SIZE:
            continue

        cache_key = f"{lang}/mingram/{path.name}"
        tokens_key = f"tokens/{eval_corpus_name}/{cache_key}"

        if cache_key in cache and tokens_key in cache:
            cache_hits += 1
            score = float(cache[cache_key])
            tokens = int(cache[tokens_key])
        else:
            cache_misses += 1
            if eval_corpus is None:
                eval_corpus = load_corpus_by_name(eval_corpus_name, baseline["model"].pretokenizer)
            model = _load_variant_model(path)
            tokens = cached_eval_total_tokens(model, eval_corpus, cache, cache_key, eval_corpus_name)
            print(f"[{lang}] MorphAlign cache miss (mingram): {path.name}")
            score = morphalign_score(model, gold_file, cache, cache_key)

        rel_tok = (tokens - baseline["tokens"]) / baseline["tokens"] * 100
        points.append({"f": f, "em": em, "p": p, "variant": "mingram",
                        "rel_tok": float(rel_tok), "morphalign": float(score)})

    print(f"[{lang}] mingram: {len(points)} points (hits={cache_hits}, misses={cache_misses}) [em={PLOT_EM}]")
    return points


def collect_bpe_init_points(
    method: str,
    train_corpus: str,
    eval_corpus_name: str,
    gold_file: Path,
    baseline: dict,
    cache: dict,
    lang: str,
) -> list[dict]:
    """Collect MorphAlign + compression points for bpe_init or bpe_init_fsp across f values."""
    overrides = FSP_OVERRIDES if method == "bpe_init_fsp" else {}
    points = []
    eval_corpus = None  # loaded lazily on first cache miss
    for f in OVERSHOOT_FACTORS:
        path = get_hybrid_model_path(train_corpus, {**DEFAULTS, **overrides, "overshoot_factor": f})
        if not path.exists():
            continue
        cache_key = f"{lang}/{method}/f{f}/{path.name}"
        tokens_key = f"tokens/{eval_corpus_name}/{cache_key}"
        # Load model only when at least one value needs computing
        if cache_key not in cache or tokens_key not in cache:
            if eval_corpus is None:
                eval_corpus = load_corpus_by_name(eval_corpus_name, baseline["model"].pretokenizer)
            model = UnigramModel.load(str(path))
            tokens = cached_eval_total_tokens(model, eval_corpus, cache, cache_key, eval_corpus_name)
            score = morphalign_score(model, gold_file, cache, cache_key)
        else:
            tokens = int(cache[tokens_key])
            score = float(cache[cache_key])
        rel_tok = (tokens - baseline["tokens"]) / baseline["tokens"] * 100
        points.append({"f": float(f), "variant": method, "rel_tok": float(rel_tok), "morphalign": float(score)})
    return points


def collect_pathpiece_refs(
    train_corpus: str,
    eval_corpus_name: str,
    gold_file: Path,
    baseline: dict,
    cache: dict,
    lang: str,
) -> dict[str, dict]:
    """Return PathPiece-BPE as a reference point for the scatter."""
    refs: dict[str, dict] = {}
    eval_corpus = None
    init_to_ref = {"bpe": "PathPiece-B"}
    for init in ("bpe",):
        model_path = get_pathpiece_model_path(train_corpus, init=init)
        if not model_path.exists():
            continue
        cache_key = f"{lang}/pathpiece_{init}/{model_path.name}"
        tokens_key = f"tokens/{eval_corpus_name}/{cache_key}"
        if cache_key not in cache or tokens_key not in cache:
            if eval_corpus is None:
                eval_corpus = load_corpus_by_name(eval_corpus_name, baseline["model"].pretokenizer)
            model = PathPieceModel.load(str(model_path))
            tokens = cached_eval_total_tokens(model, eval_corpus, cache, cache_key, eval_corpus_name)
            score = morphalign_score(model, gold_file, cache, cache_key)
        else:
            tokens = int(cache[tokens_key])
            score = float(cache[cache_key])
        rel_tok = (tokens - baseline["tokens"]) / baseline["tokens"] * 100
        refs[init_to_ref[init]] = {"rel_tok": float(rel_tok), "morphalign": float(score)}
    return refs


CONVEXTOK_PATH = "results/convextok_tokenizers/{train}/n32768_cmin50_mp200000_L32_det.json.gz"


def collect_convextok_refs(
    train_corpus: str,
    eval_corpus_name: str,
    gold_file: Path,
    baseline: dict,
    cache: dict,
    lang: str,
) -> dict[str, dict]:
    """ConvexTok as a single reference point (compression vs Unigram, MorphAlign)."""
    import os

    from script_bpe.tokenizers import load_tokenizer

    model_path = CONVEXTOK_PATH.format(train=train_corpus)
    if not os.path.exists(model_path):
        return {}
    cache_key = f"{lang}/convextok/{os.path.basename(model_path)}"
    tokens_key = f"tokens/{eval_corpus_name}/{cache_key}"
    if cache_key not in cache or tokens_key not in cache:
        eval_corpus = load_corpus_by_name(eval_corpus_name, baseline["model"].pretokenizer)
        model = load_tokenizer(model_path)
        tokens = cached_eval_total_tokens(model, eval_corpus, cache, cache_key, eval_corpus_name)
        score = morphalign_score(model, gold_file, cache, cache_key)
    else:
        tokens = int(cache[tokens_key])
        score = float(cache[cache_key])
    rel_tok = (tokens - baseline["tokens"]) / baseline["tokens"] * 100
    return {"ConvexTok": {"rel_tok": float(rel_tok), "morphalign": float(score)}}


def collect_mingram_points(
    train_corpus: str,
    eval_corpus_name: str,
    gold_file: Path,
    baseline: dict,
    cache: dict,
    lang: str,
) -> list[dict]:
    points = []
    eval_corpus = None
    for f in OVERSHOOT_FACTORS:
        model_path = get_mingram_model_path(train_corpus, f, PLOT_EM, PLOT_P)
        if not model_path.exists():
            continue
        cache_key = f"{lang}/mingram/{model_path.name}"
        tokens_key = f"tokens/{eval_corpus_name}/{cache_key}"
        if cache_key not in cache or tokens_key not in cache:
            if eval_corpus is None:
                eval_corpus = load_corpus_by_name(eval_corpus_name, baseline["model"].pretokenizer)
            model = MinGramModel.load(str(model_path))
            tokens = cached_eval_total_tokens(model, eval_corpus, cache, cache_key, eval_corpus_name)
            score = morphalign_score(model, gold_file, cache, cache_key)
        else:
            tokens = int(cache[tokens_key])
            score = float(cache[cache_key])
        rel_tok = (tokens - baseline["tokens"]) / baseline["tokens"] * 100
        points.append({"f": float(f), "variant": "mingram", "rel_tok": float(rel_tok), "morphalign": float(score)})
    return points


def mean_by_variant_and_f(points: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[tuple[str, float], list[dict]] = {}
    for pt in points:
        buckets.setdefault((pt["variant"], float(pt["f"])), []).append(pt)

    series: dict[str, list[dict]] = {}
    for (variant, f), pts in buckets.items():
        n = len(pts)
        series.setdefault(variant, []).append(
            {
                "variant": variant,
                "f": float(f),
                "n": n,
                "rel_tok": sum(p["rel_tok"] for p in pts) / n,
                "morphalign": sum(p["morphalign"] for p in pts) / n,
            }
        )
    return {variant: sorted(pts, key=lambda pt: pt["f"]) for variant, pts in series.items()}


def get_reference_points(
    train_corpus: str,
    eval_corpus_name: str,
    gold_file: Path,
    baseline: dict,
    cache: dict,
    lang: str,
) -> dict[str, dict]:
    refs: dict[str, dict] = {}
    eval_corpus = load_corpus_by_name(eval_corpus_name, baseline["model"].pretokenizer)

    default_score = morphalign_score(baseline["model"], gold_file, cache, f"{lang}/ref/Default")
    refs["Default"] = {"rel_tok": 0.0, "morphalign": default_score}

    bpe_path = Path("results/hybrid") / train_corpus / f"bpe_n{ADDITIONAL_VOCAB_SIZE}.model.json.gz"
    if bpe_path.exists():
        bpe_model = BPETokenizer.load(str(bpe_path))
        bpe_tokens = cached_eval_total_tokens(bpe_model, eval_corpus, cache, f"{lang}/ref/BPE", eval_corpus_name)
        bpe_score = morphalign_score(bpe_model, gold_file, cache, f"{lang}/ref/BPE")
        refs["BPE"] = {
            "rel_tok": (bpe_tokens - baseline["tokens"]) / baseline["tokens"] * 100,
            "morphalign": bpe_score,
        }

    fsp_path = get_hybrid_model_path(train_corpus, FSP_PARAMS)
    if not fsp_path.exists():
        fsp_path = get_unigram_model_path(train_corpus, FSP_PARAMS)
    if fsp_path.exists():
        fsp_model = UnigramModel.load(str(fsp_path))
        fsp_tokens = cached_eval_total_tokens(fsp_model, eval_corpus, cache, f"{lang}/ref/FSP", eval_corpus_name)
        fsp_score = morphalign_score(fsp_model, gold_file, cache, f"{lang}/ref/FSP")
        refs["FSP"] = {
            "rel_tok": (fsp_tokens - baseline["tokens"]) / baseline["tokens"] * 100,
            "morphalign": fsp_score,
        }

    # PathPiece-BPE is a single reference point, not part of this MorphAlign f-sweep.
    pp_refs = collect_pathpiece_refs(
        train_corpus=train_corpus,
        eval_corpus_name=eval_corpus_name,
        gold_file=gold_file,
        baseline=baseline,
        cache=cache,
        lang=lang,
    )
    refs.update(pp_refs)
    refs.update(collect_convextok_refs(
        train_corpus=train_corpus,
        eval_corpus_name=eval_corpus_name,
        gold_file=gold_file,
        baseline=baseline,
        cache=cache,
        lang=lang,
    ))
    return refs


def aggregate_mean_series(series_per_lang: list[dict[str, list[dict]]]) -> dict[str, list[dict]]:
    required_n = len(series_per_lang)
    buckets: dict[tuple[str, float], list[dict]] = {}
    for series in series_per_lang:
        for variant, pts in series.items():
            for pt in pts:
                buckets.setdefault((variant, float(pt["f"])), []).append(pt)

    out: dict[str, list[dict]] = {}
    for (variant, f), pts in buckets.items():
        n = len(pts)
        if n != required_n:
            continue
        out.setdefault(variant, []).append(
            {
                "variant": variant,
                "f": float(f),
                "n": n,
                "rel_tok": sum(p["rel_tok"] for p in pts) / n,
                "morphalign": sum(p["morphalign"] for p in pts) / n,
            }
        )
    return {variant: sorted(pts, key=lambda pt: pt["f"]) for variant, pts in out.items()}


def aggregate_mean_refs(refs_per_lang: list[dict[str, dict]]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for refs in refs_per_lang:
        for name, point in refs.items():
            grouped.setdefault(name, []).append(point)
    mean_refs = {}
    for name, values in grouped.items():
        n = len(values)
        mean_refs[name] = {
            "rel_tok": sum(v["rel_tok"] for v in values) / n,
            "morphalign": sum(v["morphalign"] for v in values) / n,
        }
    return mean_refs


def _scatter_style(style: dict) -> tuple[str, str, float]:
    if style["filled"]:
        return style["color"], "white", 1.1
    return "white", style["color"], 1.8


def _default_f_point(points: list[dict]) -> dict | None:
    for point in points:
        if abs(float(point["f"]) - DEFAULT_F) < 1e-9:
            return point
    return None


def _panel_xlim(trace_series: dict[str, list[dict]], refs: dict[str, dict] | None) -> tuple[float, float]:
    all_x: list[float] = []
    for points in trace_series.values():
        all_x.extend(-point["rel_tok"] for point in points)
    if refs:
        all_x.extend(-point["rel_tok"] for point in refs.values())
    if not all_x:
        return (-0.5, 0.5)
    lo = min(all_x)
    hi = max(all_x)
    span = hi - lo
    pad = max(0.06 * span, 0.2)
    return (lo - pad, hi + pad)


def plot_panel(ax, label: str, trace_series: dict[str, list[dict]], refs: dict[str, dict] | None) -> None:
    for variant in TRACE_ORDER:
        pts = trace_series.get(variant, [])
        if not pts:
            continue
        style = TRACE_STYLE[variant]
        xs = [-pt["rel_tok"] for pt in pts]
        ys = [morphalign_paper_score(pt["morphalign"]) for pt in pts]
        facecolor, edgecolor, linewidth = _scatter_style(style)
        ax.plot(xs, ys, color=style["color"], linewidth=1.8, alpha=0.85, zorder=2)
        default_point = _default_f_point(pts)
        if default_point is not None:
            ax.scatter(
                [-default_point["rel_tok"]],
                [morphalign_paper_score(default_point["morphalign"])],
                facecolors=facecolor,
                edgecolors=edgecolor,
                marker=style["marker"],
                s=78,
                linewidths=linewidth,
                zorder=3,
            )

    if refs:
        for name, ref in refs.items():
            style = REFERENCE_STYLE.get(name)
            if style is None:
                continue
            facecolor, edgecolor, linewidth = _scatter_style(style)
            ax.scatter(
                [-ref["rel_tok"]],
                [morphalign_paper_score(ref["morphalign"])],
                facecolors=facecolor,
                edgecolors=edgecolor,
                marker=style["marker"],
                s=95,
                linewidths=linewidth,
                zorder=4,
            )

    ax.set_title(label, fontsize=11)
    ax.grid(True, alpha=0.25, zorder=0)


def plot(rows: list[dict], out_png: Path):
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.15)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=180, squeeze=False, sharex=False)

    for ax, row in zip(axes.flat, rows, strict=True):
        plot_panel(ax, row["label"], row["trace_series"], row["refs"])
        ax.set_xlim(*_panel_xlim(row["trace_series"], row["refs"]))

    for ax in axes[1]:
        ax.set_xlabel("Compression Improvement over Unigram (%)", fontsize=10)
    axes[0][0].set_ylabel("MorphAlign Score $\\times 100$", fontsize=10)
    axes[1][0].set_ylabel("MorphAlign Score $\\times 100$", fontsize=10)

    axes[0][0].annotate(
        "Better",
        xy=(0.9, 0.9),
        xytext=(0.72, 0.76),
        xycoords="axes fraction",
        fontsize=8,
        fontweight="bold",
        color="#555555",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        arrowprops={"arrowstyle": "-|>", "color": "#777777", "alpha": 0.75, "linewidth": 1.3},
        ha="left",
        va="center",
    )

    legend_handles = []
    for name in TRACE_ORDER:
        style = TRACE_STYLE[name]
        facecolor, edgecolor, linewidth = _scatter_style(style)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=1.8,
                marker=style["marker"],
                markersize=7,
                markerfacecolor=facecolor,
                markeredgecolor=edgecolor,
                markeredgewidth=linewidth,
                label=style["label"],
            )
        )
    for name in ["BPE", "Default", "FSP", "PathPiece-B"]:
        if name not in REFERENCE_STYLE:
            continue
        style = REFERENCE_STYLE[name]
        facecolor, edgecolor, linewidth = _scatter_style(style)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=0.0,
                marker=style["marker"],
                markersize=8,
                markerfacecolor=facecolor,
                markeredgecolor=edgecolor,
                markeredgewidth=linewidth,
                label=style["label"],
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        fontsize=8.5,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Overshoot Factor Sweep", fontsize=14, y=1.03)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cache = load_cache(MORPHALIGN_CACHE_PATH)

    rows = []
    series_for_mean = []
    refs_for_mean = []
    for cfg in LANGUAGE_CONFIGS:
        print(f"Processing {cfg['label']}...")
        baseline = load_baseline(cfg["train_corpus"], cfg["eval_corpus"], cache, cfg["lang"])
        if baseline is None:
            rows.append({"label": cfg["label"], "trace_series": {}, "refs": {}})
            continue

        points: list[dict] = []
        points.extend(
            collect_bpe_init_points(
                method="bpe_init",
                train_corpus=cfg["train_corpus"],
                eval_corpus_name=cfg["eval_corpus"],
                gold_file=cfg["gold_file"],
                baseline=baseline,
                cache=cache,
                lang=cfg["lang"],
            )
        )
        points.extend(
            collect_bpe_init_points(
                method="bpe_init_fsp",
                train_corpus=cfg["train_corpus"],
                eval_corpus_name=cfg["eval_corpus"],
                gold_file=cfg["gold_file"],
                baseline=baseline,
                cache=cache,
                lang=cfg["lang"],
            )
        )
        points.extend(
            collect_mingram_points(
                train_corpus=cfg["train_corpus"],
                eval_corpus_name=cfg["eval_corpus"],
                gold_file=cfg["gold_file"],
                baseline=baseline,
                cache=cache,
                lang=cfg["lang"],
            )
        )
        trace_series = mean_by_variant_and_f(points)

        refs = get_reference_points(
            train_corpus=cfg["train_corpus"],
            eval_corpus_name=cfg["eval_corpus"],
            gold_file=cfg["gold_file"],
            baseline=baseline,
            cache=cache,
            lang=cfg["lang"],
        )

        rows.append({"label": cfg["label"], "trace_series": trace_series, "refs": refs})
        if trace_series:
            series_for_mean.append(trace_series)
        if refs:
            refs_for_mean.append(refs)

    mean_series = aggregate_mean_series(series_for_mean)
    mean_refs = aggregate_mean_refs(refs_for_mean)
    plot_rows = [{"label": "Mean", "trace_series": mean_series, "refs": mean_refs}] + rows

    out_png = paper_figure_path("morphalign_scatter_paper.png", extra=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plot(plot_rows, out_png)
    print(f"Saved: {out_png}")
    for row in plot_rows:
        total_mean = sum(len(v) for v in row["trace_series"].values())
        if total_mean:
            print(f"{row['label']}: trace points={total_mean} [em={PLOT_EM}, p={PLOT_P}]")


if __name__ == "__main__":
    main()
