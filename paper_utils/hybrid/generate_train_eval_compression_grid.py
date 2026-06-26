#!/usr/bin/env python3
"""Plot fixed-f compression bars for FineWeb-trained evaluation settings."""

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from paper_utils.hybrid.train_hybrid import get_bpe_model_path, get_model_path as get_hybrid_model_path
from paper_utils.hybrid.train_mingram import (
    ADDITIONAL_VOCAB_SIZE,
    RESULTS_DIR as MINGRAM_RESULTS_DIR,
    get_model_path as get_mingram_model_path,
)
from paper_utils.hybrid.train_pathpiece import (
    VARIANTS as PATHPIECE_VARIANTS,
    get_model_path as get_pathpiece_model_path,
)
from paper_utils.hybrid.utils import FSP_OVERRIDES, FSP_PARAMS, load_cache, paper_figure_path, save_cache
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as get_unigram_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.pathpiece import PathPieceModel
from script_bpe.tokenizers.unigram import UnigramModel
from script_bpe.tokenizers.convextok import ConvexTokModel

CONVEXTOK_PATH = "results/convextok_tokenizers/{train}/n32768_cmin50_mp200000_L32_det.json.gz"

RESULTS_DIR = Path("results/hybrid")
OUT_PNG = paper_figure_path("compression_train_eval_grid.png", extra=True)
OUT_JSON = RESULTS_DIR / "compression_train_eval_grid.json"
CACHE_PATH = RESULTS_DIR / "cache_train_eval_compression_grid.json"
BAR_F = 1.15
PLOT_EM = 2
MINGRAM_MI_F = 8.0
MINGRAM_MI_P = 0.9
CULTURAX_CORPUS = "CulturaX-subsample-100-bal2"
H6_TRAIN_CORPORA = {
    "fineweb_h6": "fineweb:hybrid6",
}

LANGUAGE_CONFIGS = [
    {
        "lang": "eng",
        "label": "English",
        "finewiki_train": "finewiki_en_1gb",
        "finewiki_eval": "finewiki_en_1gb",
        "fineweb_train": "fineweb_en_5gb",
        "fishfood_train": "eng_latn_fishfood",
        "fineweb_eval": "fineweb_en_5gb",
        "eval_fishfood": "eng_latn_fishfood",
        "eval_flores_plus": "flores_plus_eng_latn",
    },
    {
        "lang": "deu",
        "label": "German",
        "finewiki_train": "finewiki_de_1gb",
        "finewiki_eval": "finewiki_de_1gb",
        "fineweb_train": "fineweb_de_5gb",
        "fishfood_train": "deu_latn_fishfood",
        "fineweb_eval": "fineweb_de_5gb",
        "eval_fishfood": "deu_latn_fishfood",
        "eval_flores_plus": "flores_plus_deu_latn",
    },
    {
        "lang": "fin",
        "label": "Finnish",
        "finewiki_train": "finewiki_fi_1gb",
        "finewiki_eval": "finewiki_fi_1gb",
        "fineweb_train": "fineweb_fi_5gb",
        "fishfood_train": "fin_latn_fishfood",
        "fineweb_eval": "fineweb_fi_5gb",
        "eval_fishfood": "fin_latn_fishfood",
        "eval_flores_plus": "flores_plus_fin_latn",
    },
    {
        "lang": "rus",
        "label": "Russian",
        "finewiki_train": "finewiki_ru_1gb",
        "finewiki_eval": "finewiki_ru_1gb",
        "fineweb_train": "fineweb_ru_5gb",
        "fishfood_train": "rus_cyrl_fishfood",
        "fineweb_eval": "fineweb_ru_5gb",
        "eval_fishfood": "rus_cyrl_fishfood",
        "eval_flores_plus": "flores_plus_rus_cyrl",
    },
    {
        "lang": "arb",
        "label": "Arabic",
        "finewiki_train": "finewiki_ar_1gb",
        "finewiki_eval": "finewiki_ar_1gb",
        "fineweb_train": "fineweb_ar_5gb",
        "fishfood_train": "arb_arab_fishfood",
        "fineweb_eval": "fineweb_ar_5gb",
        "eval_fishfood": "arb_arab_fishfood",
        "eval_flores_plus": "flores_plus_arb_arab",
    },
    {
        "lang": "kor",
        "label": "Korean",
        "finewiki_train": "finewiki_ko_1gb",
        "finewiki_eval": "finewiki_ko_1gb",
        "fineweb_train": "fineweb_ko_5gb",
        "fishfood_train": "kor_hang_fishfood",
        "fineweb_eval": "fineweb_ko_5gb",
        "eval_fishfood": "kor_hang_fishfood",
        "eval_flores_plus": "flores_plus_kor_hang",
    },
]

PAIRINGS = [
    {"key": "fineweb->fineweb", "train_mode": "fineweb", "eval_mode": "fineweb", "title": "Train FineWeb -> Eval FineWeb"},
    {"key": "fineweb->fishfood", "train_mode": "fineweb", "eval_mode": "fishfood", "title": "Train FineWeb -> Eval Goldfish"},
    {
        "key": "fineweb->flores_plus",
        "train_mode": "fineweb",
        "eval_mode": "flores_plus",
        "title": "Train FineWeb -> Eval FLORES+",
    },
    {"key": "fineweb_h6->fineweb", "train_mode": "fineweb_h6", "eval_mode": "fineweb", "title": "Train FineWeb:hybrid6 -> Eval FineWeb"},
    {"key": "fineweb_h6->fishfood", "train_mode": "fineweb_h6", "eval_mode": "fishfood", "title": "Train FineWeb:hybrid6 -> Eval Goldfish"},
    {
        "key": "fineweb_h6->flores_plus",
        "train_mode": "fineweb_h6",
        "eval_mode": "flores_plus",
        "title": "Train FineWeb:hybrid6 -> Eval FLORES+",
    },
    {"key": "fishfood->fineweb", "train_mode": "fishfood", "eval_mode": "fineweb", "title": "Train Goldfish -> Eval FineWeb"},
    {"key": "fishfood->fishfood", "train_mode": "fishfood", "eval_mode": "fishfood", "title": "Train Goldfish -> Eval Goldfish"},
    {
        "key": "fishfood->flores_plus",
        "train_mode": "fishfood",
        "eval_mode": "flores_plus",
        "title": "Train Goldfish -> Eval FLORES+",
    },
]

# Headline panels (mono + multilingual FineWeb, both eval on Goldfish held-out
# from Goldfish / Chang et al., LREC 2026). The full grid also keeps in-domain
# FineWeb evaluation checks.
SHOWN_PAIRINGS = ["fineweb->fishfood", "fineweb_h6->fishfood"]

METHOD_ORDER = [
    "bpe",
    "fsp",
    "bpe_init",
    "bpe_init_fsp",
    "mingram",
    "mingram_mi",
    "pathpiece_bpe",
    "convextok",
]
# Methods plotted in the grid (all of METHOD_ORDER; PathPiece gaps fill once it's
# trained on the FineWeb/h6 corpora too).
GRID_PLOT_METHODS = [
    "bpe",
    "fsp",
    "bpe_init",
    "bpe_init_fsp",
    "mingram",
    "mingram_mi",
    "pathpiece_bpe",
    "convextok",
]
METHOD_LABELS = {
    "bpe": "BPE",
    "fsp": "FSP",
    "bpe_init": f"Unigram-BPE-Init (f={BAR_F})",
    "bpe_init_fsp": f"FSP-BPE-Init (f={BAR_F})",
    "mingram": f"MinGram (f={BAR_F})",
    "mingram_mi": f"MinGram-MI (f={MINGRAM_MI_F:g})",
    "pathpiece_ngram": "PathPiece (n-gram init)",
    "pathpiece_bpe": "PathPiece (BPE init)",
    "convextok": "ConvexTok",
}
METHOD_COLORS = {
    "bpe": "#ff7f0e",
    "fsp": "#9467bd",
    "bpe_init": "#d62728",
    "bpe_init_fsp": "#2ca02c",
    "mingram": "#1f77b4",
    "mingram_mi": "#009E73",
    "pathpiece_ngram": "#8c564b",
    "pathpiece_bpe": "#e377c2",
    "convextok": "#17becf",
}


def _parse_mingram_stem(stem: str) -> tuple[float, int, float, int] | None:
    match = re.match(r"mingram_f([\d.]+)_em(\d+)_p([\d.]+)_n(\d+)_", stem)
    if not match:
        return None
    return float(match.group(1)), int(match.group(2)), float(match.group(3)), int(match.group(4))


def _load_model(model_type: str, model_path: str):
    if model_type == "bpe":
        return BPETokenizer.load(model_path)
    if model_type == "unigram":
        return UnigramModel.load(model_path)
    if model_type == "mingram":
        return MinGramModel.load(model_path)
    if model_type == "pathpiece":
        return PathPieceModel.load(model_path)
    if model_type == "convextok":
        return ConvexTokModel.load(model_path)
    raise ValueError(f"Unknown model_type: {model_type}")


def _metadata_tokens(model, train_corpus: str, eval_corpus_name: str) -> int | None:
    if eval_corpus_name != train_corpus:
        return None
    for key in ("performance_train", "performance"):
        perf = model.metadata.get(key)
        if perf is not None:
            return int(perf["total_tokens_len"])
    perf_eval = model.metadata.get("performance_eval")
    if perf_eval is not None and model.metadata.get("eval_corpus") == train_corpus:
        return int(perf_eval["total_tokens_len"])
    return None


def _evaluate_task(task: dict) -> tuple[str, int]:
    model = _load_model(task["model_type"], task["model_path"])
    cached_train_tokens = _metadata_tokens(model, task["train_corpus"], task["eval_corpus"])
    if cached_train_tokens is not None:
        return task["cache_key"], cached_train_tokens

    eval_corpus = load_corpus_by_name(task["eval_corpus"], model.pretokenizer)
    perf = model.corpus_performance(eval_corpus)
    return task["cache_key"], int(perf["total_tokens_len"])


def _cache_key(train_corpus: str, eval_corpus_name: str, method: str, model_path: Path) -> str:
    return f"tokens/{train_corpus}/{eval_corpus_name}/{method}/{model_path.name}"


def _series_specs(pairing: dict) -> list[dict]:
    if pairing["train_mode"] in ("finewiki", "fineweb", "fishfood"):
        specs = []
        for cfg in LANGUAGE_CONFIGS:
            eval_corpus = {
                "finewiki": cfg["finewiki_eval"],
                "fineweb": cfg["fineweb_eval"],
                "fishfood": cfg["eval_fishfood"],
                "flores_plus": cfg["eval_flores_plus"],
            }[pairing["eval_mode"]]
            specs.append(
                {
                    "key": cfg["lang"],
                    "label": cfg["label"],
                    "train_corpus": cfg[f"{pairing['train_mode']}_train"],
                    "eval_corpus": eval_corpus,
                }
            )
        return specs

    if pairing["train_mode"] in H6_TRAIN_CORPORA:
        train_corpus = H6_TRAIN_CORPORA[pairing["train_mode"]]
        eval_key = {
            "finewiki": "finewiki_eval",
            "fineweb": "fineweb_eval",
            "fishfood": "eval_fishfood",
            "flores_plus": "eval_flores_plus",
        }[pairing["eval_mode"]]
        return [
            {
                "key": cfg["lang"],
                "label": cfg["label"],
                "train_corpus": train_corpus,
                "eval_corpus": cfg[eval_key],
            }
            for cfg in LANGUAGE_CONFIGS
        ]

    eval_key = {
        "finewiki": "finewiki_eval",
        "fishfood": "eval_fishfood",
        "flores_plus": "eval_flores_plus",
    }[pairing["eval_mode"]]
    return [
        {
            "key": cfg["lang"],
            "label": cfg["label"],
            "train_corpus": CULTURAX_CORPUS,
            "eval_corpus": cfg[eval_key],
        }
        for cfg in LANGUAGE_CONFIGS
    ]


def _task_for_model(train_corpus: str, eval_corpus_name: str, method: str, model_type: str, model_path: Path) -> dict:
    return {
        "cache_key": _cache_key(train_corpus, eval_corpus_name, method, model_path),
        "train_corpus": train_corpus,
        "eval_corpus": eval_corpus_name,
        "method": method,
        "model_type": model_type,
        "model_path": str(model_path),
    }


def _build_tasks(pairing_keys: set[str] | None = None) -> tuple[list[dict], dict[str, dict]]:
    tasks: list[dict] = []
    pairing_data: dict[str, dict] = {}

    for pairing in PAIRINGS:
        if pairing_keys is not None and pairing["key"] not in pairing_keys:
            continue

        pairing_entry = {"title": pairing["title"], "series": {}}
        pairing_data[pairing["key"]] = pairing_entry

        for series in _series_specs(pairing):
            train_corpus = series["train_corpus"]
            eval_corpus = series["eval_corpus"]
            entry = {
                "label": series["label"],
                "train_corpus": train_corpus,
                "eval_corpus": eval_corpus,
                "methods": {},
            }
            pairing_entry["series"][series["key"]] = entry

            default_path = get_unigram_model_path(train_corpus, DEFAULTS)
            if default_path.exists():
                task = _task_for_model(train_corpus, eval_corpus, "default", "unigram", default_path)
                tasks.append(task)
                entry["methods"]["default"] = {"cache_key": task["cache_key"]}

            bpe_path = get_bpe_model_path(train_corpus, ADDITIONAL_VOCAB_SIZE)
            if bpe_path.exists():
                task = _task_for_model(train_corpus, eval_corpus, "bpe", "bpe", bpe_path)
                tasks.append(task)
                entry["methods"]["bpe"] = {"cache_key": task["cache_key"]}

            fsp_path = get_unigram_model_path(train_corpus, FSP_PARAMS)
            if fsp_path.exists():
                task = _task_for_model(train_corpus, eval_corpus, "fsp", "unigram", fsp_path)
                tasks.append(task)
                entry["methods"]["fsp"] = {"cache_key": task["cache_key"]}

            bpe_init_path = get_hybrid_model_path(train_corpus, {**DEFAULTS, "overshoot_factor": BAR_F})
            if bpe_init_path.exists():
                task = _task_for_model(train_corpus, eval_corpus, "bpe_init", "unigram", bpe_init_path)
                tasks.append(task)
                entry["methods"]["bpe_init"] = {"cache_key": task["cache_key"]}

            bpe_init_fsp_path = get_hybrid_model_path(
                train_corpus,
                {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": BAR_F},
            )
            if bpe_init_fsp_path.exists():
                task = _task_for_model(train_corpus, eval_corpus, "bpe_init_fsp", "unigram", bpe_init_fsp_path)
                tasks.append(task)
                entry["methods"]["bpe_init_fsp"] = {"cache_key": task["cache_key"]}

            model_dir = MINGRAM_RESULTS_DIR / train_corpus
            mingram_cache_keys: list[str] = []
            if model_dir.exists():
                for path in sorted(model_dir.glob("mingram_*.model.json.gz")):
                    parsed = _parse_mingram_stem(path.stem.replace(".model.json", ""))
                    if parsed is None:
                        continue
                    f_value, em_value, _, vocab_size = parsed
                    if abs(f_value - BAR_F) > 1e-9 or em_value != PLOT_EM or vocab_size != ADDITIONAL_VOCAB_SIZE:
                        continue
                    task = _task_for_model(train_corpus, eval_corpus, "mingram", "mingram", path)
                    tasks.append(task)
                    mingram_cache_keys.append(task["cache_key"])
            if mingram_cache_keys:
                entry["methods"]["mingram"] = {"cache_keys": mingram_cache_keys}

            mingram_mi_path = get_mingram_model_path(
                train_corpus,
                MINGRAM_MI_F,
                PLOT_EM,
                MINGRAM_MI_P,
                prune_criterion="mi",
            )
            task = _task_for_model(train_corpus, eval_corpus, "mingram_mi", "mingram", mingram_mi_path)
            tasks.append(task)
            entry["methods"]["mingram_mi"] = {"cache_key": task["cache_key"]}

            convextok_path = Path(CONVEXTOK_PATH.format(train=train_corpus))
            if convextok_path.exists():
                task = _task_for_model(train_corpus, eval_corpus, "convextok", "convextok", convextok_path)
                tasks.append(task)
                entry["methods"]["convextok"] = {"cache_key": task["cache_key"]}

            # PathPiece (n-gram init, BPE init): one model per variant. Match the
            # main-table additional_vocab_size; ignore other configurations.
            for init in PATHPIECE_VARIANTS:
                pp_path = get_pathpiece_model_path(train_corpus, init=init)
                if init != "bpe" and not pp_path.exists():
                    continue
                method_name = f"pathpiece_{init}"
                task = _task_for_model(
                    train_corpus, eval_corpus, method_name, "pathpiece", pp_path
                )
                tasks.append(task)
                entry["methods"][method_name] = {"cache_key": task["cache_key"]}

    unique_tasks = {(task["cache_key"], task["model_path"]): task for task in tasks}
    return list(unique_tasks.values()), pairing_data


def _compute_cache(tasks: list[dict], cache: dict, max_workers: int) -> dict:
    missing_tasks = [
        task
        for task in tasks
        if task["cache_key"] not in cache and Path(task["model_path"]).exists()
    ]
    print(f"Compression bar-grid tasks: total={len(tasks)} missing={len(missing_tasks)}")
    if not missing_tasks:
        return cache

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_evaluate_task, task): task for task in missing_tasks}
        for future in as_completed(futures):
            cache_key, total_tokens = future.result()
            cache[cache_key] = int(total_tokens)
            completed += 1
            if completed % 8 == 0 or completed == len(missing_tasks):
                print(f"Completed {completed}/{len(missing_tasks)} eval tasks")
                save_cache(cache, CACHE_PATH)

    save_cache(cache, CACHE_PATH)
    return cache


def _resolve_results(pairing_data: dict[str, dict], cache: dict) -> dict[str, dict]:
    resolved: dict[str, dict] = {}
    for pairing_key, pairing in pairing_data.items():
        resolved_pairing = {"title": pairing["title"], "series": {}}
        resolved[pairing_key] = resolved_pairing

        for series_key, series in pairing["series"].items():
            methods = series["methods"]
            if "default" not in methods:
                continue
            if methods["default"]["cache_key"] not in cache:
                continue
            baseline_tokens = int(cache[methods["default"]["cache_key"]])
            resolved_series = {
                "label": series["label"],
                "train_corpus": series["train_corpus"],
                "eval_corpus": series["eval_corpus"],
            }

            for method in METHOD_ORDER:
                if method == "mingram":
                    if method not in methods:
                        resolved_series[method] = None
                        continue
                    values = [int(cache[key]) for key in methods[method]["cache_keys"] if key in cache]
                    if not values:
                        resolved_series[method] = None
                        continue
                    mean_tokens = sum(values) / len(values)
                    resolved_series[method] = (mean_tokens - baseline_tokens) / baseline_tokens * 100
                    continue

                if method not in methods:
                    resolved_series[method] = None
                    continue
                cache_key = methods[method]["cache_key"]
                if cache_key not in cache:
                    resolved_series[method] = None
                    continue
                tokens = int(cache[cache_key])
                resolved_series[method] = (tokens - baseline_tokens) / baseline_tokens * 100

            resolved_pairing["series"][series_key] = resolved_series

    return resolved


def _collect_y_values(resolved: dict[str, dict]) -> list[float]:
    values = [0.0]
    for pairing in resolved.values():
        for series in pairing["series"].values():
            for method in METHOD_ORDER:
                value = series.get(method)
                if value is not None:
                    values.append(float(value))
    return values


def plot_bars(resolved: dict[str, dict], out_png: Path, shown_keys: list[str] | None = None) -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.05)

    keys = shown_keys if shown_keys is not None else SHOWN_PAIRINGS
    shown = [p for p in PAIRINGS if p["key"] in keys]
    n = len(shown)
    ncols = 3 if n >= 3 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.0 * nrows), dpi=180, sharey=True)
    axes_flat = np.atleast_1d(axes).reshape(-1)
    y_values = _collect_y_values(resolved)
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = max((y_max - y_min) * 0.08, 0.2)
    n_langs = len(LANGUAGE_CONFIGS)
    x = np.arange(n_langs + 1)  # +1 for the mean column
    xlabels = [cfg["label"] for cfg in LANGUAGE_CONFIGS] + ["Mean"]
    n_methods = len(GRID_PLOT_METHODS)
    width = 0.8 / n_methods  # total bar span 0.8 < 1.0 spacing => clear gap between language groups
    offsets = np.linspace(-(n_methods - 1) / 2 * width, (n_methods - 1) / 2 * width, n_methods)

    for pairing, ax in zip(shown, axes_flat, strict=False):
        pairing_data = resolved[pairing["key"]]
        ax.set_title(pairing["title"], fontsize=10)
        ax.axhline(0.0, color="#222222", linewidth=0.9, alpha=0.35, zorder=1)
        ax.axvline(n_langs - 0.5, color="#888888", linewidth=0.6, alpha=0.5, linestyle=":", zorder=1)

        for idx, method in enumerate(GRID_PLOT_METHODS):
            vals = [pairing_data["series"].get(cfg["lang"], {}).get(method) for cfg in LANGUAGE_CONFIGS]
            present = [v for v in vals if v is not None]
            mean_val = float(np.mean(present)) if len(present) == n_langs else None
            vals_full = vals + [mean_val]
            ys = [value if value is not None else 0.0 for value in vals_full]
            bars = ax.bar(
                x + offsets[idx],
                ys,
                width=width,
                color=METHOD_COLORS[method],
                alpha=0.86,
                label=METHOD_LABELS[method],
                zorder=3,
            )
            for bar, value in zip(bars, vals_full, strict=True):
                if value is None:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=25, ha="right", fontsize=9)
        # bold the Mean tick label
        ax.get_xticklabels()[-1].set_fontweight("bold")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, axis="y", alpha=0.25)

    # ylabel on the leftmost axis of each row; hide any unused axes
    for row in range(nrows):
        axes_flat[row * ncols].set_ylabel("Compression Delta vs Default (%)", fontsize=10)
    for i in range(n, nrows * ncols):
        axes_flat[i].set_visible(False)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=7, alpha=0.86, label=METHOD_LABELS[method])
        for method in GRID_PLOT_METHODS
    ]
    fig.legend(
        handles=method_handles,
        loc="upper center",
        ncol=len(method_handles),
        fontsize=8.5,
        framealpha=0.92,
        bbox_to_anchor=(0.5, 0.995),
        title="Method",
        title_fontsize=9,
    )
    fig.suptitle(
        f"Compression Delta by train/eval pairing (fixed f={BAR_F}, EM={PLOT_EM} for MinGram)",
        fontsize=13,
        y=0.96,
    )
    fig.text(
        0.5,
        0.02,
        "Bars show language-specific compression delta relative to the Default tokenizer for the same train/eval pairing.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.99, 0.9))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed-f train/eval compression bars.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=min(os.cpu_count() or 4, 16),
        help="Process workers for missing token-count evaluations.",
    )
    parser.add_argument(
        "--all-panels", action="store_true",
        help="Render all 6 pairings instead of the SHOWN_PAIRINGS subset (writes _all suffix).",
    )
    parser.add_argument(
        "--pairings",
        nargs="+",
        choices=[pairing["key"] for pairing in PAIRINGS],
        help="Restrict evaluation to selected train/eval pairings and merge them into the existing JSON output.",
    )
    args = parser.parse_args()

    pairing_keys = set(args.pairings) if args.pairings else None
    tasks, pairing_data = _build_tasks(pairing_keys)
    cache = load_cache(CACHE_PATH)
    cache = _compute_cache(tasks, cache, args.max_workers)
    resolved = _resolve_results(pairing_data, cache)
    if pairing_keys is not None and OUT_JSON.exists():
        existing = json.loads(OUT_JSON.read_text())
        existing.update(resolved)
        resolved = existing
    save_cache(cache, CACHE_PATH)
    OUT_JSON.write_text(json.dumps(resolved, indent=2))
    if args.all_panels:
        out = OUT_PNG.with_name(OUT_PNG.stem + "_all" + OUT_PNG.suffix)
        plot_bars(resolved, out, shown_keys=[p["key"] for p in PAIRINGS])
    else:
        plot_bars(resolved, OUT_PNG)


if __name__ == "__main__":
    main()
