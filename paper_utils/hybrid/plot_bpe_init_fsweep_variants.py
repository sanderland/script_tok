#!/usr/bin/env python3
"""Appendix plot for the overshoot factor sweep.

The plot shows compression deltas relative to Default Unigram as the BPE-derived
initial vocabulary overshoot factor varies.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_utils.hybrid.utils import paper_figure_path
from paper_utils.hybrid.train_mingram import get_model_path as _mingram_model_path
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as _unigram_model_path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
DATA = RESULTS_DIR / "bpe_init_fsp_fsweep.json"
COMPRESSION_CACHE = RESULTS_DIR / "cache_train_eval_compression_grid.json"

LANG_ORDER = ["eng", "deu", "fin", "rus", "arb", "kor"]
# (train corpus, fish-food eval corpus) per language, for the p=0.9 iterative-pruning trace.
_LANG_CORPORA = {
    "eng": ("fineweb_en_5gb", "eng_latn_fishfood"),
    "deu": ("fineweb_de_5gb", "deu_latn_fishfood"),
    "fin": ("fineweb_fi_5gb", "fin_latn_fishfood"),
    "rus": ("fineweb_ru_5gb", "rus_cyrl_fishfood"),
    "arb": ("fineweb_ar_5gb", "arb_arab_fishfood"),
    "kor": ("fineweb_ko_5gb", "kor_hang_fishfood"),
}
MINGRAM_EM, MINGRAM_P09 = 2, 0.9
METHODS = {
    "bpe_init": {
        "label": "Unigram-BPE-Init",
        "color": "#6F6F6F",
        "linestyle": "--",
        "marker": "o",
    },
    "bpe_init_fsp": {
        "label": "FSP-BPE-Init",
        "color": "#7E57C2",
        "linestyle": "-",
        "marker": "s",
    },
    "mingram": {
        "label": "MinGram",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "D",
    },
}


def _load() -> tuple[list[float], dict]:
    data = json.loads(DATA.read_text())
    return data["factors"], data["languages"]


def _matrix(langs: dict, factors: list[float], method: str) -> np.ndarray:
    rows = [
        [langs[lang]["by_f"][str(f)][method] for f in factors]
        for lang in LANG_ORDER
    ]
    return np.asarray(rows, dtype=float)


def _mingram_p09_matrix(factors: list[float]) -> np.ndarray:
    """Compression delta (%) vs Default Unigram for MinGram with iterative pruning (p=0.9),
    read from the shared compression cache (token counts), same axis as the p=0 series."""
    cache = json.loads(COMPRESSION_CACHE.read_text())

    def key(train, eval_c, method, name):
        return f"tokens/{train}/{eval_c}/{method}/{name}"

    out = np.full((len(LANG_ORDER), len(factors)), np.nan)
    for i, lang in enumerate(LANG_ORDER):
        train, eval_c = _LANG_CORPORA[lang]
        base = cache.get(key(train, eval_c, "default", _unigram_model_path(train, DEFAULTS).name))
        if base is None:
            continue
        for j, f in enumerate(factors):
            model_path = _mingram_model_path(train, f, MINGRAM_EM, MINGRAM_P09)
            tok = cache.get(key(train, eval_c, "mingram", model_path.name))
            if tok is not None:
                out[i, j] = (tok - base) / base * 100.0
    return out


def _mingram_pp_matrix(factors: list[float]) -> np.ndarray:
    """Compression delta (%) vs Default Unigram for MinGram with CAREFUL (Minimum-Increase)
    pruning at p=0.9 -- same iterative schedule as the p=0.9 usage-count trace, but the prune
    criterion is the corpus-token-count increase instead of usage rank."""
    cache = json.loads(COMPRESSION_CACHE.read_text())

    def key(train, eval_c, method, name):
        return f"tokens/{train}/{eval_c}/{method}/{name}"

    out = np.full((len(LANG_ORDER), len(factors)), np.nan)
    for i, lang in enumerate(LANG_ORDER):
        train, eval_c = _LANG_CORPORA[lang]
        base = cache.get(key(train, eval_c, "default", _unigram_model_path(train, DEFAULTS).name))
        if base is None:
            continue
        for j, f in enumerate(factors):
            mp = _mingram_model_path(train, f, MINGRAM_EM, MINGRAM_P09, prune_criterion="mi")
            tok = cache.get(key(train, eval_c, "mingram", mp.name))
            if tok is not None:
                out[i, j] = (tok - base) / base * 100.0
    return out


# SCRIPT atomic-unit count; a full no-overshoot PathPiece vocab is 32768 merges + atoms.
_PATHPIECE_ATOMS = 1916


def _pathpiece_matrix(factors: list[float], prune_batch_fraction: float | None = None) -> np.ndarray:
    """Compression delta (%) vs Default Unigram for PathPiece-BPE on the SAME overshoot axis:
    init_vocab = round(f*32768) + atoms, so f=1 is a full no-overshoot vocab (matches MinGram f=1).
    prune_batch_fraction=None uses the paper main 10% pruning batch."""
    from paper_utils.hybrid.train_pathpiece import get_model_path as pp_path
    cache = json.loads(COMPRESSION_CACHE.read_text())
    pb_kw = {} if prune_batch_fraction is None else {"prune_batch_fraction": prune_batch_fraction}

    def key(train, eval_c, method, name):
        return f"tokens/{train}/{eval_c}/{method}/{name}"

    out = np.full((len(LANG_ORDER), len(factors)), np.nan)
    for i, lang in enumerate(LANG_ORDER):
        train, eval_c = _LANG_CORPORA[lang]
        base = cache.get(key(train, eval_c, "default", _unigram_model_path(train, DEFAULTS).name))
        if base is None:
            continue
        for j, f in enumerate(factors):
            model_path = pp_path(train, init="bpe", init_vocab_size=round(f * 32768) + _PATHPIECE_ATOMS, **pb_kw)
            tok = cache.get(key(train, eval_c, "pathpiece_bpe", model_path.name))
            if tok is not None:
                out[i, j] = (tok - base) / base * 100.0
    return out


def _finish(fig, stem: str) -> None:
    for suffix in ("png", "pdf"):
        out = paper_figure_path(f"{stem}.{suffix}", appendix=True, extra=suffix == "pdf")
        fig.savefig(out, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def plot_anchored_band(factors: list[float], langs: dict) -> None:
    """Compression vs overshoot factor f, as % relative to Default Unigram (absolute;
    lower is better). Per-method mean across the 6 languages, clean lines (no bands)."""
    x = np.arange(len(factors))
    fig, ax = plt.subplots(figsize=(5.9, 3.35), layout="constrained")

    for method, style in METHODS.items():
        mean = np.nanmean(_matrix(langs, factors, method), axis=0)  # already Δ% vs Default Unigram
        ax.plot(
            x, mean, label=style["label"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"],
            markersize=4.6, linewidth=2.2, markerfacecolor="white", markeredgewidth=1.1,
        )

    p09 = _mingram_p09_matrix(factors)  # MinGram iterative pruning (p=0.9): dashed, same colour
    if not np.isnan(p09).all():
        ax.plot(
            x, np.nanmean(p09, axis=0), label=r"MinGram ($p{=}0.9$)",
            color=METHODS["mingram"]["color"], linestyle="--", marker="D",
            markersize=4.6, linewidth=2.2, markerfacecolor=METHODS["mingram"]["color"], markeredgewidth=1.1,
        )

    pp10 = _pathpiece_matrix(factors)  # PathPiece-BPE, canonical pb=0.1 (MAIN_PRUNE_BATCH_FRACTION)
    if not np.isnan(pp10).all():
        ax.plot(
            x, np.nanmean(pp10, axis=0), label="PathPiece-BPE",
            color="#D55E00", linestyle="-", marker="D",
            markersize=4.6, linewidth=2.2, markerfacecolor="white", markeredgewidth=1.1,
        )

    mi = _mingram_pp_matrix(factors)  # MinGram p=0.9 with careful (MinGram-PP) pruning
    if not np.isnan(mi).all():
        ax.plot(
            x, np.nanmean(mi, axis=0), label=r"MinGram-PP ($p{=}0.9$)",
            color="#009E73", linestyle="-", marker="P",
            markersize=5.4, linewidth=2.4, markerfacecolor="#009E73", markeredgewidth=1.1,
        )

    ax.set_ylim(top=0.15)
    ax.axhline(0, color="#999999", linewidth=0.7, linestyle=":")  # Default Unigram baseline
    if 1.15 in factors:
        ax.axvline(factors.index(1.15), color="#999999", linewidth=0.6, linestyle="--", alpha=0.55, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in factors])
    ax.set_xlabel(r"Overshoot factor $f$")
    ax.set_ylabel(r"Compression $\Delta$ vs. Default Unigram (\%)")
    ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.7)
    ax.grid(True, axis="x", color="#EEEEEE", linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94)
    _finish(fig, "bpe_init_fsp_fsweep_anchored_band")


def main() -> None:
    factors, langs = _load()
    plot_anchored_band(factors, langs)


if __name__ == "__main__":
    main()
