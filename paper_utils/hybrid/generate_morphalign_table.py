#!/usr/bin/env python3
"""Generate the default-configuration MorphAlign table (T3).

Reads `results/hybrid/cache_morphalign_scatter.json` and produces a
LaTeX tabular body at `results/mingram_paper/extra/table_morphalign.tex` with one
row per method and one column per MorphAlign-eval language
(eng/deu/fin).

Methods reported:
  BPE, Unigram (Default), FSP, Unigram-BPE-Init (f=MAIN_F),
  FSP-BPE-Init (f=MAIN_F), MinGram (f=MAIN_F)

Cache key conventions (mirroring generate_morphalign_scatter.py):
  BPE    : "{lang}/ref/BPE"
  Default: "{lang}/ref/Default"
  FSP    : "{lang}/ref/FSP"
  Unigram-BPE-Init at overshoot factor f : "{lang}/bpe_init/f{f}/{model_filename}"
  FSP-BPE-Init at overshoot factor f    : "{lang}/bpe_init_fsp/f{f}/{model_filename}"
  MinGram at overshoot factor f         : "{lang}/mingram/{mingram_model_filename}"

For the factor-indexed methods we enumerate cache keys and pick the one
matching MAIN_F (and em=PLOT_EM, p=0.0 for MinGram). The MinGram cache
key contains the full model filename, which encodes f/em/p; we parse it
to filter.
"""

import json
import re
from pathlib import Path

from paper_utils.hybrid.train_mingram import ADDITIONAL_VOCAB_SIZE
from paper_utils.hybrid.train_hybrid import get_model_path as get_hybrid_model_path
from paper_utils.hybrid.utils import FSP_OVERRIDES, morphalign_paper_score, paper_table_path
from paper_utils.unigram.train_hyperparameters import DEFAULTS

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
CACHE_JSON = RESULTS_DIR / "cache_morphalign_scatter.json"
OUT_TEX = paper_table_path("table_morphalign.tex", extra=True)

MAIN_F = 1.15
PLOT_EM = 2
PLOT_P = 0.0

METHOD_ORDER = ["BPE", "Default", "FSP", "Unigram-BPE-Init", "FSP-BPE-Init", "MinGram", "PathPiece", "ConvexTok"]
METHOD_LABEL = {
    "BPE": "BPE",
    "Default": "Unigram",
    "FSP": "FSP",
    "Unigram-BPE-Init": f"Unigram-BPE-Init ($f{{=}}{MAIN_F}$)",
    "FSP-BPE-Init": f"FSP-BPE-Init ($f{{=}}{MAIN_F}$)",
    "MinGram": f"MinGram ($f{{=}}{MAIN_F}$)",
    "PathPiece": "PathPiece (BPE init)",
    "ConvexTok": "ConvexTok",
}
CONVEXTOK_MODEL_FILE = "n32768_cmin50_mp200000_L32_det.json.gz"

LANG_ORDER = ["eng", "deu", "fin"]
LANG_LABEL = {
    "eng": "English",
    "deu": "German",
    "fin": "Finnish",
}


def _compute_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import (
        LANGUAGE_CONFIGS,
        morphalign_score,
    )
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


def _compute_bpe_init_fsp(cache: dict, lang: str, f: float) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import (
        LANGUAGE_CONFIGS,
        morphalign_score,
    )
    from paper_utils.hybrid.train_hybrid import get_model_path as get_hybrid_model_path
    from paper_utils.unigram.train_hyperparameters import DEFAULTS
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


def _lookup_reference(cache: dict, lang: str, ref_name: str) -> float | None:
    key = f"{lang}/ref/{ref_name}"
    if key in cache:
        return float(cache[key])
    return None


def _lookup_bpe_init_fsp(cache: dict, lang: str, f: float) -> float | None:
    prefix = f"{lang}/bpe_init_fsp/f{f}/"
    for key, value in cache.items():
        if key.startswith(prefix):
            return float(value)
    return _compute_bpe_init_fsp(cache, lang, f)


def _lookup_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    prefix = f"{lang}/bpe_init/f{f}/"
    for key, value in cache.items():
        if key.startswith(prefix):
            return float(value)
    return _compute_bpe_init(cache, lang, f)


def _lookup_mingram(cache: dict, lang: str, f: float, em: int, p: float) -> float | None:
    prefix = f"{lang}/mingram/"
    for key, value in cache.items():
        if not key.startswith(prefix):
            continue
        if key.startswith("tokens/"):
            continue
        stem_match = re.match(
            rf"{re.escape(lang)}/mingram/mingram_f(?P<f>[\d.]+)_em(?P<em>\d+)_p(?P<p>[\d.]+)_n(?P<n>\d+)_",
            key,
        )
        if stem_match is None:
            continue
        if abs(float(stem_match["f"]) - f) > 1e-9:
            continue
        if int(stem_match["em"]) != em:
            continue
        if abs(float(stem_match["p"]) - p) > 1e-9:
            continue
        if int(stem_match["n"]) != ADDITIONAL_VOCAB_SIZE:
            continue
        return float(value)
    return None


def _lookup(cache: dict, method: str, lang: str) -> float | None:
    if method == "BPE":
        return _lookup_reference(cache, lang, "BPE")
    if method == "Default":
        return _lookup_reference(cache, lang, "Default")
    if method == "FSP":
        return _lookup_reference(cache, lang, "FSP")
    if method == "Unigram-BPE-Init":
        return _lookup_bpe_init(cache, lang, MAIN_F)
    if method == "FSP-BPE-Init":
        return _lookup_bpe_init_fsp(cache, lang, MAIN_F)
    if method == "MinGram":
        return _lookup_mingram(cache, lang, MAIN_F, PLOT_EM, PLOT_P)
    if method == "ConvexTok":
        key = f"{lang}/convextok/{CONVEXTOK_MODEL_FILE}"
        return float(cache[key]) if key in cache else None
    if method == "PathPiece":
        prefix = f"{lang}/pathpiece_bpe/"
        vals = [float(v) for k, v in cache.items() if k.startswith(prefix)]
        return vals[0] if vals else None
    raise ValueError(f"Unknown method: {method}")


def _fmt_cell(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{morphalign_paper_score(value):.2f}"


def build_table(cache: dict) -> str:
    col_spec = "l" + "r" * len(LANG_ORDER)
    header_cells = ["Method"] + [LANG_LABEL[lang] for lang in LANG_ORDER]

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    for method in METHOD_ORDER:
        cells = [METHOD_LABEL[method]]
        cells.extend(_fmt_cell(_lookup(cache, method, lang)) for lang in LANG_ORDER)
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    cache = json.loads(CACHE_JSON.read_text())
    tex = build_table(cache)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"Wrote {OUT_TEX}")
    print()
    print(tex)

    missing = []
    for method in METHOD_ORDER:
        for lang in LANG_ORDER:
            if _lookup(cache, method, lang) is None:
                missing.append((method, lang))
    if missing:
        print()
        print("WARNING: missing cells (method, lang):")
        for method, lang in missing:
            print(f"  {method} / {lang}")


if __name__ == "__main__":
    main()
