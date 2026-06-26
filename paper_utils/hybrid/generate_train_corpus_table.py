#!/usr/bin/env python3
"""Generate the appendix FineWeb-trained robustness table.

Reads `results/hybrid/compression_train_eval_grid.json` for compression,
computes missing MorphAlign cells from the trained models when needed,
and writes the extra table at `results/mingram_paper/extra/app_table_train_corpus.tex`.
Training is FineWeb-only; robustness varies evaluation between in-domain
FineWeb and held-out Goldfish corpora.
"""

import json
from pathlib import Path

from paper_utils.hybrid.generate_morphalign_scatter import (
    LANGUAGE_CONFIGS as MORPHALIGN_LANGUAGE_CONFIGS,
    morphalign_score,
)
from paper_utils.hybrid.train_hybrid import get_bpe_model_path, get_model_path as get_hybrid_model_path
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import FSP_OVERRIDES, FSP_PARAMS, morphalign_paper_score, paper_table_path
from paper_utils.unigram.train_hyperparameters import (
    ADDITIONAL_VOCAB_SIZE,
    DEFAULTS,
    get_model_path as get_unigram_model_path,
)
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.convextok import ConvexTokModel
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.pathpiece import PathPieceModel
from script_bpe.tokenizers.unigram import UnigramModel

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
GRID_JSON = RESULTS_DIR / "compression_train_eval_grid.json"
MORPHALIGN_JSON = RESULTS_DIR / "cache_morphalign_scatter.json"
OUT_TEX = paper_table_path("table_train_corpus.tex", appendix=True, extra=True)
CONVEXTOK_PATH = "results/convextok_tokenizers/{train}/n32768_cmin50_mp200000_L32_det.json.gz"

MAIN_F = 1.15
PLOT_EM = 2
PLOT_P = 0.0
MINGRAM_PP_F = 8.0
MINGRAM_PP_P = 0.9

PAIRINGS = [
    ("fineweb->fineweb", "\\shortstack[l]{Single-language FineWeb\\\\eval FineWeb}"),
    ("fineweb->fishfood", "\\shortstack[l]{Single-language FineWeb\\\\eval Goldfish}"),
    ("fineweb_h6->fineweb", "\\shortstack[l]{Six-language FineWeb\\\\eval FineWeb}"),
    ("fineweb_h6->fishfood", "\\shortstack[l]{Six-language FineWeb\\\\eval Goldfish}"),
]

MORPHALIGN_TRAIN_SETUPS = [
    ("fineweb->fishfood", "\\shortstack[l]{Single-language\\\\FineWeb}"),
    ("fineweb_h6->fishfood", "\\shortstack[l]{Six-language\\\\FineWeb}"),
]

COMPRESSION_METHOD_ORDER = [
    "bpe",
    "fsp",
    "bpe_init",
    "bpe_init_fsp",
    "mingram",
    "mingram_pp",
    "pathpiece_bpe",
    "convextok",
]
COMPRESSION_METHOD_LABEL = {
    "bpe": "BPE",
    "fsp": "FSP",
    "bpe_init": "\\shortstack{Unigram-\\\\BPE-Init}",
    "bpe_init_fsp": "\\shortstack{FSP-\\\\BPE-Init}",
    "mingram": "MinGram",
    "mingram_pp": "\\mingrampptab{}",
    "pathpiece_bpe": "\\shortstack{PathPiece-\\\\BPE}",
    "convextok": "ConvexTok",
}

MORPHALIGN_METHOD_ORDER = [
    "bpe",
    "default",
    "fsp",
    "bpe_init",
    "bpe_init_fsp",
    "mingram",
    "mingram_pp",
    "pathpiece_bpe",
    "convextok",
]
MORPHALIGN_METHOD_LABEL = {
    "bpe": "BPE",
    "default": "Unigram",
    "fsp": "FSP",
    "bpe_init": "Unigram\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "bpe_init_fsp": "FSP\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "mingram": "MinGram",
    "mingram_pp": "\\mingrampp{}",
    "pathpiece_bpe": "PathPiece\\hspace{0pt}-BPE",
    "convextok": "ConvexTok",
}

LANG_ORDER = ["eng", "deu", "fin", "rus", "arb", "kor"]
LANG_LABEL = {
    "eng": "English",
    "deu": "German",
    "fin": "Finnish",
    "rus": "Russian",
    "arb": "Arabic",
    "kor": "Korean",
}

MORPHALIGN_CFG_BY_LANG = {cfg["lang"]: cfg for cfg in MORPHALIGN_LANGUAGE_CONFIGS}


def _fmt_comp(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{-value:+.2f}\\%"


def _fmt_morph(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{morphalign_paper_score(value):.2f}"


def _morphalign_cache_key(train_corpus: str, lang: str, method: str, model_path: Path) -> str:
    if method == "mingram_pp":
        return f"train_corpus/{train_corpus}/{lang}/mingram_pp/{model_path.name}"
    if method == "mingram":
        return f"{lang}/mingram/{model_path.name}"
    if method == "pathpiece_bpe":
        return f"{lang}/pathpiece_bpe/{model_path.name}"
    if method == "convextok":
        return f"{lang}/convextok/{model_path.name}"
    return f"train_corpus/{train_corpus}/{lang}/{method}/{model_path.name}"


def _model_path(train_corpus: str, method: str) -> Path:
    if method == "bpe":
        return get_bpe_model_path(train_corpus, ADDITIONAL_VOCAB_SIZE)
    if method == "default":
        return get_unigram_model_path(train_corpus, DEFAULTS)
    if method == "fsp":
        return get_unigram_model_path(train_corpus, FSP_PARAMS)
    if method == "bpe_init":
        return get_hybrid_model_path(
            train_corpus,
            {**DEFAULTS, "overshoot_factor": MAIN_F},
        )
    if method == "bpe_init_fsp":
        return get_hybrid_model_path(
            train_corpus,
            {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": MAIN_F},
        )
    if method == "mingram":
        return get_mingram_model_path(train_corpus, MAIN_F, PLOT_EM, PLOT_P)
    if method == "mingram_pp":
        return get_mingram_model_path(
            train_corpus,
            MINGRAM_PP_F,
            PLOT_EM,
            MINGRAM_PP_P,
            prune_criterion="mi",
        )
    if method == "pathpiece_bpe":
        return get_pathpiece_model_path(train_corpus, init="bpe")
    if method == "convextok":
        return Path(CONVEXTOK_PATH.format(train=train_corpus))
    raise ValueError(f"Unknown method: {method}")


def _load_model(train_corpus: str, method: str):
    model_path = _model_path(train_corpus, method)
    if method == "bpe":
        return BPETokenizer.load(str(model_path)), model_path
    if method in {"default", "fsp", "bpe_init", "bpe_init_fsp"}:
        return UnigramModel.load(str(model_path)), model_path
    if method in {"mingram", "mingram_pp"}:
        return MinGramModel.load(str(model_path)), model_path
    if method == "pathpiece_bpe":
        return PathPieceModel.load(str(model_path)), model_path
    if method == "convextok":
        return ConvexTokModel.load(str(model_path)), model_path
    raise ValueError(f"Unknown method: {method}")


def _lookup_or_compute_morphalign(morph_cache: dict, train_corpus: str, lang: str, method: str) -> float | None:
    cfg = MORPHALIGN_CFG_BY_LANG[lang]
    model_path = _model_path(train_corpus, method)
    cache_key = _morphalign_cache_key(train_corpus, lang, method, model_path)
    if cache_key in morph_cache:
        return float(morph_cache[cache_key])
    if not model_path.exists():
        return None  # model not built yet (e.g. hybrid6 cells pending) -> render as "--"
    model, _model_path_loaded = _load_model(train_corpus, method)
    return morphalign_score(model, cfg["gold_file"], morph_cache, cache_key)


def _morphalign_row(morph_cache: dict, train_corpus: str, lang: str) -> dict[str, float | None]:
    if lang not in MORPHALIGN_CFG_BY_LANG:
        return {method: None for method in MORPHALIGN_METHOD_ORDER}
    return {
        method: _lookup_or_compute_morphalign(morph_cache, train_corpus, lang, method)
        for method in MORPHALIGN_METHOD_ORDER
    }


def _available_pairings(grid: dict) -> list[tuple[str, str]]:
    return [
        (pair_key, label)
        for pair_key, label in PAIRINGS
        if any(lang in grid.get(pair_key, {}).get("series", {}) for lang in LANG_ORDER)
    ]


def build_table(grid: dict, morph_cache: dict) -> str:
    pairings = _available_pairings(grid)
    lines = [
        "% Requires \\usepackage{booktabs} for \\toprule, \\midrule, \\bottomrule.",
        "% Requires \\usepackage{multirow} for \\multirow.",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llrrrrrrrr@{}}",
        "\\toprule",
        "\\multicolumn{10}{@{}l}{\\textit{Compression improvement over Unigram (\\%, higher is better)}} \\\\",
        " & "
        .join(
            [
                "\\shortstack[l]{Training/eval\\\\condition}",
                "Language",
                *[COMPRESSION_METHOD_LABEL[method] for method in COMPRESSION_METHOD_ORDER],
            ]
        )
        + " \\\\",
        "\\midrule",
    ]

    for pair_idx, (pair_key, pair_label) in enumerate(pairings):
        series = grid[pair_key]["series"]
        for row_idx, lang in enumerate(LANG_ORDER):
            row = series[lang]

            cells: list[str] = []
            if row_idx == 0:
                cells.append(f"\\multirow{{{len(LANG_ORDER)}}}{{*}}{{{pair_label}}}")
            else:
                cells.append("")
            cells.append(LANG_LABEL[lang])
            cells.extend(
                _fmt_comp(row.get(method) if row.get(method) is None else float(row[method]))
                for method in COMPRESSION_METHOD_ORDER
            )
            lines.append(" & ".join(cells) + " \\\\")
        if pair_idx < len(pairings) - 1:
            lines.append("\\midrule")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{0.55em}",
        "\\begin{tabular}{@{}llrrr@{}}",
        "\\toprule",
        "\\multicolumn{5}{@{}l}{\\textit{MorphAlign Score}} \\\\",
        "Condition & Method & English & German & Finnish \\\\",
        "\\midrule",
    ]

    morph_langs = ["eng", "deu", "fin"]
    morph_setups = [(key, label) for key, label in MORPHALIGN_TRAIN_SETUPS if key in grid and grid[key]["series"]]
    for pair_idx, (pair_key, pair_label) in enumerate(morph_setups):
        series = grid[pair_key]["series"]
        for method_idx, method in enumerate(MORPHALIGN_METHOD_ORDER):
            cells = []
            if method_idx == 0:
                cells.append(f"\\multirow{{{len(MORPHALIGN_METHOD_ORDER)}}}{{*}}{{{pair_label}}}")
            else:
                cells.append("")
            cells.append(MORPHALIGN_METHOD_LABEL[method])
            for lang in morph_langs:
                train_corpus = series[lang]["train_corpus"]
                morph_row = _morphalign_row(morph_cache, train_corpus, lang)
                cells.append(_fmt_morph(morph_row[method]))
            lines.append(" & ".join(cells) + " \\\\")
        if pair_idx < len(morph_setups) - 1:
            lines.append("\\midrule")

    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    grid = json.loads(GRID_JSON.read_text())
    morph_cache = json.loads(MORPHALIGN_JSON.read_text())
    tex = build_table(grid, morph_cache)

    MORPHALIGN_JSON.write_text(json.dumps(morph_cache, indent=2))
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)

    print(f"Wrote {OUT_TEX}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
