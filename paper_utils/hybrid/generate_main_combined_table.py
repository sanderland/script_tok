#!/usr/bin/env python3
"""Generate the main results tables: compression + MorphAlign.

Writes two tabular bodies:

  - `results/mingram_paper/tables/table_main_compression.tex` — full-width
    (six languages + mean), intended to be wrapped in `table*`.
  - `results/mingram_paper/tables/table_main_morphalign.tex` — single-column
    (three UniMorph languages + mean), intended to be wrapped
    in `table`.

Conventions:
  - Unigram is omitted from the compression table (it is the baseline).
  - Compression cells are bolded if best-in-column (most negative value),
    with second-best underlined.
  - MorphAlign cells are bolded if best-in-column (highest value), with
    second-best underlined.
"""

import json
from pathlib import Path

from paper_utils.hybrid.train_hybrid import get_model_path as get_hybrid_model_path
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import FSP_OVERRIDES, geomean, morphalign_paper_score, paper_table_path
from paper_utils.unigram.train_hyperparameters import DEFAULTS

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
GRID_JSON = RESULTS_DIR / "compression_train_eval_grid.json"
MORPHALIGN_JSON = RESULTS_DIR / "cache_morphalign_scatter.json"
OUT_COMP_TEX = paper_table_path("table_main_compression.tex")
OUT_MORPH_TEX = paper_table_path("table_main_morphalign.tex")

MAIN_F = 1.15
PLOT_EM = 2
PLOT_P = 0.0
MINGRAM_PP_F = 8.0
MINGRAM_PP_P = 0.9
CONVEXTOK_MODEL = "results/convextok_tokenizers/{train}/n32768_cmin50_mp200000_L32_det.json.gz"

METHOD_ORDER = [
    "bpe",
    "unigram",
    "fsp",
    "bpe_init",
    "fsp_bpe_init",
    "mingram",
    "mingram_pp",
    "pathpiece_bpe",
    "convextok",
]
COMPRESSION_METHOD_ORDER = [method for method in METHOD_ORDER if method != "unigram"]
METHOD_LABEL = {
    "bpe": "BPE",
    "unigram": "Unigram",
    "fsp": "FSP",
    "bpe_init": "Unigram\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "fsp_bpe_init": "FSP\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "mingram": "MinGram",
    "mingram_pp": "\\mingrampp{}",
    "pathpiece_ngram": "PathPiece\\hspace{0pt}-N\\hspace{0pt}-gram",
    "pathpiece_bpe": "PathPiece\\hspace{0pt}-BPE",
    "convextok": "ConvexTok",
}

GRID_KEY = {
    "bpe": "bpe",
    "unigram": "default",
    "fsp": "fsp",
    "bpe_init": "bpe_init",
    "fsp_bpe_init": "bpe_init_fsp",
    "mingram": "mingram",
    "mingram_pp": "mingram_pp",
    "pathpiece_ngram": "pathpiece_ngram",
    "pathpiece_bpe": "pathpiece_bpe",
    "convextok": "convextok",
}

COMPRESSION_LANGS = ["eng", "deu", "fin", "rus", "arb", "kor"]
MORPHALIGN_LANGS = ["eng", "deu", "fin"]

LANG_LABEL = {
    "eng": "English",
    "deu": "German",
    "fin": "Finnish",
    "rus": "Russian",
    "arb": "Arabic",
    "kor": "Korean",
}

TRAIN_CORPUS_BY_LANG = {
    "eng": "fineweb_en_5gb",
    "deu": "fineweb_de_5gb",
    "fin": "fineweb_fi_5gb",
    "rus": "fineweb_ru_5gb",
    "arb": "fineweb_ar_5gb",
    "kor": "fineweb_ko_5gb",
}

def _compression_row(grid: dict, method: str) -> dict[str, float | None]:
    series = grid["fineweb->fishfood"]["series"]
    out: dict[str, float | None] = {}
    for lang in COMPRESSION_LANGS:
        row = series.get(lang)
        if row is None:
            out[lang] = None
            continue
        value = row.get(GRID_KEY[method])
        out[lang] = None if value is None else float(value)
    return out


def _morphalign_reference(cache: dict, lang: str, ref_name: str) -> float | None:
    key = f"{lang}/ref/{ref_name}"
    if key not in cache:
        return None
    return float(cache[key])


def _language_cfg(lang: str) -> dict:
    from paper_utils.hybrid.generate_morphalign_scatter import LANGUAGE_CONFIGS

    return next(cfg for cfg in LANGUAGE_CONFIGS if cfg["lang"] == lang)


def _compute_unigram_family_morphalign(cache: dict, lang: str, model_path: Path, cache_key: str) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import morphalign_score
    from script_bpe.tokenizers.unigram import UnigramModel

    if not model_path.exists():
        return None
    cfg = _language_cfg(lang)
    model = UnigramModel.load(str(model_path))
    return morphalign_score(model, cfg["gold_file"], cache, cache_key)


def _compute_mingram_morphalign(cache: dict, lang: str, model_path: Path, cache_key: str) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import morphalign_score
    from script_bpe.tokenizers.mingram.model import MinGramModel

    if not model_path.exists():
        return None
    cfg = _language_cfg(lang)
    model = MinGramModel.load(str(model_path))
    return morphalign_score(model, cfg["gold_file"], cache, cache_key)


def _compute_pathpiece_morphalign(cache: dict, lang: str, model_path: Path, cache_key: str) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import morphalign_score
    from script_bpe.tokenizers.pathpiece import PathPieceModel

    if not model_path.exists():
        return None
    cfg = _language_cfg(lang)
    model = PathPieceModel.load(str(model_path))
    return morphalign_score(model, cfg["gold_file"], cache, cache_key)


def _compute_convextok_morphalign(cache: dict, lang: str, model_path: Path, cache_key: str) -> float | None:
    from paper_utils.hybrid.generate_morphalign_scatter import morphalign_score
    from script_bpe.tokenizers import load_tokenizer

    if not model_path.exists():
        return None
    cfg = _language_cfg(lang)
    model = load_tokenizer(str(model_path))
    return morphalign_score(model, cfg["gold_file"], cache, cache_key)


def _morphalign_pathpiece(cache: dict, lang: str, init: str) -> float | None:
    train_corpus = TRAIN_CORPUS_BY_LANG[lang]
    model_path = get_pathpiece_model_path(train_corpus, init=init)
    cache_key = f"{lang}/pathpiece_{init}/{model_path.name}"
    if cache_key in cache:
        return float(cache[cache_key])
    return _compute_pathpiece_morphalign(cache, lang, model_path, cache_key)


def _morphalign_convextok(cache: dict, lang: str) -> float | None:
    train_corpus = TRAIN_CORPUS_BY_LANG[lang]
    model_path = Path(CONVEXTOK_MODEL.format(train=train_corpus))
    cache_key = f"{lang}/convextok/{model_path.name}"
    if cache_key in cache:
        return float(cache[cache_key])
    return _compute_convextok_morphalign(cache, lang, model_path, cache_key)


def _morphalign_fsp_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    train_corpus = TRAIN_CORPUS_BY_LANG[lang]
    model_path = get_hybrid_model_path(
        train_corpus,
        {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": f},
    )
    cache_key = f"{lang}/bpe_init_fsp/f{f}/{model_path.name}"
    if cache_key in cache:
        return float(cache[cache_key])
    return _compute_unigram_family_morphalign(cache, lang, model_path, cache_key)


def _morphalign_bpe_init(cache: dict, lang: str, f: float) -> float | None:
    train_corpus = TRAIN_CORPUS_BY_LANG[lang]
    model_path = get_hybrid_model_path(
        train_corpus,
        {**DEFAULTS, "overshoot_factor": f},
    )
    cache_key = f"{lang}/bpe_init/f{f}/{model_path.name}"
    if cache_key in cache:
        return float(cache[cache_key])
    return _compute_unigram_family_morphalign(cache, lang, model_path, cache_key)


def _morphalign_mingram(
    cache: dict,
    lang: str,
    f: float,
    em: int,
    p: float,
    prune_criterion: str = "usage_count",
) -> float | None:
    train_corpus = TRAIN_CORPUS_BY_LANG[lang]
    model_path = get_mingram_model_path(train_corpus, f, em, p, prune_criterion=prune_criterion)
    cache_key = f"{lang}/mingram/{model_path.name}"
    if cache_key in cache:
        return float(cache[cache_key])
    return _compute_mingram_morphalign(cache, lang, model_path, cache_key)


def _morphalign_row(cache: dict, method: str) -> dict[str, float | None]:
    ref_name = {"bpe": "BPE", "unigram": "Default", "fsp": "FSP"}.get(method)
    out: dict[str, float | None] = {}
    for lang in MORPHALIGN_LANGS:
        if ref_name is not None:
            out[lang] = _morphalign_reference(cache, lang, ref_name)
        elif method == "bpe_init":
            out[lang] = _morphalign_bpe_init(cache, lang, MAIN_F)
        elif method == "fsp_bpe_init":
            out[lang] = _morphalign_fsp_bpe_init(cache, lang, MAIN_F)
        elif method == "mingram":
            out[lang] = _morphalign_mingram(cache, lang, MAIN_F, PLOT_EM, PLOT_P)
        elif method == "mingram_pp":
            out[lang] = _morphalign_mingram(
                cache,
                lang,
                MINGRAM_PP_F,
                PLOT_EM,
                MINGRAM_PP_P,
                prune_criterion="mi",
            )
        elif method == "pathpiece_ngram":
            out[lang] = _morphalign_pathpiece(cache, lang, "ngram")
        elif method == "pathpiece_bpe":
            out[lang] = _morphalign_pathpiece(cache, lang, "bpe")
        elif method == "convextok":
            out[lang] = _morphalign_convextok(cache, lang)
        else:
            out[lang] = None
    return out


def _mean_or_none(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    xs = [value for value in values if value is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _geomean_or_none(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    xs = [value for value in values if value is not None]
    if not xs:
        return None
    return geomean(xs)


def _fmt_comp(value: float | None, style: str | None = None) -> str:
    if value is None:
        return "--"
    text = f"{value:+.2f}\\%"
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\underline{{{text}}}"
    return text


def _fmt_morph(value: float | None, style: str | None = None) -> str:
    if value is None:
        return "--"
    text = f"{morphalign_paper_score(value):.2f}"
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\underline{{{text}}}"
    return text


def _column_top2_compression(values: dict[str, float | None]) -> tuple[str | None, str | None]:
    pairs = [(method, value) for method, value in values.items() if value is not None]
    if not pairs:
        return None, None
    ordered = sorted(pairs, key=lambda pair: pair[1])
    best = ordered[0][0]
    second = ordered[1][0] if len(ordered) > 1 else None
    return best, second


def _column_top2_morph(values: dict[str, float | None]) -> tuple[str | None, str | None]:
    pairs = [(method, value) for method, value in values.items() if value is not None]
    if not pairs:
        return None, None
    ordered = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    best = ordered[0][0]
    second = ordered[1][0] if len(ordered) > 1 else None
    return best, second


def _build_compression(comp_cells, comp_means, comp_top2_by_lang, comp_top2_mean) -> str:
    comp_col_spec = "l" + "r" * len(COMPRESSION_LANGS) + "r"
    method_order = sorted(
        COMPRESSION_METHOD_ORDER,
        key=lambda method: comp_means[method] if comp_means[method] is not None else float("inf"),
    )
    lines = [
        "% Intended to be wrapped in \\begin{table*}...\\end{table*}",
        "% Requires \\usepackage{booktabs}.",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{" + comp_col_spec + "}",
        "\\toprule",
        "Method & " + " & ".join(LANG_LABEL[lang] for lang in COMPRESSION_LANGS) + " & Mean \\\\",
        "\\midrule",
    ]
    for method in method_order:
        row = [METHOD_LABEL[method]]
        for lang in COMPRESSION_LANGS:
            best, second = comp_top2_by_lang[lang]
            style = "best" if best == method else "second" if second == method else None
            row.append(_fmt_comp(comp_cells[method][lang], style))
        best_mean, second_mean = comp_top2_mean
        mean_style = "best" if best_mean == method else "second" if second_mean == method else None
        row.append(_fmt_comp(comp_means[method], mean_style))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def _build_morphalign(morph_cells, morph_means, morph_top2_by_lang, morph_top2_mean) -> str:
    morph_col_spec = "l" + "r" * len(MORPHALIGN_LANGS) + "r"
    method_order = sorted(
        METHOD_ORDER,
        key=lambda method: morph_means[method] if morph_means[method] is not None else float("-inf"),
        reverse=True,
    )
    lines = [
        "% Intended to be wrapped in \\begin{table}...\\end{table}",
        "% Requires \\usepackage{booktabs}.",
        "%\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{" + morph_col_spec + "}",
        "\\toprule",
        "Method & " + " & ".join(LANG_LABEL[lang] for lang in MORPHALIGN_LANGS) + " & G. Mean \\\\",
        "\\midrule",
    ]
    for method in method_order:
        row = [METHOD_LABEL[method]]
        for lang in MORPHALIGN_LANGS:
            best, second = morph_top2_by_lang[lang]
            style = "best" if best == method else "second" if second == method else None
            row.append(_fmt_morph(morph_cells[method][lang], style))
        best_mean, second_mean = morph_top2_mean
        mean_style = "best" if best_mean == method else "second" if second_mean == method else None
        row.append(_fmt_morph(morph_means[method], mean_style))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def build_tables(grid: dict, morph_cache: dict) -> tuple[str, str]:
    comp_cells: dict[str, dict[str, float | None]] = {}
    morph_cells: dict[str, dict[str, float | None]] = {}
    comp_means: dict[str, float | None] = {}
    morph_means: dict[str, float | None] = {}

    for method in METHOD_ORDER:
        comp_row = _compression_row(grid, method)
        morph_row = _morphalign_row(morph_cache, method)
        if method == "unigram":
            comp_row = {lang: None for lang in COMPRESSION_LANGS}
        comp_cells[method] = comp_row
        morph_cells[method] = morph_row
        comp_means[method] = _mean_or_none(list(comp_row.values()))
        morph_means[method] = _geomean_or_none(list(morph_row.values()))

    comp_top2_by_lang = {
        lang: _column_top2_compression({method: comp_cells[method][lang] for method in COMPRESSION_METHOD_ORDER})
        for lang in COMPRESSION_LANGS
    }
    comp_top2_mean = _column_top2_compression({method: comp_means[method] for method in COMPRESSION_METHOD_ORDER})
    morph_top2_by_lang = {
        lang: _column_top2_morph({method: morph_cells[method][lang] for method in METHOD_ORDER})
        for lang in MORPHALIGN_LANGS
    }
    morph_top2_mean = _column_top2_morph(morph_means)

    comp_tex = _build_compression(comp_cells, comp_means, comp_top2_by_lang, comp_top2_mean)
    morph_tex = _build_morphalign(morph_cells, morph_means, morph_top2_by_lang, morph_top2_mean)
    return comp_tex, morph_tex


def main() -> None:
    grid = json.loads(GRID_JSON.read_text())
    morph_cache = json.loads(MORPHALIGN_JSON.read_text())
    comp_tex, morph_tex = build_tables(grid, morph_cache)
    MORPHALIGN_JSON.write_text(json.dumps(morph_cache, indent=2))
    OUT_COMP_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_COMP_TEX.write_text(comp_tex)
    OUT_MORPH_TEX.write_text(morph_tex)
    print(f"Wrote {OUT_COMP_TEX}")
    print(comp_tex)
    print()
    print(f"Wrote {OUT_MORPH_TEX}")
    print(morph_tex)

    missing = []
    for method in METHOD_ORDER:
        if method != "unigram":
            comp_row = _compression_row(grid, method)
            for lang, value in comp_row.items():
                if value is None:
                    missing.append((method, "compression", lang))
        morph_row = _morphalign_row(morph_cache, method)
        for lang, value in morph_row.items():
            if value is None:
                missing.append((method, "morphalign", lang))

    if missing:
        print()
        print("WARNING: missing cells (method, axis, lang):")
        for item in missing:
            print(f"  {item}")


if __name__ == "__main__":
    main()
