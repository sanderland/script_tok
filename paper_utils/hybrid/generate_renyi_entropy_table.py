#!/usr/bin/env python3
"""Generate the paper Rényi-efficiency table.

The table reports normalized Rényi-3 entropy, matching Cognetta et al. (2024):

    H_3(token distribution) / log2(nonzero vocabulary)

This script evaluates the current MinGram-paper tokenizer set, then writes the
paper table under ``results/mingram_paper/tables/``. By default it generates the
held-out panel used in the paper; pass ``--settings heldout train`` to also
compute the full training-corpus panel.
"""

import argparse
import json
import math
from pathlib import Path

from paper_utils.hybrid.train_hybrid import get_bpe_model_path, get_model_path as get_hybrid_model_path
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import FSP_OVERRIDES, FSP_PARAMS, paper_table_path
from paper_utils.unigram.train_hyperparameters import ADDITIONAL_VOCAB_SIZE, DEFAULTS, get_model_path as get_unigram_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers import load_tokenizer

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
CACHE_JSON = RESULTS_DIR / "cache_renyi_entropy.json"
OUT_TEX = paper_table_path("table_renyi_entropy.tex")

MAIN_F = 1.15
PLOT_EM = 2
PLOT_P = 0.0
RENYI_ALPHA = 3.0
RENYI_KEY = "3.0"
CONVEXTOK_PATH = "results/convextok_tokenizers/{train}/n32768_cmin50_mp200000_L32_det.json.gz"

LANG_ORDER = ["eng", "deu", "fin", "rus", "arb", "kor"]
LANG_LABEL = {
    "eng": "English",
    "deu": "German",
    "fin": "Finnish",
    "rus": "Russian",
    "arb": "Arabic",
    "kor": "Korean",
}
TRAIN_EVAL_BY_LANG = {
    "eng": ("fineweb_en_5gb", "eng_latn_fishfood"),
    "deu": ("fineweb_de_5gb", "deu_latn_fishfood"),
    "fin": ("fineweb_fi_5gb", "fin_latn_fishfood"),
    "rus": ("fineweb_ru_5gb", "rus_cyrl_fishfood"),
    "arb": ("fineweb_ar_5gb", "arb_arab_fishfood"),
    "kor": ("fineweb_ko_5gb", "kor_hang_fishfood"),
}
EVAL_SETTINGS = [
    ("heldout", "Held-out Goldfish Data"),
    ("train", "Training corpus"),
]
EVAL_SETTING_LABEL = dict(EVAL_SETTINGS)

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
METHOD_LABEL = {
    "bpe": "BPE",
    "unigram": "Unigram",
    "fsp": "FSP",
    "bpe_init": "Unigram\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "fsp_bpe_init": "FSP\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "mingram": "MinGram",
    "mingram_pp": "\\mingrampp{}",
    "pathpiece_bpe": "PathPiece\\hspace{0pt}-BPE",
    "convextok": "ConvexTok",
}


def _model_path(train_corpus: str, method: str) -> Path:
    if method == "bpe":
        return get_bpe_model_path(train_corpus, ADDITIONAL_VOCAB_SIZE)
    if method == "unigram":
        return get_unigram_model_path(train_corpus, DEFAULTS)
    if method == "fsp":
        return get_unigram_model_path(train_corpus, FSP_PARAMS)
    if method == "bpe_init":
        return get_hybrid_model_path(train_corpus, {**DEFAULTS, "overshoot_factor": MAIN_F})
    if method == "fsp_bpe_init":
        return get_hybrid_model_path(train_corpus, {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": MAIN_F})
    if method == "mingram":
        return get_mingram_model_path(train_corpus, MAIN_F, PLOT_EM, PLOT_P)
    if method == "mingram_pp":
        # MinGram-PP candidate = f=8 (its compression optimum; f=1.15 is only stock MinGram's)
        return get_mingram_model_path(train_corpus, 8.0, PLOT_EM, 0.9, prune_criterion="mi")
    if method == "pathpiece_bpe":
        return get_pathpiece_model_path(train_corpus, init="bpe")
    if method == "convextok":
        return Path(CONVEXTOK_PATH.format(train=train_corpus))
    raise ValueError(f"Unknown method: {method}")


def _renyi_efficiency(renyi_bits: float, nonzero_vocab: int) -> float:
    if nonzero_vocab <= 1:
        return 0.0
    return renyi_bits / math.log2(nonzero_vocab)


def _load_cache() -> dict:
    if CACHE_JSON.exists():
        return json.loads(CACHE_JSON.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CACHE_JSON.write_text(json.dumps(cache, indent=2))


def _eval_method(train_corpus: str, eval_corpus_name: str, method: str, cache: dict) -> dict:
    model_path = _model_path(train_corpus, method)
    cache_key = f"alpha{RENYI_KEY}/{train_corpus}/{eval_corpus_name}/{method}/{model_path.name}"
    if cache_key in cache:
        return cache[cache_key]

    model = load_tokenizer(str(model_path))
    eval_corpus = load_corpus_by_name(eval_corpus_name, model.pretokenizer)
    perf = model.corpus_performance(eval_corpus, alphas=(1.0, RENYI_ALPHA))
    renyi_bits = float(perf["renyi_bits"][RENYI_KEY])
    nonzero_vocab = int(perf["nonzero_vocab"])
    row = {
        "renyi_bits": renyi_bits,
        "renyi_eff": _renyi_efficiency(renyi_bits, nonzero_vocab),
        "shannon_bits": float(perf["shannon_bits"]),
        "nonzero_vocab": nonzero_vocab,
        "bytes_per_token": float(perf["bytes_per_token"]),
    }
    cache[cache_key] = row
    _save_cache(cache)
    print(f"[cache] {method} {train_corpus}->{eval_corpus_name}: H3eff={row['renyi_eff']:.3f}", flush=True)
    return row


def _eval_corpus_for_setting(lang: str, setting: str) -> tuple[str, str]:
    train_corpus, heldout_corpus = TRAIN_EVAL_BY_LANG[lang]
    if setting == "heldout":
        return train_corpus, heldout_corpus
    if setting == "train":
        return train_corpus, train_corpus
    raise ValueError(f"Unknown evaluation setting: {setting}")


def collect_rows(cache: dict, setting: str) -> list[dict]:
    rows = []
    for method in METHOD_ORDER:
        values: dict[str, float] = {}
        bits: dict[str, float] = {}
        for lang in LANG_ORDER:
            train_corpus, eval_corpus_name = _eval_corpus_for_setting(lang, setting)
            perf = _eval_method(train_corpus, eval_corpus_name, method, cache)
            values[lang] = float(perf["renyi_eff"])
            bits[lang] = float(perf["renyi_bits"])
        rows.append(
            {
                "method": method,
                "values": values,
                "bits": bits,
                "mean": sum(values.values()) / len(values),
            }
        )
    return sorted(rows, key=lambda row: row["mean"], reverse=True)


def _top2_by_column(rows: list[dict]) -> dict[str, tuple[str | None, str | None]]:
    out = {}
    for lang in [*LANG_ORDER, "mean"]:
        if lang == "mean":
            pairs = [(row["method"], row["mean"]) for row in rows]
        else:
            pairs = [(row["method"], row["values"][lang]) for row in rows]
        ordered = sorted(pairs, key=lambda pair: pair[1], reverse=True)
        out[lang] = (ordered[0][0], ordered[1][0] if len(ordered) > 1 else None)
    return out


def _fmt(value: float, style: str | None = None) -> str:
    text = f"{value:.3f}"
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\underline{{{text}}}"
    return text


def _build_panel(title: str, rows: list[dict]) -> list[str]:
    top2 = _top2_by_column(rows)
    lines = [
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}: R\\'enyi-3 efficiency $H_3 / \\log_2 |V_{{\\mathrm{{used}}}}|$}}}} \\\\",
        "Method & " + " & ".join(LANG_LABEL[lang] for lang in LANG_ORDER) + " & Mean \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = row["method"]
        cells = [METHOD_LABEL[method]]
        for lang in LANG_ORDER:
            best, second = top2[lang]
            style = "best" if best == method else "second" if second == method else None
            cells.append(_fmt(row["values"][lang], style))
        best, second = top2["mean"]
        style = "best" if best == method else "second" if second == method else None
        cells.append(_fmt(row["mean"], style))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return lines


def build_table(rows_by_setting: dict[str, list[dict]], settings: list[str]) -> str:
    lines = [
        "% Extra appendix table generated by paper_utils/hybrid/generate_renyi_entropy_table.py.",
        "% Requires \\usepackage{booktabs}.",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    for i, setting in enumerate(settings):
        if i:
            lines += ["", "\\vspace{0.75em}", ""]
        lines += _build_panel(EVAL_SETTING_LABEL[setting], rows_by_setting[setting])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=[setting for setting, _ in EVAL_SETTINGS],
        default=["heldout"],
        help="Evaluation panels to compute.",
    )
    args = parser.parse_args()

    cache = _load_cache()
    rows_by_setting = {setting: collect_rows(cache, setting) for setting in args.settings}
    _save_cache(cache)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(build_table(rows_by_setting, args.settings))
    print(f"Wrote {OUT_TEX}")
    print()
    print(OUT_TEX.read_text())


if __name__ == "__main__":
    main()
