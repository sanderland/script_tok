#!/usr/bin/env python3
"""Generate the appendix pruning-schedule table.

Reports mean MorphAlign and compression for MinGram at fixed EM=2 across
overshoot factors. Single-step pruning (`p=0.0`) is shown as the reference and
iterative pruning (`p=0.9`) is shown as a delta from that reference.
"""

from pathlib import Path

from paper_utils.hybrid.generate_morphalign_scatter import (
    LANGUAGE_CONFIGS as MORPHALIGN_LANGUAGE_CONFIGS,
    morphalign_score,
)
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.utils import geomean, load_cache, morphalign_paper_score, paper_table_path, save_cache
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as get_unigram_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.unigram import UnigramModel

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
MORPHALIGN_JSON = RESULTS_DIR / "cache_morphalign_scatter.json"
COMPRESSION_CACHE_JSON = RESULTS_DIR / "cache_train_eval_compression_grid.json"
OUT_TEX = paper_table_path("table_pruning_schedule.tex", appendix=True)

EM = 2
F_VALUES = [1.1, 1.15, 1.25, 1.5, 2.0, 3.0, 5.0]
P_VALUES = [0.0, 0.9]
P_LABEL = {
    0.0: "single-step",
    0.9: "iterative",
}

LANGUAGE_CONFIGS = [
    cfg for cfg in MORPHALIGN_LANGUAGE_CONFIGS if cfg["lang"] in {"eng", "deu", "fin"}
]
LANG_ORDER = [cfg["lang"] for cfg in LANGUAGE_CONFIGS]
LANG_LABEL = {
    "eng": "English",
    "deu": "German",
    "fin": "Finnish",
}


def _compression_cache_key(train_corpus: str, eval_corpus: str, method: str, model_path: Path) -> str:
    return f"tokens/{train_corpus}/{eval_corpus}/{method}/{model_path.name}"


def _token_count(model, eval_corpus_name: str, compression_cache: dict, cache_key: str) -> int:
    if cache_key in compression_cache:
        return int(compression_cache[cache_key])
    eval_corpus = load_corpus_by_name(eval_corpus_name, model.pretokenizer)
    total_tokens = int(model.corpus_performance(eval_corpus)["total_tokens_len"])
    compression_cache[cache_key] = total_tokens
    return total_tokens


def _mean_or_none(values: dict[str, float | None]) -> float | None:
    xs = [value for value in values.values() if value is not None]
    if len(xs) != len(values):
        return None
    return sum(xs) / len(xs)


def _gmean(values: dict[str, float]) -> float:
    return geomean(values.values())


def _fmt_f(value: float) -> str:
    return f"{value:g}"


def _fmt_morph(value: float) -> str:
    if value == 0.0:
        return "$0$"
    mantissa, exponent = f"{morphalign_paper_score(value):.1e}".split("e")
    return rf"${mantissa} \cdot 10^{{{int(exponent)}}}$"


def _fmt_morph_decimal(value: float) -> str:
    return f"{morphalign_paper_score(value):.2f}"


def _fmt_morph_delta(value: float) -> str:
    value = morphalign_paper_score(value)
    if abs(value) < 0.005:
        value = 0.0
    return f"{value:+.2f}"


def _fmt_pp_delta(value: float) -> str:
    return f"{value:+.2f}"


def _fmt_comp_mean(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:+.2f}\\%"


def _fmt_comp_delta(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{_fmt_pp_delta(value)} pp"


def collect_rows(morphalign_cache: dict, compression_cache: dict) -> list[dict]:
    baseline_tokens: dict[str, int] = {}
    for cfg in LANGUAGE_CONFIGS:
        default_path = get_unigram_model_path(cfg["train_corpus"], DEFAULTS)
        cache_key = _compression_cache_key(cfg["train_corpus"], cfg["eval_corpus"], "default", default_path)
        if cache_key in compression_cache:
            baseline_tokens[cfg["lang"]] = int(compression_cache[cache_key])
        else:
            default_model = UnigramModel.load(str(default_path))
            baseline_tokens[cfg["lang"]] = _token_count(
                default_model,
                cfg["eval_corpus"],
                compression_cache,
                cache_key,
            )

    rows = []
    for f in F_VALUES:
        for p in P_VALUES:
            morph_row: dict[str, float] = {}
            comp_row: dict[str, float | None] = {}
            for cfg in LANGUAGE_CONFIGS:
                model_path = get_mingram_model_path(cfg["train_corpus"], f, EM, p)

                morph_key = f"{cfg['lang']}/mingram/{model_path.name}"
                model = None
                if morph_key in morphalign_cache:
                    morph_row[cfg["lang"]] = float(morphalign_cache[morph_key])
                else:
                    model = MinGramModel.load(str(model_path))
                    morph_row[cfg["lang"]] = morphalign_score(
                        model,
                        cfg["gold_file"],
                        morphalign_cache,
                        morph_key,
                    )

                comp_key = _compression_cache_key(cfg["train_corpus"], cfg["eval_corpus"], "mingram", model_path)
                if comp_key in compression_cache:
                    total_tokens = int(compression_cache[comp_key])
                    baseline = baseline_tokens[cfg["lang"]]
                    comp_row[cfg["lang"]] = (total_tokens - baseline) / baseline * 100
                else:
                    comp_row[cfg["lang"]] = None

            rows.append(
                {
                    "f": f,
                    "p": p,
                    "label": P_LABEL[p],
                    "morphalign": morph_row,
                    "morphalign_mean": _gmean(morph_row),
                    "compression": comp_row,
                    "compression_mean": _mean_or_none(comp_row),
                }
            )
    return rows


def build_table(rows: list[dict]) -> str:
    by_factor = {(row["f"], row["p"]): row for row in rows}

    lines = [
        "% Requires \\usepackage{booktabs} for \\toprule, \\midrule, \\bottomrule.",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{MorphAlign Score $\\times 100$} & \\multicolumn{2}{c}{Compression $\\Delta$ mean} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "$f$ & single-step & iter. $-$ single & single-step & iter. $-$ single \\\\",
        "\\midrule",
    ]

    for f in F_VALUES:
        single = by_factor[(f, 0.0)]
        iterative = by_factor[(f, 0.9)]
        morph_delta = iterative["morphalign_mean"] - single["morphalign_mean"]
        compression_delta = None
        if iterative["compression_mean"] is not None and single["compression_mean"] is not None:
            compression_delta = iterative["compression_mean"] - single["compression_mean"]
        cells = [
            _fmt_f(f),
            _fmt_morph_decimal(single["morphalign_mean"]),
            _fmt_morph_delta(morph_delta),
            _fmt_comp_mean(single["compression_mean"]),
            _fmt_comp_delta(compression_delta),
        ]
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    morphalign_cache = load_cache(MORPHALIGN_JSON)
    compression_cache = load_cache(COMPRESSION_CACHE_JSON)
    rows = collect_rows(morphalign_cache, compression_cache)
    tex = build_table(rows)

    save_cache(morphalign_cache, MORPHALIGN_JSON)
    save_cache(compression_cache, COMPRESSION_CACHE_JSON)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)

    print(f"Wrote {OUT_TEX}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
