#!/usr/bin/env python3
"""Generate the appendix EM-ablation table.

Reads the shared compression and MorphAlign caches, computes any missing
metrics for MinGram models at `f=1.15` (MAIN_F), `p=0.0`, adds PathPiece-BPE as a
reference point, and writes the appendix table at
`results/mingram_paper/tables/app_table_em_ablation.tex`.
"""

import json
from pathlib import Path

from paper_utils.hybrid.generate_morphalign_scatter import (
    LANGUAGE_CONFIGS as MORPHALIGN_LANGUAGE_CONFIGS,
    morphalign_score,
)
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import geomean, morphalign_paper_score, paper_table_path
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as get_unigram_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.pathpiece import PathPieceModel
from script_bpe.tokenizers.unigram import UnigramModel

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
MORPHALIGN_JSON = RESULTS_DIR / "cache_morphalign_scatter.json"
COMPRESSION_CACHE_JSON = RESULTS_DIR / "cache_train_eval_compression_grid.json"
OUT_TEX = paper_table_path("table_em_ablation.tex", appendix=True)

MAIN_F = 1.15
PLOT_P = 0.0
EM_VALUES = [0, 1, 2, 3, 4]
PATHPIECE_ROW_LABEL = "PathPiece\\hspace{0pt}-BPE"

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


def _mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def _gmean(values: dict[str, float]) -> float:
    return geomean(values.values())


def collect_rows(morphalign_cache: dict, compression_cache: dict) -> list[dict]:
    baseline_tokens: dict[str, int] = {}
    for cfg in LANGUAGE_CONFIGS:
        default_path = get_unigram_model_path(cfg["train_corpus"], DEFAULTS)
        cache_key = _compression_cache_key(cfg["train_corpus"], cfg["eval_corpus"], "default", default_path)
        if cache_key in compression_cache:
            baseline_tokens[cfg["lang"]] = int(compression_cache[cache_key])
        else:
            default_model = UnigramModel.load(str(default_path))
            baseline_tokens[cfg["lang"]] = _token_count(default_model, cfg["eval_corpus"], compression_cache, cache_key)

    rows = []
    for em in EM_VALUES:
        morph_row: dict[str, float] = {}
        comp_row: dict[str, float] = {}
        for cfg in LANGUAGE_CONFIGS:
            model_path = get_mingram_model_path(cfg["train_corpus"], MAIN_F, em, PLOT_P)
            morph_key = f"{cfg['lang']}/mingram/{model_path.name}"
            comp_key = _compression_cache_key(cfg["train_corpus"], cfg["eval_corpus"], "mingram", model_path)

            model = None
            if morph_key in morphalign_cache:
                morph_row[cfg["lang"]] = float(morphalign_cache[morph_key])
            else:
                model = MinGramModel.load(str(model_path))
                morph_row[cfg["lang"]] = morphalign_score(model, cfg["gold_file"], morphalign_cache, morph_key)

            if comp_key in compression_cache:
                total_tokens = int(compression_cache[comp_key])
            else:
                if model is None:
                    model = MinGramModel.load(str(model_path))
                total_tokens = _token_count(model, cfg["eval_corpus"], compression_cache, comp_key)
            baseline = baseline_tokens[cfg["lang"]]
            comp_row[cfg["lang"]] = (total_tokens - baseline) / baseline * 100

        rows.append(
            {
                "em": em,
                "label": str(em),
                "morphalign": morph_row,
                "morphalign_mean": _gmean(morph_row),
                "compression": comp_row,
                "compression_mean": _mean(comp_row),
            }
        )

    morph_row: dict[str, float] = {}
    comp_row: dict[str, float] = {}
    for cfg in LANGUAGE_CONFIGS:
        model_path = get_pathpiece_model_path(cfg["train_corpus"], init="bpe")
        morph_key = f"{cfg['lang']}/pathpiece_bpe/{model_path.name}"
        comp_key = _compression_cache_key(cfg["train_corpus"], cfg["eval_corpus"], "pathpiece_bpe", model_path)

        model = None
        if morph_key in morphalign_cache:
            morph_row[cfg["lang"]] = float(morphalign_cache[morph_key])
        else:
            model = PathPieceModel.load(str(model_path))
            morph_row[cfg["lang"]] = morphalign_score(model, cfg["gold_file"], morphalign_cache, morph_key)

        if comp_key in compression_cache:
            total_tokens = int(compression_cache[comp_key])
        else:
            if model is None:
                model = PathPieceModel.load(str(model_path))
            total_tokens = _token_count(model, cfg["eval_corpus"], compression_cache, comp_key)
        baseline = baseline_tokens[cfg["lang"]]
        comp_row[cfg["lang"]] = (total_tokens - baseline) / baseline * 100

    rows.append(
        {
            "em": None,
            "label": PATHPIECE_ROW_LABEL,
            "morphalign": morph_row,
            "morphalign_mean": _gmean(morph_row),
            "compression": comp_row,
            "compression_mean": _mean(comp_row),
        }
    )
    return rows


def build_table(rows: list[dict]) -> str:
    col_spec = "l" + "r" * (len(LANG_ORDER) + 1)
    header = ["Setting"] + [LANG_LABEL[lang] for lang in LANG_ORDER] + ["Mean"]
    n_cols = len(header)

    lines = [
        "% Requires \\usepackage{booktabs} for \\toprule, \\midrule, \\bottomrule.",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{MorphAlign Score}}}} \\\\",
    ]

    for row in rows:
        cells = [row["label"]]
        cells.extend(f"{morphalign_paper_score(row['morphalign'][lang]):.2f}" for lang in LANG_ORDER)
        cells.append(f"{morphalign_paper_score(row['morphalign_mean']):.2f}")
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\midrule",
        f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Compression $\\Delta$ vs Unigram (\\%, lower is better)}}}} \\\\",
    ]

    for row in rows:
        cells = [row["label"]]
        cells.extend(f"{row['compression'][lang]:+.2f}\\%" for lang in LANG_ORDER)
        cells.append(f"{row['compression_mean']:+.2f}\\%")
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    morphalign_cache = json.loads(MORPHALIGN_JSON.read_text())
    compression_cache = json.loads(COMPRESSION_CACHE_JSON.read_text())
    rows = collect_rows(morphalign_cache, compression_cache)
    tex = build_table(rows)

    MORPHALIGN_JSON.write_text(json.dumps(morphalign_cache, indent=2))
    COMPRESSION_CACHE_JSON.write_text(json.dumps(compression_cache, indent=2))
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)

    print(f"Wrote {OUT_TEX}")
    print()
    print(tex)


if __name__ == "__main__":
    main()
