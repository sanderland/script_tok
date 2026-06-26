#!/usr/bin/env python3
"""Generate a qualitative tokenization examples table.

The output is a LaTeX `tabular` body at
`results/mingram_paper/tables/table_tokenization_examples.tex`. Each method column carries
a single colour bar immediately below the header row, summarising that
method's boundary quality across all example words (green = no extra
predicted boundaries, yellow = one extra, red = two or more). Language
section headings are plain text labels; the bars are aggregated across all
examples, not per language.
"""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import re
import sys

from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import FSP_PARAMS, paper_table_path
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

RESULTS_DIR = Path("results/hybrid")
CONVEXTOK_PATH = "results/convextok_tokenizers/{corpus}/n32768_cmin50_mp200000_L32_det.json.gz"
OUT_TEX = paper_table_path("table_tokenization_examples.tex")

MAIN_F = 1.15
PLOT_EM = 2
PLOT_P = 0.0
MINGRAM_PP_F = 8.0
MINGRAM_PP_P = 0.9

EXAMPLES = [
    # English
    {"language": "English", "corpus": "fineweb_en_5gb", "word": "outspoken", "gold": ["out", "spoken"]},
    {"language": "English", "corpus": "fineweb_en_5gb", "word": "spaceship", "gold": ["space", "ship"]},
    {"language": "English", "corpus": "fineweb_en_5gb", "word": "southeast", "gold": ["south", "east"]},
    {"language": "English", "corpus": "fineweb_en_5gb", "word": "coloured", "gold": ["colour", "ed"]},
    {"language": "English", "corpus": "fineweb_en_5gb", "word": "roughneck", "gold": ["rough", "neck"]},
    {"language": "English", "corpus": "fineweb_en_5gb", "word": "shoplifter", "gold": ["shop", "lift", "er"]},
    {"language": "English", "corpus": "fineweb_en_5gb", "word": "ultramodern", "gold": ["ultra", "modern"]},

    # German
    {"language": "German", "corpus": "fineweb_de_5gb", "word": "Achtelnote", "gold": ["Ach", "tel", "note"]},
    {"language": "German", "corpus": "fineweb_de_5gb", "word": "Nachtfalter", "gold": ["Nacht", "falter"]},
    {"language": "German", "corpus": "fineweb_de_5gb", "word": "bedrängen", "gold": ["be", "dräng", "en"]},
    {"language": "German", "corpus": "fineweb_de_5gb", "word": "zerkratzen", "gold": ["zer", "kratz", "en"]},

    # Finnish
    {"language": "Finnish", "corpus": "fineweb_fi_5gb", "word": "kestänyt", "gold": ["kestä", "nyt"]},
    {"language": "Finnish", "corpus": "fineweb_fi_5gb", "word": "toistaisit", "gold": ["toista", "isi", "t"]},
    {"language": "Finnish", "corpus": "fineweb_fi_5gb", "word": "virtaamme", "gold": ["virtaa", "mme"]},
    {"language": "Finnish", "corpus": "fineweb_fi_5gb", "word": "matkustaisimme", "gold": ["matkusta", "isi", "mme"]},
    {"language": "Finnish", "corpus": "fineweb_fi_5gb", "word": "muodostan", "gold": ["muodosta", "n"]},
]

BASE_METHOD_ORDER = ["bpe", "unigram", "fsp", "mingram", "pathpiece_bpe", "convextok"]
METHOD_ORDER = BASE_METHOD_ORDER
METHOD_LABEL = {
    "bpe": "BPE",
    "unigram": "Unigram",
    "fsp": "FSP",
    "mingram": "MinGram",
    "mingram_pp": r"\mingrampp{}",
    "pathpiece_bpe": "PathPiece-BPE",
    "convextok": "ConvexTok",
}

COLOR_KEYS = ("tokgood", "tokmid", "tokbad")


def _load_model(method: str, corpus: str):
    if method == "bpe":
        path = RESULTS_DIR / corpus / f"bpe_n{ADDITIONAL_VOCAB_SIZE}.model.json.gz"
        return BPETokenizer.load(str(path))
    if method == "unigram":
        return UnigramModel.load(str(get_unigram_model_path(corpus, DEFAULTS)))
    if method == "fsp":
        return UnigramModel.load(str(get_unigram_model_path(corpus, FSP_PARAMS)))
    if method == "mingram":
        return MinGramModel.load(str(get_mingram_model_path(corpus, MAIN_F, PLOT_EM, PLOT_P)))
    if method == "mingram_pp":
        return MinGramModel.load(
            str(get_mingram_model_path(corpus, MINGRAM_PP_F, PLOT_EM, MINGRAM_PP_P, prune_criterion="mi"))
        )
    if method == "pathpiece_bpe":
        return PathPieceModel.load(str(get_pathpiece_model_path(corpus, init="bpe")))
    if method == "convextok":
        return ConvexTokModel.load(CONVEXTOK_PATH.format(corpus=corpus))
    raise ValueError(f"Unknown method: {method}")


def _tokenize(model, word: str) -> list[str]:
    token_ids = model.encode(word)
    pieces = [model.pretokenizer.decode(model.tokens[int(token_id)].atomic_tokens) for token_id in token_ids]
    assert "".join(pieces) == word, f"Roundtrip mismatch: {word!r} vs {pieces!r}"
    return pieces


def _escape_latex(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == " ":
            out.append(r"\textvisiblespace{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch in {"_", "%", "&", "#", "$", "{", "}"}:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _format_pieces(pieces: list[str]) -> str:
    return r"\tokens{" + ",".join(_escape_latex(piece) for piece in pieces) + "}"


def _boundary_positions(pieces: list[str]) -> set[int]:
    pos = 0
    boundaries: set[int] = set()
    for piece in pieces[:-1]:
        pos += len(piece)
        boundaries.add(pos)
    return boundaries


def _quality_color(gold_pieces: list[str], pred_pieces: list[str]) -> str:
    gold_boundaries = _boundary_positions(gold_pieces)
    pred_boundaries = _boundary_positions(pred_pieces)
    extra_boundaries = pred_boundaries - gold_boundaries
    if len(extra_boundaries) == 0:
        return "tokgood"
    if len(extra_boundaries) == 1:
        return "tokmid"
    return "tokbad"


def _format_prediction(gold_pieces: list[str], pred_pieces: list[str]) -> str:
    color = _quality_color(gold_pieces, pred_pieces)
    return r"\tokens[" + color + "]{" + ",".join(
        _escape_latex(piece) for piece in pred_pieces
    ) + "}"


def _rows_from_models() -> list[dict[str, object]]:
    models_by_corpus: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for example in EXAMPLES:
        corpus = example["corpus"]
        if corpus not in models_by_corpus:
            models_by_corpus[corpus] = {method: _load_model(method, corpus) for method in METHOD_ORDER}
        per_method = {
            method: _tokenize(models_by_corpus[corpus][method], example["word"])
            for method in METHOD_ORDER
        }
        rows.append(
            {
                "language": example["language"],
                "word": example["word"],
                "gold": example["gold"],
                "tokenizations": per_method,
                "colors": {
                    method: _quality_color(example["gold"], per_method[method])
                    for method in METHOD_ORDER
                },
            }
        )
    return rows


def _latex_piece_to_text(piece: str) -> str:
    replacements = {
        r"\textbackslash{}": "\\",
        r"\textvisiblespace{}": " ",
        r"\textasciitilde{}": "~",
        r"\textasciicircum{}": "^",
    }
    for escaped, raw in replacements.items():
        piece = piece.replace(escaped, raw)
    for ch in "_%&#${}":
        piece = piece.replace("\\" + ch, ch)
    return piece


def _parse_token_cell(cell: str) -> list[str]:
    match = re.search(r"\\tokens(?:\[[^\]]+\])?\{([^{}]*)\}", cell)
    if match is None:
        raise ValueError(f"Could not parse token cell: {cell!r}")
    body = match.group(1)
    if not body:
        return []
    return [_latex_piece_to_text(piece) for piece in body.split(",")]


def _rows_from_existing_table(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)

    data_lines = [
        line.removesuffix(r" \\")
        for line in path.read_text().splitlines()
        if line.startswith(r"\tokens{")
    ]
    if len(data_lines) != len(EXAMPLES):
        raise ValueError(f"Expected {len(EXAMPLES)} tokenization rows in {path}, found {len(data_lines)}")

    rows: list[dict[str, object]] = []
    for example, line in zip(EXAMPLES, data_lines, strict=True):
        cells = line.split(" & ")
        if len(cells) != len(METHOD_ORDER) + 1:
            raise ValueError(f"Expected {len(METHOD_ORDER) + 1} cells in row: {line!r}")
        gold = _parse_token_cell(cells[0])
        per_method = {
            method: _parse_token_cell(cell)
            for method, cell in zip(METHOD_ORDER, cells[1:], strict=True)
        }
        rows.append(
            {
                "language": example["language"],
                "word": example["word"],
                "gold": gold,
                "tokenizations": per_method,
                "colors": {
                    method: _quality_color(gold, per_method[method])
                    for method in METHOD_ORDER
                },
            }
        )
    return rows


def _proportions(counter: Counter[str]) -> tuple[float, float, float]:
    total = sum(counter.values())
    if total == 0:
        return (0.0, 0.0, 0.0)
    return tuple(counter[color] / total for color in COLOR_KEYS)


def _method_bar(counter: Counter[str]) -> str:
    g, m, b = _proportions(counter)
    return rf"\tokmethodbar{{{g:.3f}}}{{{m:.3f}}}{{{b:.3f}}}"


def _overall_per_method_counts(rows: list[dict[str, object]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {method: Counter() for method in METHOD_ORDER}
    for row in rows:
        for method, color in row["colors"].items():
            counts[method][color] += 1
    return counts


def build_table(rows: list[dict[str, object]]) -> str:
    method_counts = _overall_per_method_counts(rows)

    headers = ["Gold"] + [METHOD_LABEL[method] for method in METHOD_ORDER]
    n_cols = len(headers)
    bar_cells = [""] + [_method_bar(method_counts[method]) for method in METHOD_ORDER]
    lines = [
        "% Intended to be wrapped in \\begin{table}...\\end{table}",
        "% Requires \\usepackage{booktabs}.",
        "% Requires \\usepackage{tikz}.",
        "% Requires \\tokens[<color>]{...} and \\tokmethodbar{good}{mid}{bad} macros.",
        "% Each method column carries a single colour bar summarising that method's",
        "% green/yellow/red distribution across all example words.",
        r"\begingroup",
        r"\setlength{\tabcolsep}{1.8pt}",
        r"\tiny",
        r"\begin{tabular}{@{}" + "l" * n_cols + r"@{}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        " & ".join(bar_cells) + r" \\",
        r"\midrule",
    ]

    previous_language = None
    for idx, row in enumerate(rows):
        language = _escape_latex(row["language"])
        if language != previous_language:
            if previous_language is not None:
                lines.append(r"\addlinespace[0.35em]")
            lines.append(rf"\multicolumn{{{n_cols}}}{{@{{}}l@{{}}}}{{\small\textbf{{{language}}}}} \\")
            previous_language = language

        cells = [_format_pieces(row["gold"])]
        for method in METHOD_ORDER:
            cells.append(_format_prediction(row["gold"], row["tokenizations"][method]))
        lines.append(" & ".join(cells) + r" \\")
        next_language = None if idx == len(rows) - 1 else _escape_latex(rows[idx + 1]["language"])
        if next_language == language:
            lines.append(r"\addlinespace[0.2em]")

    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup"]
    return "\n".join(lines)


def _load_rows(allow_fallback: bool = True) -> list[dict[str, object]]:
    try:
        return _rows_from_models()
    except FileNotFoundError as exc:
        if not allow_fallback:
            raise
        print(
            f"Warning: {exc}. Falling back to tokenizations already present in {OUT_TEX}.",
            file=sys.stderr,
        )
        return _rows_from_existing_table(OUT_TEX)


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-mingram-mi",
        action="store_true",
        help="Add the MinGram-PP column. Requires the f8 MinGram-PP tokenizer models locally.",
    )
    args = parser.parse_args()

    global METHOD_ORDER
    if args.include_mingram_pp:
        METHOD_ORDER = ["bpe", "unigram", "fsp", "mingram", "mingram_pp", "pathpiece_bpe", "convextok"]

    rows = _load_rows(allow_fallback=not args.include_mingram_pp)
    tex = build_table(rows)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
