#!/usr/bin/env python3
r"""How much of a vocabulary goes on holding the same word twice.

The leading-space convention gives a word two forms, `' the'` and `'the'`, and a trainer
that meets both often enough spends an entry on each. Capitalisation does the same to
`'The'`. This counts what that costs, per language and per trainer, against the schemes
that are meant to remove it: the boundary markers for the space form, the case codes for
the cased one.

What is counted
---------------
Two entries are duplicates when their surface forms differ only by

    space   one leading space         `' the'` and `'the'`
    case    capitalisation            `'The'` and `'the'`, `'<^><|>the<|>'` and `'<|>the<|>'`

and the figure reported is the share of the vocabulary sitting in such a group, counting
*every* entry in it rather than the surplus: a word held in two forms occupies two slots,
and which of the two is the duplicate is not a question the vocabulary answers. So a
32,768-entry vocabulary reading 26% has roughly 4,300 words in it twice.

The two overlap -- `' The'` is a duplicate of `'the'` under both -- so they do not add.
The union is measured too and kept in the JSON, but a third column of it says little the
first two do not.

Surface forms, not decoded text
-------------------------------
`tokens_repr` decodes, and decoding is where the markers go: `'<|>the<|>'`, `'<|>the'` and
`'the'` all come back as `the`, which would count three distinct entries as two duplicate
pairs and report a marker scheme as the worst of the lot. The repr here keeps the marker
and case-code atomics visible, so a delimited word and a word-internal fragment are the
different entries they are.

Under that repr the marker schemes are exactly zero in the space column, and it is worth
being clear that this is arithmetic rather than a result: a word span is delimited on both
sides unconditionally, the space between two spans is elided, and so no entry can carry a
leading space in front of a delimited word for another entry to duplicate. The measurement
confirms the scheme does what it says; the number to compare against it is the baseline's.

Reads the trained grid when it is there and the committed cache when it is not, so the
table regenerates on a machine with no tokenizers.

    uv run python paper_utils/boundary/vocab_duplicates.py
    uv run python paper_utils/boundary/vocab_duplicates.py --force
"""

import collections
import json
import os

import cyclopts

from script_bpe.tokenizers.load import load_tokenizer
from paper_utils.boundary.boundary_pretokenizer import BoundaryScriptPretokenizer
from paper_utils.boundary.make_intrinsic_table import (
    ARM_LABEL,
    ARM_ORDER,
    KNOWN_TRAINERS,
    LANG_LABEL,
    LANG_ORDER,
    MISSING,
    PLAIN_LABEL,
    TRAINER_LABEL,
)
from paper_utils.boundary.utils import GENERATED, TOKENIZERS, rel

# Registers `BoundaryScriptPretokenizer` so the marker tokenizers load.
import paper_utils.boundary.downstream.boundary_tokenizer  # noqa: F401

ARMS = ["plain", *ARM_ORDER]
MEASURES = ["space", "case"]
MEASURE_LABEL = {"space": "space", "case": "case"}

app = cyclopts.App()


def special_texts(pretokenizer) -> dict[int, str]:
    """Atomic ids that decode to nothing readable, mapped to how the table should show them.

    Only the boundary pretokenizer has any: the marker and the two case codes. Everything
    else in a vocabulary entry is ordinary text and goes through `decode`.
    """
    if not isinstance(pretokenizer, BoundaryScriptPretokenizer):
        return {}
    codes = {pretokenizer.shift_token_id: pretokenizer.SHIFT_TEXT,
             pretokenizer.caps_token_id: pretokenizer.CAPS_TEXT}
    return {pretokenizer.marker_token_id: pretokenizer.MARKER_TEXT,
            **{tid: text for tid, text in codes.items() if tid is not None}}


def surface(pretokenizer, atomic_token_ids, specials: dict[int, str]) -> str:
    """One vocabulary entry as text, with markers and case codes left visible."""
    parts: list[str] = []
    run: list[int] = []
    for tid in atomic_token_ids:
        if tid in specials:
            if run:
                parts.append(pretokenizer.decode(run, errors="backslashreplace"))
                run = []
            parts.append(specials[tid])
        else:
            run.append(tid)
    if run:
        parts.append(pretokenizer.decode(run, errors="backslashreplace"))
    return "".join(parts)


def despace(form: str) -> str:
    return form[1:] if form.startswith(" ") else form


def decase(form: str) -> str:
    """Lowercase, with a leading case code stripped: the code *is* the capitalisation."""
    for code in (BoundaryScriptPretokenizer.CAPS_TEXT, BoundaryScriptPretokenizer.SHIFT_TEXT):
        if form.startswith(code):
            return form[len(code):].lower()
    return form.lower()


def duplicate_slots(forms: list[str], key) -> int:
    """Entries whose key another entry shares, the group counted whole."""
    groups = collections.Counter(key(f) for f in forms)
    return sum(n for n in groups.values() if n > 1)


def measure(path: str) -> dict:
    tokenizer = load_tokenizer(path)
    pretokenizer = tokenizer.pretokenizer
    specials = special_texts(pretokenizer)
    forms = [surface(pretokenizer, token.atomic_tokens, specials)
             for token in tokenizer.tokens.values()]
    # Distinct entries must have distinct surface forms, or the counts below are measuring
    # a lossy repr rather than the vocabulary -- which is exactly what decoding does to the
    # markers. Loud, because a silent collision reads as duplication that is not there.
    collisions = [f for f, n in collections.Counter(forms).items() if n > 1]
    assert not collisions, f"{len(collisions)} colliding surface form(s) in {path}: {collisions[:5]}"
    return {
        "vocab_size": len(forms),
        "space_slots": duplicate_slots(forms, despace),
        "case_slots": duplicate_slots(forms, decase),
        "either_slots": duplicate_slots(forms, lambda f: decase(despace(f))),
    }


def read_cells(data: dict) -> dict[tuple[str, str, str], dict]:
    """(lang, arm, trainer) -> measurement, from the cache's own cell keys."""
    return {(c["lang"], c["arm"], c["trainer"]): c for c in data.values()}


def percentage(cell: dict | None, measure_name: str) -> float | None:
    if cell is None:
        return None
    return 100 * cell[f"{measure_name}_slots"] / cell["vocab_size"]


def mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def fmt(value: float | None) -> str:
    return MISSING if value is None else f"{value:.1f}"


def block(cells, langs, trainers, arms) -> list[str]:
    """One `\\midrule`-separated block of rows per trainer, in the trainer's order."""
    lines = []
    for i, trainer in enumerate(trainers):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + 2 * (len(langs) + 1)}}}{{l}}{{\emph{{{TRAINER_LABEL[trainer]}}}}} \\")
        for arm in arms:
            values = {m: [percentage(cells.get((lg, arm, trainer)), m) for lg in langs]
                      for m in MEASURES}
            # A language's measures sit side by side, so the per-measure lists interleave.
            cols = [fmt(values[m][i]) for i in range(len(langs)) for m in MEASURES]
            cols += [fmt(mean(values[m])) for m in MEASURES]
            label = PLAIN_LABEL if arm == "plain" else ARM_LABEL[arm]
            lines.append(f"{label} & " + " & ".join(cols) + r" \\")
    return lines


@app.default
def main(
    langs: str = ",".join(LANG_ORDER),
    arms: str = ",".join(ARMS),
    trainers: str = ",".join(KNOWN_TRAINERS),
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    cache: str = os.path.join(GENERATED, "vocab_duplicates.json"),
    out: str = os.path.join(GENERATED, "table_vocab_duplicates.tex"),
    label: str = "tab:vocab-duplicates",
    force: bool = False,
) -> None:
    """Measure duplicate vocabulary entries per cell and write the appendix table.

    Args:
        langs: Comma-separated languages, one column pair each.
        arms: Comma-separated schemes, one row each.
        trainers: Comma-separated trainers, one block each.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        cache: JSON of per-cell counts. Extended in place, and the table's only source, so
            a machine with no trained tokenizers still regenerates the table.
        out: LaTeX file to write, \\input{} from a table* environment.
        label: LaTeX label.
        force: Remeasure cells already in the cache instead of keeping them.
    """
    langs = [x.strip() for x in langs.split(",") if x.strip()]
    arms = [x.strip() for x in arms.split(",") if x.strip()]
    trainers = [x.strip() for x in trainers.split(",") if x.strip()]

    counts = json.load(open(cache)) if os.path.exists(cache) else {}
    measured, missing = 0, 0
    for lang in langs:
        for arm in arms:
            for trainer in trainers:
                key = f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}"
                if key in counts and not force:
                    continue
                path = os.path.join(TOKENIZERS, f"{key}.json.gz")
                if not os.path.exists(path):
                    missing += 1
                    continue
                counts[key] = {"lang": lang, "arm": arm, "trainer": trainer, **measure(path)}
                measured += 1
                print(f"  {key}: " + "  ".join(
                    f"{m} {percentage(counts[key], m):.1f}%" for m in [*MEASURES, "either"]), flush=True)
                with open(cache, "w") as f:
                    json.dump(counts, f, indent=2, sort_keys=True)
    print(f"[json] {cache}: {len(counts)} cell(s), {measured} measured now, "
          f"{missing} without a trained tokenizer")

    cells = read_cells(counts)
    langs = [lg for lg in langs if any(c[0] == lg for c in cells)]
    trainers = [tr for tr in trainers if any(c[2] == tr for c in cells)]
    arms = [arm for arm in ARMS if arm in arms and any(c[1] == arm for c in cells)]
    if not langs:
        raise SystemExit(f"no usable cells in {cache}")

    columns = [LANG_LABEL[lg] for lg in langs] + ["Mean"]
    subheads = [MEASURE_LABEL[m] for m in MEASURES]
    header = [
        r"\begin{tabular}{l" + " rr" * len(columns) + "}",
        r"\toprule",
        "Scheme " + "".join(rf"& \multicolumn{{2}}{{c}}{{{c}}} " for c in columns) + r"\\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(columns))),
        " & " + " & ".join(subheads * len(columns)) + r" \\",
        r"\midrule",
    ]
    caption = "\n".join([
        r"\caption{Vocabulary entries duplicating another entry, as a percentage of",
        r"vocabulary size. \emph{space}: differing only by a leading space.",
        r"\emph{case}: differing only by capitalization, including a case code plus",
        r"its span. Both entries of a pair count, and the measures overlap.",
        r"Marker schemes are zero for \emph{space} by construction.",
        r"Case codes lower \emph{case} without clearing it, as mixed case is left",
        r"literal and frequent title-case forms still earn entries.}",
    ])
    tex = "\n".join([
        "% Generated by paper_utils/boundary/vocab_duplicates.py. Do not edit.",
        r"% Requires booktabs, xcolor and the paper's \bnds and \plainscheme macros.",
        f"% source: {rel(cache)} (quick corpus)",
        r"\centering",
        r"\tiny",
        # Same reasoning as the per-language compression table: fifteen columns need the
        # gutters cut from the 6pt default or the padding alone overfills the page.
        r"\setlength{\tabcolsep}{3pt}",
        *header,
        *block(cells, langs, trainers, arms),
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        rf"\label{{{label}}}",
        "",
    ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(tex)
    filled = sum(1 for lg in langs for tr in trainers for arm in arms if (lg, arm, tr) in cells)
    print(f"[tex] {out}")
    print(f"[tex] {filled}/{len(langs) * len(trainers) * len(arms)} cells: {len(langs)} language(s), "
          f"trainer(s) {', '.join(trainers)}, scheme(s) {', '.join(arms)}")


if __name__ == "__main__":
    app()
