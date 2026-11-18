#!/usr/bin/env python3
r"""LaTeX tables for the English/Korean shared-vocabulary study.

    table_bilingual_main.tex       total vocabulary 34,685, BPE and MinGram, every scheme
    table_bilingual_learned.tex    MinGram, 64,000 learned entries, two English/Korean mixtures

Each row is one tokenizer trained on English and Korean text together. Columns are
characters per token on English and on Korean evaluation text, the learned entries by
script, and the share of learned entries that duplicate another one by a leading space or
by capitalisation. The `plain` row of each block is absolute; the other rows are changes
against it, except the duplicate shares, which are absolute in every row.

Input
-----
`bilingual_tables.json`, a tracked extract of the two files `summarize_bilingual.py` writes
(`bilingual_results.json`, `bilingual_results_learned64000.json`). Those are gitignored:
they come from run directories that exist only on the machine that ran the grid. The
extract keeps the fields these tables read, for the runs they show, plus the sha256 of each
source file, so the tables rebuild from a fresh clone. Refresh it with `extract`:

    uv run python paper_utils/boundary/make_bilingual_tables.py extract
    uv run python paper_utils/boundary/make_bilingual_tables.py

The run set is declared below, block by block. Both commands refuse anything that does not
match it -- a missing or extra run, a block whose runs saw different training text, a
round-trip failure, a script category the columns do not show -- rather than print a
partial table.
"""

import hashlib
import json
import os

import cyclopts

from paper_utils.boundary.make_intrinsic_table import (
    ARM_LABEL,
    ARM_ORDER,
    PLAIN_LABEL,
    TRAINER_LABEL,
    absolute_cell,
    delta_cell,
    render,
)
from paper_utils.boundary.utils import GENERATED, rel

EXTRACT = os.path.join(GENERATED, "bilingual_tables.json")
SOURCE_TOTAL = os.path.join(GENERATED, "bilingual_results.json")
SOURCE_LEARNED = os.path.join(GENERATED, "bilingual_results_learned64000.json")

ALL_ARMS = ["plain", *ARM_ORDER]
LANGS = ["en", "ko"]
LANG_LABEL = {"en": "English", "ko": "Korean"}
# Script categories of bilingual_analysis.py. The table shows the first four; the last two
# are zero in every run and must stay zero, or the four shown would not add up.
SHOWN_SCRIPTS = ["latin", "hangul", "other_script", "nonlexical"]
SCRIPT_LABEL = {"latin": "Latin", "hangul": "Hangul", "other_script": "Other", "nonlexical": "Nonlex."}
HIDDEN_SCRIPTS = ["mixed", "undecodable"]
DUPLICATE_MEASURES = ["space", "case"]

# Printed digits: absolute characters per token, its percentage change, duplicate shares.
ABS_DIGITS = 3
DELTA_DIGITS = 2
SHARE_DIGITS = 1

MIX_2B_EN50 = "fineweb_enko_2000m_en50_local_v2"
MIX_5B_EN80 = "fineweb_enko_5000m_en80_local_v2"

# The run set, declared. `budget` selects runs the way summarize_bilingual.py does: by total
# vocabulary for the main table, by learned entries for the other. The 20M-character pilots
# are left out on purpose; the bilingual report calls them exploratory.
TABLES = {
    "main": {
        "source": SOURCE_TOTAL,
        "budget": ("total", 34_685),
        "out": os.path.join(GENERATED, "table_bilingual_main.tex"),
        "label": "tab:bilingual-main",
        "block_by": "trainer",
        "blocks": [(MIX_2B_EN50, "bpe", ALL_ARMS), (MIX_2B_EN50, "mingram", ALL_ARMS)],
    },
    "learned": {
        "source": SOURCE_LEARNED,
        "budget": ("learned", 64_000),
        "out": os.path.join(GENERATED, "table_bilingual_learned.tex"),
        "label": "tab:bilingual-learned",
        "block_by": "mixture",
        "blocks": [(MIX_2B_EN50, "mingram", ALL_ARMS),
                   (MIX_5B_EN80, "mingram", ["plain", "bnd_w", "bnd_wpd", "bnd_wpd_caps"])],
    },
}

app = cyclopts.App()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compact(row: dict) -> dict:
    """The fields the tables read, from one row of summarize_bilingual.py's output."""
    ev, sample = row["evaluation"], row["training"]["sample"]
    return {
        "corpus": row["corpus"],
        "trainer": row["trainer"],
        "arm": row["arm"],
        "total_vocab": row["vocab"],
        "atomic_vocab": row["atomic"],
        "learned_vocab": sum(row["learned_categories"].values()),
        "learned_categories": row["learned_categories"],
        "duplicates_involved": {m: row["duplicates"][m]["involved_entries"] for m in DUPLICATE_MEASURES},
        "evaluation": {lg: {k: ev[lg][k] for k in ("characters", "tokens", "chars_per_token",
                                                   "roundtrip_failures")} for lg in LANGS},
        "evaluation_sha256": {lg: row["evaluation_provenance"][lg]["selected_sha256"] for lg in LANGS},
        "train_budget_chars": sample["budgets"],
        "train_sample_sha256": {lg: sample["sources"][lg]["sample_sha256"] for lg in LANGS},
        "tokenizer_sha256": row["training"]["tokenizer_sha256"],
    }


def check(name: str, spec: dict, runs: list[dict]) -> dict:
    """(corpus, trainer, arm) -> run, after every check the table depends on."""
    kind, size = spec["budget"]
    field = {"total": "total_vocab", "learned": "learned_vocab"}[kind]
    expected = {(c, t, a) for c, t, arms in spec["blocks"] for a in arms}
    cells = {}
    for run in runs:
        key = (run["corpus"], run["trainer"], run["arm"])
        if key in cells:
            raise SystemExit(f"[{name}] run {key} appears twice")
        cells[key] = run
    if set(cells) != expected:
        raise SystemExit(f"[{name}] run set differs from the declaration: "
                         f"missing {sorted(expected - set(cells))}, extra {sorted(set(cells) - expected)}")
    eval_hashes = {json.dumps(r["evaluation_sha256"], sort_keys=True) for r in cells.values()}
    if len(eval_hashes) != 1:
        raise SystemExit(f"[{name}] runs were evaluated on different text: {sorted(eval_hashes)}")
    for key, run in cells.items():
        if run[field] != size:
            raise SystemExit(f"[{name}] {key}: {field} is {run[field]}, expected {size}")
        hidden = {c: run["learned_categories"][c] for c in HIDDEN_SCRIPTS if run["learned_categories"][c]}
        if hidden:
            raise SystemExit(f"[{name}] {key}: nonzero script categories the table omits: {hidden}")
        failures = {lg: run["evaluation"][lg]["roundtrip_failures"] for lg in LANGS
                    if run["evaluation"][lg]["roundtrip_failures"]}
        if failures:
            raise SystemExit(f"[{name}] {key}: round-trip failures on evaluation text: {failures}")
    for corpus, trainer, arms in spec["blocks"]:
        samples = {json.dumps(cells[(corpus, trainer, a)]["train_sample_sha256"], sort_keys=True)
                   for a in arms}
        if len(samples) != 1:
            raise SystemExit(f"[{name}] {corpus} {trainer}: schemes trained on different text")
    return cells


@app.command
def extract(total: str = SOURCE_TOTAL, learned: str = SOURCE_LEARNED, out: str = EXTRACT) -> None:
    """Write the tracked extract from summarize_bilingual.py's two result files.

    Args:
        total: bilingual_results.json, runs selected by total vocabulary.
        learned: bilingual_results_learned64000.json, runs selected by learned entries.
        out: The extract to write.
    """
    paths = {"main": total, "learned": learned}
    for path in paths.values():
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}: it is written by summarize_bilingual.py, and the "
                             f"committed copy is in git at 642e10a")
    data = {"sources": {}, "tables": {}}
    for name, spec in TABLES.items():
        rows = json.load(open(paths[name]))
        wanted = {(c, t, a) for c, t, arms in spec["blocks"] for a in arms}
        runs = [compact(r) for r in rows if (r["corpus"], r["trainer"], r["arm"]) in wanted]
        check(name, spec, runs)
        data["sources"][name] = {"file": os.path.basename(paths[name]), "sha256": sha256(paths[name])}
        data["tables"][name] = runs
    with open(out, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print(f"[json] {out}: " + ", ".join(f"{n} {len(r)} runs" for n, r in data["tables"].items()))


def mixture_label(run: dict) -> str:
    budgets = run["train_budget_chars"]
    total = sum(budgets.values())
    english = round(100 * budgets["en"] / total)
    return rf"{total / 1e9:g}B characters, {english}\% English"


def row_cells(run: dict, base: dict) -> list:
    """One row's cells; `base` is the block's plain run, equal to `run` on the plain row."""
    is_base = run is base
    cells = []
    for lg in LANGS:
        cpt, base_cpt = (r["evaluation"][lg]["chars_per_token"] for r in (run, base))
        cells.append(absolute_cell(cpt, ABS_DIGITS) if is_base
                     else delta_cell(100 * (cpt / base_cpt - 1), DELTA_DIGITS))
    for script in SHOWN_SCRIPTS:
        n, base_n = (r["learned_categories"][script] for r in (run, base))
        # {,} keeps math mode from spacing the separator like a list comma.
        cells.append(f"{n:,}" if is_base else f"${n - base_n:+,}$".replace(",", "{,}"))
    for measure in DUPLICATE_MEASURES:
        share = 100 * run["duplicates_involved"][measure] / run["learned_vocab"]
        cells.append(f"{share:.{SHARE_DIGITS}f}")
    return [c if isinstance(c, str) else render(c, [None] * len(c.parts)) for c in cells]


def caption(name: str, spec: dict, cells: dict) -> str:
    any_run = next(iter(cells.values()))
    chars = {lg: any_run["evaluation"][lg]["characters"] for lg in LANGS}
    if name == "main":
        what = (rf"Tokenizers trained on English and Korean together, total vocabulary "
                rf"{spec['budget'][1]:,}, on {mixture_label(any_run)}, with the English share "
                r"measured in characters.")
    else:
        what = (rf"MinGram tokenizers trained on English and Korean together, with "
                rf"{spec['budget'][1]:,} learned entries each, on the two mixtures named in the "
                r"block headings, with English shares measured in characters.")
    return "\n".join([
        rf"\caption{{{what}",
        rf"Characters per token on {chars['en']:,} characters of English and {chars['ko']:,} of",
        r"Korean Goldfish text: absolute in the \plainscheme{} row, percentage change against",
        r"it in the other rows, positive meaning fewer tokens.",
        r"Learned entries by script: entries beyond the atomic alphabet,",
        r"grouped by the script of their letters, with \emph{Other} for letters of any other",
        r"script and \emph{Nonlex.} for entries without letters; counts in the \plainscheme{}",
        r"row, change in count in the other rows.",
        r"Duplicates: learned entries that differ from another learned entry only by a",
        r"leading space (\emph{space}) or only by capitalization (\emph{case}), as a",
        r"percentage of learned entries, both entries of a pair counted. Unlike",
        r"\Cref{tab:vocab-duplicates}, atomic tokens are excluded, and \emph{space}",
        r"includes whitespace-only pairs such as a newline and a space followed by a newline.}",
    ])


def write_table(name: str, spec: dict, cells: dict, source: str) -> None:
    n_cols = len(LANGS) + len(SHOWN_SCRIPTS) + len(DUPLICATE_MEASURES)
    first_script = 2 + len(LANGS)
    first_dup = first_script + len(SHOWN_SCRIPTS)
    lines = [
        "% Generated by paper_utils/boundary/make_bilingual_tables.py. Do not edit.",
        r"% Requires booktabs and the paper's \bnds and \plainscheme macros.",
        f"% source: {rel(source)}, extract of {os.path.basename(spec['source'])}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * n_cols + "}",
        r"\toprule",
        rf" & \multicolumn{{{len(LANGS)}}}{{c}}{{Characters per token}}"
        rf" & \multicolumn{{{len(SHOWN_SCRIPTS)}}}{{c}}{{Learned entries by script}}"
        rf" & \multicolumn{{{len(DUPLICATE_MEASURES)}}}{{c}}{{Duplicates (\%)}} \\",
        rf"\cmidrule(lr){{2-{first_script - 1}}} \cmidrule(lr){{{first_script}-{first_dup - 1}}}"
        rf" \cmidrule(lr){{{first_dup}-{first_dup + len(DUPLICATE_MEASURES) - 1}}}",
        "Scheme & " + " & ".join([LANG_LABEL[lg] for lg in LANGS]
                                 + [SCRIPT_LABEL[s] for s in SHOWN_SCRIPTS]
                                 + DUPLICATE_MEASURES) + r" \\",
        r"\midrule",
    ]
    for i, (corpus, trainer, arms) in enumerate(spec["blocks"]):
        if i:
            lines.append(r"\midrule")
        base = cells[(corpus, trainer, "plain")]
        heading = TRAINER_LABEL[trainer] if spec["block_by"] == "trainer" else mixture_label(base)
        lines.append(rf"\multicolumn{{{1 + n_cols}}}{{l}}{{\emph{{{heading}}}}} \\")
        for arm in arms:
            label = PLAIN_LABEL if arm == "plain" else ARM_LABEL[arm]
            lines.append(f"{label} & " + " & ".join(row_cells(cells[(corpus, trainer, arm)], base)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", caption(name, spec, cells), rf"\label{{{spec['label']}}}", ""]
    with open(spec["out"], "w") as f:
        f.write("\n".join(lines))
    print(f"[tex] {spec['out']}: {len(cells)} runs in {len(spec['blocks'])} blocks")


@app.default
def main(extract_path: str = EXTRACT) -> None:
    """Write both tables from the tracked extract.

    Args:
        extract_path: The extract written by `extract`.
    """
    if not os.path.exists(extract_path):
        raise SystemExit(f"missing {extract_path}: run `make_bilingual_tables.py extract` "
                         "on a machine holding summarize_bilingual.py's output")
    data = json.load(open(extract_path))
    if set(data["tables"]) != set(TABLES):
        raise SystemExit(f"{extract_path} holds tables {sorted(data['tables'])}, expected {sorted(TABLES)}")
    for name, spec in TABLES.items():
        write_table(name, spec, check(name, spec, data["tables"][name]), extract_path)


if __name__ == "__main__":
    app()
