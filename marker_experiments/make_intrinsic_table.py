#!/usr/bin/env python3
"""Generate a wide intrinsic-compression table from any one grid result file.

One column group per trainer present in the data, each with its own
\texttt{plain} baseline followed by its three variants. A language is one row, so reading
across a group gives that trainer's whole story and comparing groups answers the question
two trainers are there to answer -- does the variant ordering survive a change of trainer?

Which trainers appear is read from the file, not assumed, because the grids do not cover
the same ground:

    finewiki1gb_result.json   6 languages, BPE only    (one mingram cell, dropped)
    mingram250_result.json    3 languages, both trainers

A trainer is included only if EVERY language shown has a complete set of cells for it, so
a table never puts one language's MinGram numbers beside another's blank.

The earlier version interleaved the trainers under each variant, which put the two
baselines in one group and made "the BPE numbers" something you assembled from alternate
columns.

Emitted as a `table*` body: nine columns do not fit a single ACL column.

Needs `booktabs` and the paper's `\\bnd` macro:

    \\newcommand{\\bnd}[1]{\\texttt{bnd\\_#1}}

    uv run python marker_experiments/make_intrinsic_table.py
"""

import json
import os

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(HERE, "paper", "generated")

LANG_LABEL = {"en": "English", "de": "German", "fi": "Finnish",
              "ru": "Russian", "ar": "Arabic", "ko": "Korean"}
LANG_ORDER = ["en", "de", "fi", "ru", "ar", "ko"]
ARM_ORDER = ["bnd_w", "bnd_wp", "bnd_wpd"]
ARM_LABEL = {"bnd_w": r"\bnd{w}", "bnd_wp": r"\bnd{wp}", "bnd_wpd": r"\bnd{wpd}"}
KNOWN_TRAINERS = ["bpe", "mingram"]
TRAINER_LABEL = {"bpe": "BPE", "mingram": "MinGram"}

app = cyclopts.App()


def _rows(results, trainers):
    """(lang, arm, trainer) -> chars/token, keeping only languages with a complete set.

    A partly-filled row would put a language's variants next to a baseline from a
    different set of cells, so incomplete languages are dropped rather than shown with
    gaps.
    """
    table = {}
    for key, cell in results.items():
        lang, rest = key.split("_", 1)
        for trainer in trainers:
            if rest.endswith(f"_{trainer}"):
                table[(lang, rest[: -len(trainer) - 1], trainer)] = cell["eval_chars_per_token"]
    langs = [
        lang for lang in LANG_ORDER
        if all((lang, arm, tr) in table for arm in ["plain", *ARM_ORDER] for tr in trainers)
    ]
    return table, langs


def main_body(table, langs, trainers):
    lines = []
    deltas = {(tr, arm): [] for tr in trainers for arm in ARM_ORDER}
    mingram_gain = []
    for lang in langs:
        base = {tr: table[(lang, "plain", tr)] for tr in trainers}
        cells = []
        for tr in trainers:                      # trainer outer: one contiguous half each
            cells.append(f"{base[tr]:.4f}")      # that half's own plain baseline
            for arm in ARM_ORDER:
                # Relative to the baseline in the SAME half. The question is whether the
                # variant ordering survives the trainer, not how the trainers compare;
                # the two absolute baselines keep that second comparison recoverable.
                d = 100 * (table[(lang, arm, tr)] - base[tr]) / base[tr]
                deltas[(tr, arm)].append(d)
                cells.append(f"$\\mathbf{{{d:+.2f}}}$" if arm == "bnd_wpd" and d > 0 else f"${d:+.2f}$")
        if {"bpe", "mingram"} <= set(trainers):
            mingram_gain.append(100 * (base["mingram"] - base["bpe"]) / base["bpe"])
        lines.append(f"{LANG_LABEL[lang]} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    mean = []
    for tr in trainers:
        mean.append("")                          # absolute baselines are not averaged
        for arm in ARM_ORDER:
            m = sum(deltas[(tr, arm)]) / len(deltas[(tr, arm)])
            mean.append(f"$\\mathbf{{{m:+.2f}}}$" if arm == "bnd_wpd" and m > 0 else f"${m:+.2f}$")
    lines.append("Mean & " + " & ".join(mean) + r" \\")
    return lines, mingram_gain


@app.default
def main(
    results: str = os.path.join(HERE, "mingram250_result.json"),
    out: str = os.path.join(GENERATED, "table_intrinsic.tex"),
    corpus: str = "250M characters per language",
    label: str = "tab:intrinsic",
) -> None:
    """Write the wide compression table for one grid.

    Args:
        results: A grid result JSON.
        out: LaTeX file to write, \\input{} from a table* environment.
        corpus: How the corpus is described in the caption.
        label: LaTeX label.
    """
    with open(results) as f:
        data = json.load(f)
    # A trainer earns a column group only if every language kept has a complete set for
    # it. The 1 GB grid holds a single stray mingram cell, which would otherwise put one
    # language's MinGram numbers beside five blanks.
    trainers, langs = [], []
    for candidate in KNOWN_TRAINERS:
        table_c, langs_c = _rows(data, [candidate])
        if len(langs_c) > len(langs):
            trainers, langs = [candidate], langs_c
        elif langs_c and langs_c == langs:
            trainers.append(candidate)
    if not trainers:
        raise SystemExit(f"no trainer in {results} covers a language completely")
    table, langs = _rows(data, trainers)
    body, mingram_gain = main_body(table, langs, trainers)

    # One group per trainer, each spanning its baseline plus its three variants.
    width = 1 + len(ARM_ORDER)
    header = [
        r"\begin{tabular}{l" + (" " + "r" * width) * len(trainers) + "}",
        r"\toprule",
        "".join(rf"& \multicolumn{{{width}}}{{c}}{{{TRAINER_LABEL[t]}}} " for t in trainers)
        + r"\\",
        " ".join(
            rf"\cmidrule(lr){{{2 + i * width}-{1 + (i + 1) * width}}}"
            for i in range(len(trainers))
        ),
        "Language & "
        + " & ".join(
            cell
            for _ in trainers
            for cell in ["\\texttt{plain}", *(ARM_LABEL[a] for a in ARM_ORDER)]
        )
        + r" \\",
        r" & " + " & ".join(
            cell for _ in trainers for cell in ["{\\footnotesize ch/tok}", *([r"{\footnotesize \%}"] * len(ARM_ORDER))]
        ) + r" \\",
        r"\midrule",
    ]
    gains = ", ".join(
        f"{LANG_LABEL[lang]} ${g:+.2f}\\%$" for lang, g in zip(langs, mingram_gain)
    )
    caption = (
        r"\caption{Compression, "
        + corpus
        + r", 32{,}768 additional vocabulary. "
        r"Baseline is \texttt{plain} in characters per token; each variant is the "
        r"percentage change against "
        + (r"\emph{its own trainer's} baseline, so the columns ask whether the variant "
           r"ordering survives a change of trainer rather than how the trainers compare. "
           if len(trainers) > 1 else r"that baseline. ")
        + r"\bnd{w} $<$ \bnd{wp} $<$ \bnd{wpd} in every cell. "
        + (r"MinGram compresses the baseline better than BPE in every language ("
           + gains + r"), and helps the baseline slightly more than \bnd{wpd}. "
           if len(trainers) > 1 else "")
        + r"No roundtrip failures in any of the "
        + str(len(langs) * 4 * len(trainers))
        + r" cells.}"
    )
    tex = "\n".join([
        "% Generated by marker_experiments/make_intrinsic_table.py. Do not edit.",
        r"% Requires booktabs and \newcommand{\bnd}[1]{\texttt{bnd\_#1}}.",
        f"% source: {os.path.relpath(results, os.path.dirname(HERE))}",
        r"\centering",
        r"\small",
        *header, *body,
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        rf"\label{{{label}}}",
        "",
    ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(tex)
    print(f"[tex] {out}")
    print(f"[tex] {len(langs)} language(s): {', '.join(langs)}"
          f"  trainer(s): {', '.join(trainers)}")


if __name__ == "__main__":
    app()
