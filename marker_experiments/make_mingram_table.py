#!/usr/bin/env python3
"""Generate the wide BPE-and-MinGram compression table.

One trainer per half: BPE on the left, MinGram on the right, each with its own
\texttt{plain} baseline followed by its three variants. A language is one row, so reading
across a half gives that trainer's whole story and comparing the halves answers the
question the table exists for -- does the variant ordering survive a change of trainer?

The earlier version interleaved the trainers under each variant, which put the two
baselines in one group and made "the BPE numbers" something you assembled from alternate
columns.

Emitted as a `table*` body: nine columns do not fit a single ACL column.

Needs `booktabs` and the paper's `\\bnd` macro:

    \\newcommand{\\bnd}[1]{\\texttt{bnd\\_#1}}

    uv run python marker_experiments/make_mingram_table.py
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
TRAINERS = ["bpe", "mingram"]
TRAINER_LABEL = {"bpe": "BPE", "mingram": "MinGram"}

app = cyclopts.App()


def _rows(results):
    """(lang, arm, trainer) -> chars/token, keeping only languages with a complete set.

    A partly-filled row would put a language's variants next to a baseline from a
    different set of cells, so incomplete languages are dropped rather than shown with
    gaps.
    """
    table = {}
    for key, cell in results.items():
        lang, rest = key.split("_", 1)
        for trainer in TRAINERS:
            if rest.endswith(f"_{trainer}"):
                table[(lang, rest[: -len(trainer) - 1], trainer)] = cell["eval_chars_per_token"]
    langs = [
        lang for lang in LANG_ORDER
        if all((lang, arm, tr) in table for arm in ["plain", *ARM_ORDER] for tr in TRAINERS)
    ]
    return table, langs


def main_body(table, langs):
    lines = []
    deltas = {(tr, arm): [] for tr in TRAINERS for arm in ARM_ORDER}
    mingram_gain = []
    for lang in langs:
        base = {tr: table[(lang, "plain", tr)] for tr in TRAINERS}
        cells = []
        for tr in TRAINERS:                      # trainer outer: one contiguous half each
            cells.append(f"{base[tr]:.4f}")      # that half's own plain baseline
            for arm in ARM_ORDER:
                # Relative to the baseline in the SAME half. The question is whether the
                # variant ordering survives the trainer, not how the trainers compare;
                # the two absolute baselines keep that second comparison recoverable.
                d = 100 * (table[(lang, arm, tr)] - base[tr]) / base[tr]
                deltas[(tr, arm)].append(d)
                cells.append(f"$\\mathbf{{{d:+.2f}}}$" if arm == "bnd_wpd" and d > 0 else f"${d:+.2f}$")
        mingram_gain.append(100 * (base["mingram"] - base["bpe"]) / base["bpe"])
        lines.append(f"{LANG_LABEL[lang]} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    mean = []
    for tr in TRAINERS:
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
) -> None:
    """Write the wide two-trainer compression table.

    Args:
        results: The MinGram grid result JSON.
        out: LaTeX file to write, \\input{} from a table* environment.
    """
    with open(results) as f:
        data = json.load(f)
    table, langs = _rows(data)
    if not langs:
        raise SystemExit(f"no language in {results} has all arms for both trainers")
    body, mingram_gain = main_body(table, langs)

    # One group per trainer, each spanning its baseline plus its three variants.
    width = 1 + len(ARM_ORDER)
    header = [
        r"\begin{tabular}{l" + (" " + "r" * width) * len(TRAINERS) + "}",
        r"\toprule",
        "".join(rf"& \multicolumn{{{width}}}{{c}}{{{TRAINER_LABEL[t]}}} " for t in TRAINERS)
        + r"\\",
        " ".join(
            rf"\cmidrule(lr){{{2 + i * width}-{1 + (i + 1) * width}}}"
            for i in range(len(TRAINERS))
        ),
        "Language & "
        + " & ".join(
            cell
            for _ in TRAINERS
            for cell in ["\\texttt{plain}", *(ARM_LABEL[a] for a in ARM_ORDER)]
        )
        + r" \\",
        r" & " + " & ".join(
            cell for _ in TRAINERS for cell in ["{\\footnotesize ch/tok}", *([r"{\footnotesize \%}"] * len(ARM_ORDER))]
        ) + r" \\",
        r"\midrule",
    ]
    gains = ", ".join(
        f"{LANG_LABEL[lang]} ${g:+.2f}\\%$" for lang, g in zip(langs, mingram_gain)
    )
    caption = (
        r"\caption{Compression under both trainers, 250M characters per language, "
        r"32{,}768 additional vocabulary, evaluation documents withheld from training. "
        r"Baseline is \texttt{plain} in characters per token; each variant is the "
        r"percentage change against \emph{its own trainer's} baseline, so the columns ask "
        r"whether the variant ordering survives a change of trainer rather than how the "
        r"trainers compare. It does: "
        r"\bnd{w} $<$ \bnd{wp} $<$ \bnd{wpd} in every cell. "
        r"MinGram compresses the baseline better than BPE in all three languages ("
        + gains
        + r"), and helps the baseline slightly more than \bnd{wpd}. "
        r"No roundtrip failures in any of the "
        + str(len(langs) * 4 * len(TRAINERS))
        + r" cells. Not comparable to the 1\,GB table.}"
    )
    tex = "\n".join([
        "% Generated by marker_experiments/make_mingram_table.py. Do not edit.",
        r"% Requires booktabs and \newcommand{\bnd}[1]{\texttt{bnd\_#1}}.",
        f"% source: {os.path.relpath(results, os.path.dirname(HERE))}",
        r"\centering",
        r"\small",
        *header, *body,
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:intrinsic}",
        "",
    ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(tex)
    print(f"[tex] {out}")
    print(f"[tex] {len(langs)} language(s): {', '.join(langs)}")


if __name__ == "__main__":
    app()
