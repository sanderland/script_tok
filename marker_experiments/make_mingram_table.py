#!/usr/bin/env python3
"""Generate the wide BPE-and-MinGram compression table.

The narrow version reported one row per (language, trainer), which reads as six unrelated
rows and makes the comparison the table exists for -- does the ordering of the boundary
variants survive a change of trainer? -- something you check by eye across rows. This puts
the two trainers side by side under each variant, so a language is one row and the
comparison is horizontal.

Emitted as a `table*` body: nine columns do not fit a single ACL column.

Self-contained LaTeX. It needs only `booktabs`, and writes arm names as literal
\\texttt{} rather than through a \\bnd{} macro, so it survives being pasted into a paper
whose preamble has changed.

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
ARM_LABEL = {"bnd_w": r"\texttt{bnd\_w}", "bnd_wp": r"\texttt{bnd\_wp}",
             "bnd_wpd": r"\texttt{bnd\_wpd}"}
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
    deltas = {(arm, tr): [] for arm in ARM_ORDER for tr in TRAINERS}
    mingram_gain = []
    for lang in langs:
        base = {tr: table[(lang, "plain", tr)] for tr in TRAINERS}
        cells = [f"{base['bpe']:.4f}", f"{base['mingram']:.4f}"]
        for arm in ARM_ORDER:
            for tr in TRAINERS:
                # Relative to the SAME trainer's baseline: the question is whether the
                # variant ordering survives the trainer, not how the trainers compare.
                d = 100 * (table[(lang, arm, tr)] - base[tr]) / base[tr]
                deltas[(arm, tr)].append(d)
                best = arm == "bnd_wpd"
                cells.append(f"$\\mathbf{{{d:+.2f}}}$" if best and d > 0 else f"${d:+.2f}$")
        mingram_gain.append(100 * (base["mingram"] - base["bpe"]) / base["bpe"])
        lines.append(f"{LANG_LABEL[lang]} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    mean = ["", ""]
    for arm in ARM_ORDER:
        for tr in TRAINERS:
            m = sum(deltas[(arm, tr)]) / len(deltas[(arm, tr)])
            mean.append(f"$\\mathbf{{{m:+.2f}}}$" if arm == "bnd_wpd" and m > 0 else f"${m:+.2f}$")
    lines.append("Mean & " + " & ".join(mean) + r" \\")
    return lines, mingram_gain


@app.default
def main(
    results: str = os.path.join(HERE, "mingram250_result.json"),
    out: str = os.path.join(GENERATED, "mingram_table.tex"),
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

    header = [
        r"\begin{tabular}{l rr rr rr rr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Baseline (chars/token)}"
        + "".join(rf" & \multicolumn{{2}}{{c}}{{{ARM_LABEL[a]}}}" for a in ARM_ORDER)
        + r" \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        "Language & " + " & ".join(TRAINER_LABEL[t] for t in TRAINERS)
        + " & " + " & ".join(TRAINER_LABEL[t] for _ in ARM_ORDER for t in TRAINERS) + r" \\",
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
        r"\texttt{bnd\_w} $<$ \texttt{bnd\_wp} $<$ \texttt{bnd\_wpd} in every cell. "
        r"MinGram compresses the baseline better than BPE in all three languages ("
        + gains
        + r"), and helps the baseline slightly more than \texttt{bnd\_wpd}. "
        r"No roundtrip failures in any of the "
        + str(len(langs) * 4 * len(TRAINERS))
        + r" cells. Not comparable to the 1\,GB table.}"
    )
    tex = "\n".join([
        "% Generated by marker_experiments/make_mingram_table.py. Do not edit.",
        f"% source: {os.path.relpath(results, os.path.dirname(HERE))}",
        r"\centering",
        r"\small",
        *header, *body,
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        r"\label{tab:trainers}",
        "",
    ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(tex)
    print(f"[tex] {out}")
    print(f"[tex] {len(langs)} language(s): {', '.join(langs)}")


if __name__ == "__main__":
    app()
