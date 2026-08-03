#!/usr/bin/env python3
r"""Generate a wide intrinsic-compression table from one grid's evaluation numbers.

One column group per trainer present in the data, each with its own \texttt{plain}
baseline followed by its variants. A language is one row, so reading across a group gives
that trainer's whole story and comparing groups answers the question two trainers are
there to answer -- does the variant ordering survive a change of trainer?

Two source formats, because the grids they come from are not the same experiment:

    marker_experiments/*_result.json           FineWiki grids, keys `<lang>_<arm>_<trainer>`,
                                               matched *additional* vocabulary (32,768)
    paper/generated/eval_goldfish.json         FineWeb 5 GB grid evaluated on held-out
                                               Goldfish, keys `fineweb_<lang>_5gb[_quick]_
                                               <arm>_<trainer>_v<vocab>`, matched *total*
                                               vocabulary

The format is detected from the keys. The FineWeb grid holds two corpora -- the reservoir
sample and the `_quick` read-until-full sample -- so it yields one table each, selected
with `--corpus`.

What is shown is read from the file, not assumed, because these grids fill in cell by cell
over days on several machines:

  * A language appears only if it has \texttt{plain} and every core variant for every
    trainer shown, so a table never puts one language's numbers beside another's blank.
  * A trainer earns a column group only if it covers exactly the languages shown.
  * An optional arm (the \texttt{\_caps} and \texttt{\_extcaps} forms) earns a column only
    if every language shown has it. Columns therefore appear as the grid fills, and a
    half-trained arm never silently drops rows.

Emitted as a `table*` body: nine columns do not fit a single ACL column.

Needs `booktabs` and the paper's `\bnd` macro:

    \newcommand{\bnd}[1]{\texttt{bnd\_#1}}

    uv run python marker_experiments/make_intrinsic_table.py
    uv run python marker_experiments/make_intrinsic_table.py --corpus quick
"""

import json
import os

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(HERE, "paper", "generated")

LANG_LABEL = {"en": "English", "de": "German", "fi": "Finnish",
              "ru": "Russian", "ar": "Arabic", "ko": "Korean"}
LANG_ORDER = ["en", "de", "fi", "ru", "ar", "ko"]

# The three boundary scopes every grid trains. A language without all of them is dropped.
CORE_ARMS = ["bnd_w", "bnd_wp", "bnd_wpd"]
# Case-handling forms, each shown next to the scope it modifies, and each only once every
# language shown has it. `_extcaps` is the canonical case form outside the markers,
# `_caps` the same thing inside them -- the placement ablation.
ARM_ORDER = ["bnd_w", "bnd_w_extcaps", "bnd_wp", "bnd_wp_extcaps",
             "bnd_wpd", "bnd_wpd_extcaps", "bnd_wpd_caps"]
ARM_LABEL = {arm: r"\bnd{" + arm.removeprefix("bnd_").replace("_", r"\_") + "}" for arm in ARM_ORDER}
KNOWN_TRAINERS = ["bpe", "mingram"]
TRAINER_LABEL = {"bpe": "BPE", "mingram": "MinGram"}

app = cyclopts.App()


def read_cells(data, corpus):
    """(lang, arm, trainer) -> chars/token, for one corpus variant.

    Handles both source formats. The FineWiki grids have one corpus per file and their
    keys carry no corpus at all, so `corpus` only selects among the FineWeb ones.
    """
    cells, meta = {}, {}
    for key, info in data.items():
        if key.startswith("_"):                        # `_note`, `_common`, ... metadata
            continue
        if key.startswith("fineweb_"):
            stem, _, _vocab = key.rpartition("_v")     # ..._<arm>_<trainer>_v<vocab>
            rest, _, trainer = stem.rpartition("_")
            lang = rest.split("_")[1]
            arm = rest.removeprefix(f"fineweb_{lang}_5gb_")
            quick = arm.startswith("quick_")
            if quick != (corpus == "quick"):
                continue
            arm = arm.removeprefix("quick_")
        else:                                          # <lang>_<arm>_<trainer>
            lang, _, rest = key.partition("_")
            arm, _, trainer = rest.rpartition("_")
        if trainer not in KNOWN_TRAINERS or lang not in LANG_LABEL:
            continue
        cells[(lang, arm, trainer)] = info["eval_chars_per_token"]
        meta[(lang, arm, trainer)] = info
    return cells, meta


def select(cells):
    """Which languages, trainers and arms the table can show without gaps.

    Trainers are taken greedily: the one covering the most languages sets the row set, and
    another trainer joins only if it covers exactly those languages. A trainer half way
    through the grid would otherwise cost every language it has not reached yet.
    """
    def complete(lang, trainer, arms):
        return all((lang, arm, trainer) in cells for arm in ["plain", *arms])

    trainers, langs = [], []
    for candidate in KNOWN_TRAINERS:
        covered = [lg for lg in LANG_ORDER if complete(lg, candidate, CORE_ARMS)]
        if len(covered) > len(langs):
            trainers, langs = [candidate], covered
        elif covered and covered == langs:
            trainers.append(candidate)
    arms = [
        arm for arm in ARM_ORDER
        if all((lg, arm, tr) in cells for lg in langs for tr in trainers)
    ]
    return langs, trainers, arms


def roundtrip_sentence(meta, langs, trainers, arms):
    """Report what was measured rather than assert zero.

    The FineWiki slices did round-trip cleanly, but a slice drawn from a different corpus
    need not, and a caption that claims zero failures while the numbers say otherwise is
    the one error this table must not make. A failure count that is the same for the
    baseline and every variant of a language is a property of that text under the base
    script encoding rather than an effect of the markers, so the two are separated: only
    the amount by which a variant exceeds its baseline is the markers' doing.
    """
    shown = [(lg, arm, tr) for lg in langs for tr in trainers for arm in ["plain", *arms]]
    if not max(meta[c]["roundtrip_failures"] for c in shown):
        return rf"No roundtrip failures in any of the {len(shown)} cells."
    shares = []
    for lg in langs:
        base = max(meta[(lg, "plain", tr)]["roundtrip_failures"] for tr in trainers)
        if base:
            docs = meta[(lg, "plain", trainers[0])]["eval_docs"]
            shares.append(f"{LANG_LABEL[lg]} {100 * base / docs:.2f}\\%")
    over = max(meta[c]["roundtrip_failures"] - meta[(c[0], "plain", c[2])]["roundtrip_failures"]
               for c in shown)
    return (
        r"The \texttt{plain} baseline itself fails to round-trip part of this slice ("
        + ", ".join(shares) + r" of documents), which is the base script encoding's "
        r"handling of that text rather than an effect of the markers; "
        + (r"no variant fails on more documents than its own baseline does."
           if over <= 0 else
           rf"the worst variant adds {over} failures over its own baseline.")
    )


def main_body(cells, langs, trainers, arms):
    lines = []
    deltas = {(tr, arm): [] for tr in trainers for arm in arms}
    mingram_gain = []
    for lang in langs:
        base = {tr: cells[(lang, "plain", tr)] for tr in trainers}
        row = []
        for tr in trainers:                      # trainer outer: one contiguous half each
            row.append(f"{base[tr]:.4f}")        # that half's own plain baseline
            for arm in arms:
                # Relative to the baseline in the SAME half. The question is whether the
                # variant ordering survives the trainer, not how the trainers compare;
                # the two absolute baselines keep that second comparison recoverable.
                d = 100 * (cells[(lang, arm, tr)] - base[tr]) / base[tr]
                deltas[(tr, arm)].append(d)
                row.append(f"$\\mathbf{{{d:+.2f}}}$" if d > 0 else f"${d:+.2f}$")
        if {"bpe", "mingram"} <= set(trainers):
            mingram_gain.append(100 * (base["mingram"] - base["bpe"]) / base["bpe"])
        lines.append(f"{LANG_LABEL[lang]} & " + " & ".join(row) + r" \\")

    lines.append(r"\midrule")
    mean = []
    for tr in trainers:
        mean.append("")                          # absolute baselines are not averaged
        for arm in arms:
            m = sum(deltas[(tr, arm)]) / len(deltas[(tr, arm)])
            mean.append(f"$\\mathbf{{{m:+.2f}}}$" if m > 0 else f"${m:+.2f}$")
    lines.append("Mean & " + " & ".join(mean) + r" \\")
    return lines, mingram_gain


@app.default
def main(
    results: str = os.path.join(GENERATED, "eval_goldfish.json"),
    out: str | None = None,
    corpus: str = "full",
    setting: str | None = None,
    label: str | None = None,
) -> None:
    """Write the wide compression table for one grid.

    Args:
        results: A grid result JSON or `eval_goldfish.json`.
        out: LaTeX file to write, \\input{} from a table* environment. Defaults to
            `table_intrinsic[_quick].tex` beside the other generated artifacts.
        corpus: `full` or `quick`, for sources holding both. `quick` is the
            read-until-full sample -- useful for iterating, not for reported results.
        setting: How the corpus and vocabulary are described in the caption. Defaults to
            the FineWeb grid's description.
        label: LaTeX label. Defaults to `tab:intrinsic[_quick]`.
    """
    if corpus not in ("full", "quick"):
        raise SystemExit(f"--corpus must be full or quick, not {corpus!r}")
    with open(results) as f:
        data = json.load(f)
    cells, meta = read_cells(data, corpus)
    langs, trainers, arms = select(cells)
    if not trainers:
        raise SystemExit(f"no trainer in {results} ({corpus}) covers a language completely")
    body, mingram_gain = main_body(cells, langs, trainers, arms)

    out = out or os.path.join(GENERATED, f"table_intrinsic{'_quick' if corpus == 'quick' else ''}.tex")
    label = label or f"tab:intrinsic{'-quick' if corpus == 'quick' else ''}"
    setting = setting or (
        "5\\,GB of FineWeb per language"
        + (" (the \\texttt{quick} read-until-full sample, which is not uniform over the "
           "source)" if corpus == "quick" else "")
        + ", evaluated on held-out Goldfish, 34{,}685 matched total vocabulary"
    )

    # One group per trainer, each spanning its baseline plus its variants.
    width = 1 + len(arms)
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
            cell for _ in trainers
            for cell in ["\\texttt{plain}", *(ARM_LABEL[a] for a in arms)]
        )
        + r" \\",
        r" & " + " & ".join(
            cell for _ in trainers
            for cell in ["{\\footnotesize ch/tok}", *([r"{\footnotesize \%}"] * len(arms))]
        ) + r" \\",
        r"\midrule",
    ]
    gains = ", ".join(
        f"{LANG_LABEL[lang]} ${g:+.2f}\\%$" for lang, g in zip(langs, mingram_gain)
    )
    ordered = all(
        cells[(lg, a, tr)] < cells[(lg, b, tr)]
        for lg in langs for tr in trainers
        for a, b in zip(CORE_ARMS, CORE_ARMS[1:])
    )
    caption = (
        r"\caption{Compression, " + setting + r". "
        r"Baseline is \texttt{plain} in characters per token; each variant is the "
        r"percentage change against "
        + (r"\emph{its own trainer's} baseline, so the columns ask whether the variant "
           r"ordering survives a change of trainer rather than how the trainers compare. "
           if len(trainers) > 1 else r"that baseline. ")
        + (r"\bnd{w} $<$ \bnd{wp} $<$ \bnd{wpd} in every cell. " if ordered else "")
        + (r"MinGram compresses the baseline better than BPE in every language ("
           + gains + r"). " if len(trainers) > 1 and all(g > 0 for g in mingram_gain) else "")
        + roundtrip_sentence(meta, langs, trainers, arms)
        + r"}"
    )
    tex = "\n".join([
        "% Generated by marker_experiments/make_intrinsic_table.py. Do not edit.",
        r"% Requires booktabs and \newcommand{\bnd}[1]{\texttt{bnd\_#1}}.",
        f"% source: {os.path.relpath(results, os.path.dirname(HERE))} ({corpus} corpus)",
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
          f"  trainer(s): {', '.join(trainers)}  arm(s): plain, {', '.join(arms)}")
    dropped = sorted({(lg, tr) for lg, _, tr in cells if lg not in langs or tr not in trainers})
    if dropped:
        print("[tex] incomplete, not shown: "
              + ", ".join(f"{lg}/{tr}" for lg, tr in dropped))


if __name__ == "__main__":
    app()
