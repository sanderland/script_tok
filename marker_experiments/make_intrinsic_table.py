#!/usr/bin/env python3
r"""Generate the intrinsic-compression table from one grid's measurements.

A scheme is a row, grouped into a block per trainer, and a language is a column pair --
its training corpus and its held-out evaluation corpus. Transposed relative to the older
layout, which put a language on a row: with two trainers and five schemes the grid is
taller than it is wide, and the transposed form has room for both corpora per language
rather than only the evaluation one.

Nothing is dropped for being incomplete. A cell the grid has not reached is `--`, so a
trainer that has only started still shows the languages it has, and the table fills in
place as the grid runs. This is a snapshot of a running experiment, not a finished one.

The source is `paper/generated/eval_goldfish.json`: the FineWeb 5 GB grid, matched total
vocabulary, evaluated on held-out Goldfish. It holds two corpora -- the reservoir sample
and the `_quick` read-until-full sample -- so it yields one table each, selected with
`--corpus`. The older FineWiki grid JSONs are no longer read; their tables are committed
under `paper/generated/` and stay as they are.

Both corpus columns are characters per token, and every non-baseline cell a percentage
change against the same trainer's `plain`, signed so that positive means better
compression. The evaluation figure is measured here; the training one divides the
sampler's own `selected_chars` by the token count the trainer left behind, so it costs
nothing to compute -- but the sample cache lives where the corpus was built, so a corpus
trained on another machine has percentages (characters cancel) and no absolute baseline.

Two layouts. `--layout languages` is the appendix table: every language, both corpora,
fifteen columns of `\tiny` in a `table*`. `--layout mean` is the main-paper one: means
only, with `n` for how many languages each covers, narrow enough for a single ACL column.

Needs `booktabs` and the paper's boundary macros. The `\textuparrow` key is the canonical
case form; it belongs to the `_extcaps` arms, which put it outside the markers and are the
ones expected to stay. The inside-the-markers placement is the ablation that is likely to
be dropped, so it takes the marked-up `in` key rather than the clean one:

    \colorlet{bndcol}{OliveGreen!75!black}
    \makeatletter
    \newcommand{\bnd@key}[1]{\@ifundefined{bnd@#1}{??#1??}{\@nameuse{bnd@#1}}}
    \@namedef{bnd@w}{w}
    \@namedef{bnd@wp}{w,p}
    \@namedef{bnd@wpd}{w,p,d}
    \@namedef{bnd@wcaps}{w,\textuparrow}
    \@namedef{bnd@wpcaps}{w,p,\textuparrow}
    \@namedef{bnd@wpdcaps}{w,p,d,\textuparrow}
    \@namedef{bnd@wpdcapsin}{w,p,d,\textuparrow\textsubscript{in}}
    % prose
    \newcommand{\bnd}[1]{\textcolor{bndcol}{{boundary[\bnd@key{#1}]}}}
    % tables
    \newcommand{\bnds}[1]{\textcolor{bndcol}{{[\bnd@key{#1}]}}}
    \makeatother
    \newcommand{\plainscheme}{\textcolor{bndcol}{plain}}

    uv run python marker_experiments/make_intrinsic_table.py
    uv run python marker_experiments/make_intrinsic_table.py --corpus quick
    uv run python marker_experiments/make_intrinsic_table.py --layout mean --corpus quick
"""

import json
import os

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(HERE, "paper", "generated")

LANG_LABEL = {"en": "English", "de": "German", "fi": "Finnish",
              "ru": "Russian", "ar": "Arabic", "ko": "Korean"}
LANG_ORDER = ["en", "de", "fi", "ru", "ar", "ko"]

# Every arm the grid trains, in display order: each boundary scope followed by its
# case-handling form. `_extcaps` puts the canonical case form outside the markers and is
# the form that stays, so it gets the plain `caps` notation; `bnd_wpd_caps` puts it inside
# them, exists only as the placement ablation, and is marked `in` until it is dropped.
ARM_ORDER = ["bnd_w", "bnd_w_extcaps", "bnd_wp", "bnd_wp_extcaps",
             "bnd_wpd", "bnd_wpd_extcaps", "bnd_wpd_caps"]
ARM_LABEL = {
    "bnd_w": r"\bnds{w}",
    "bnd_wp": r"\bnds{wp}",
    "bnd_wpd": r"\bnds{wpd}",
    "bnd_w_extcaps": r"\bnds{wcaps}",
    "bnd_wp_extcaps": r"\bnds{wpcaps}",
    "bnd_wpd_extcaps": r"\bnds{wpdcaps}",
    "bnd_wpd_caps": r"\bnds{wpdcapsin}",
}
PLAIN_LABEL = r"\plainscheme"
KNOWN_TRAINERS = ["bpe", "mingram"]
TRAINER_LABEL = {"bpe": "BPE", "mingram": "MinGram"}
MISSING = "--"

app = cyclopts.App()


def read_cells(data, corpus):
    """(lang, arm, trainer) -> measurement, for one corpus variant.

    Keys are `fineweb_<lang>_5gb[_quick]_<arm>_<trainer>_v<vocab>`, the naming
    `train_multilang.py` gives each cell.
    """
    cells = {}
    for key, info in data.items():
        stem, _, _vocab = key.rpartition("_v")         # ..._<arm>_<trainer>_v<vocab>
        rest, _, trainer = stem.rpartition("_")
        lang = rest.split("_")[1]
        arm = rest.removeprefix(f"fineweb_{lang}_5gb_")
        if arm.startswith("quick_") != (corpus == "quick"):
            continue
        arm = arm.removeprefix("quick_")
        if trainer not in KNOWN_TRAINERS or lang not in LANG_LABEL:
            continue
        cells[(lang, arm, trainer)] = info
    return cells


def select(cells):
    """Languages, trainers and schemes to show: everything the file has, in a fixed order.

    Incompleteness costs an entry, never a row or a column -- a trainer one language into
    the grid still earns its block, and the rest of that block is `--` until it fills.
    """
    langs = [lg for lg in LANG_ORDER if any(c[0] == lg for c in cells)]
    trainers = [tr for tr in KNOWN_TRAINERS if any(c[2] == tr for c in cells)]
    arms = [arm for arm in ARM_ORDER if any(c[1] == arm for c in cells)]
    return langs, trainers, arms


def deltas(cells, lang, arm, trainer):
    """(training, evaluation) percentage change against `plain`, positive = better.

    The training corpus records tokens rather than characters per token, but every arm of
    a language trains on the same text, so the character count cancels and the change in
    characters per token is exactly the inverse change in tokens.
    """
    base, cell = cells.get((lang, "plain", trainer)), cells.get((lang, arm, trainer))
    if base is None or cell is None:
        return None, None
    train = None
    if "train_tokens" in base and "train_tokens" in cell:
        train = 100 * (base["train_tokens"] / cell["train_tokens"] - 1)
    ev = 100 * (cell["eval_chars_per_token"] / base["eval_chars_per_token"] - 1)
    return train, ev


def fmt(value):
    return MISSING if value is None else (
        f"$\\mathbf{{{value:+.2f}}}$" if value > 0 else f"${value:+.2f}$")


def chars_per_token(cell, corpus):
    """Absolute characters per token on one corpus, or None where the count is missing.

    The evaluation figure is measured directly. The training one divides the corpus
    character count -- the sampler's own `selected_chars`, recorded per cell -- by the
    token count the trainer left behind, so it costs nothing to compute but is only
    available where that corpus was built.
    """
    if cell is None:
        return None
    if corpus == "eval":
        return cell["eval_chars_per_token"]
    return None if cell.get("train_chars") is None else cell["train_chars"] / cell["train_tokens"]


def fmt_chars_per_token(cell, corpus):
    value = chars_per_token(cell, corpus)
    return MISSING if value is None else f"{value:.4f}"


def row_mean(values):
    """Mean of the languages a row actually has, and which ones those were.

    Restricting instead to the languages complete across a whole block keeps rows exactly
    comparable, but in a grid that fills one cell at a time it empties the column: a single
    newly started arm is enough to leave no language complete. What each mean covers
    travels with it instead, so the caller can label it or withhold it.
    """
    present = [v for v in values if v is not None]
    return (sum(present) / len(present) if present else None), frozenset(
        i for i, v in enumerate(values) if v is not None)


def matched_means(pairs, digits=None):
    """The (train, eval) means of a row, with train withheld unless it covers the same
    languages as eval.

    Only the appendix layout uses this. It has no room for a count column, so a train mean
    over one language sitting beside an evaluation mean over six -- which is what the
    unbuilt training corpora produce here -- would read as a comparison and is not one.
    """
    (train, train_set), (ev, ev_set) = (row_mean([p[i] for p in pairs]) for i in (0, 1))
    if train_set != ev_set:
        train = None
    if digits is None:
        return [fmt(train), fmt(ev)]
    return [MISSING if v is None else f"{v:.{digits}f}" for v in (train, ev)]


def body(cells, langs, trainers, arms):
    """One block per trainer: the absolute `plain` anchors, then a row per variant."""
    lines, means = [], {}
    for i, tr in enumerate(trainers):
        means[tr] = [lg for lg in langs if (lg, "plain", tr) in cells]
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + 2 * (len(langs) + 1)}}}{{l}}"  # noqa: E501
                     rf"{{\emph{{{TRAINER_LABEL[tr]}}}}} \\")

        anchors, absolute = [], []
        for lg in langs:
            base = cells.get((lg, "plain", tr))
            anchors += [fmt_chars_per_token(base, c) for c in ("train", "eval")]
            absolute.append(tuple(chars_per_token(base, c) for c in ("train", "eval")))
        anchors += matched_means(absolute, digits=4)
        lines.append(f"{PLAIN_LABEL} & " + " & ".join(anchors) + r" \\")

        for arm in arms:
            pairs = [deltas(cells, lg, arm, tr) for lg in langs]
            row = [fmt(value) for pair in pairs for value in pair]
            lines.append(f"{ARM_LABEL[arm]} & " + " & ".join(row + matched_means(pairs))
                         + r" \\")
    return lines, means


def mean_body(cells, langs, trainers, arms):
    """The main-paper layout: one row per scheme, one column pair per trainer, means only.

    The \\plainscheme row averages the absolute baselines, which no single language has but
    which gives the rest of the column its scale; per-language baselines are in the
    appendix table.
    """
    anchors = []
    for tr in trainers:
        bases = [cells.get((lg, "plain", tr)) for lg in langs]
        counts = set()
        for corpus in ("train", "eval"):
            mean, covered = row_mean([chars_per_token(b, corpus) for b in bases])
            anchors.append(MISSING if mean is None else f"{mean:.4f}")
            counts.add(len(covered))
        anchors.append(str(max(counts)))
    lines = [f"{PLAIN_LABEL} & " + " & ".join(anchors) + r" \\"]

    for arm in arms:
        row = []
        for tr in trainers:
            pairs = [deltas(cells, lg, arm, tr) for lg in langs]
            counts = set()
            for i in (0, 1):
                mean, covered = row_mean([p[i] for p in pairs])
                row.append(fmt(mean))
                counts.add(len(covered))
            row.append(str(max(counts)))
        lines.append(f"{ARM_LABEL[arm]} & " + " & ".join(row) + r" \\")
    return lines, {tr: [lg for lg in langs if (lg, "plain", tr) in cells] for tr in trainers}


def excess_roundtrip_failures(cells, langs, trainers, arms):
    """Documents a variant fails to round-trip that its own baseline does not.

    Not a caption's business, and not reported in one: the raw counts are per cell in the
    source JSON. What matters to a table of markers is only whether a marker scheme costs
    a round-trip the baseline kept, and that is one number -- normally zero. The shared
    part is the base script encoding on that text (on the Goldfish slice, Arabic diacritic
    sequences come back canonically reordered under every arm alike), which is a property
    of SCRIPT-v3 rather than of anything this table varies.
    """
    return max(
        (cells[(lg, arm, tr)]["roundtrip_failures"] - cells[(lg, "plain", tr)]["roundtrip_failures"]
         for lg in langs for tr in trainers for arm in arms
         if (lg, arm, tr) in cells and (lg, "plain", tr) in cells),
        default=0,
    )


@app.default
def main(
    results: str = os.path.join(GENERATED, "eval_goldfish.json"),
    out: str | None = None,
    corpus: str = "full",
    layout: str = "languages",
    setting: str | None = None,
    label: str | None = None,
    detail: str | None = None,
) -> None:
    """Write the compression table for one grid.

    Args:
        results: `eval_goldfish.json`, or any file with the same per-cell keys.
        out: LaTeX file to write, \\input{} from a table* environment. Defaults to
            `table_intrinsic[_quick|_main].tex` beside the other generated artifacts.
        corpus: `full` or `quick`, for sources holding both. `quick` is the
            read-until-full sample -- useful for iterating, not for reported results.
        layout: `languages` for the per-language appendix table, or `mean` for the
            main-paper one: means only, four columns, fits a single ACL column.
        setting: How the corpora and vocabulary are described in the caption. Defaults to
            the FineWeb grid's description.
        label: LaTeX label. Defaults to `tab:intrinsic[-quick|-main]`.
        detail: Label the `mean` layout points at for the per-language numbers.
    """
    if corpus not in ("full", "quick"):
        raise SystemExit(f"--corpus must be full or quick, not {corpus!r}")
    if layout not in ("languages", "mean"):
        raise SystemExit(f"--layout must be languages or mean, not {layout!r}")
    with open(results) as f:
        data = json.load(f)
    cells = read_cells(data, corpus)
    langs, trainers, arms = select(cells)
    if not langs:
        raise SystemExit(f"no usable cells in {results} ({corpus} corpus)")

    suffix = "_main" if layout == "mean" else ("_quick" if corpus == "quick" else "")
    out = out or os.path.join(GENERATED, f"table_intrinsic{suffix}.tex")
    label = label or f"tab:intrinsic{suffix.replace('_', '-')}"
    # The per-language table of the same corpus, which is where the main table's reader
    # goes for the numbers behind a mean.
    detail = detail or f"tab:intrinsic{'-quick' if corpus == 'quick' else ''}"
    setting = setting or (
        "trained on 5\\,GB of FineWeb per language"
        + (" (the \\texttt{quick} read-until-full sample, which is not uniform over the "
           "source)" if corpus == "quick" else "")
        + ", 34{,}685 matched total vocabulary, evaluated on held-out Goldfish"
    )
    ordered = all(
        (d := [deltas(cells, lg, arm, tr)[1] for arm in ["bnd_w", "bnd_wp", "bnd_wpd"]])
        and None not in d and d == sorted(d)
        for lg in langs for tr in trainers if (lg, "plain", tr) in cells
    )

    if layout == "mean":
        lines, means = mean_body(cells, langs, trainers, arms)
        columns = [TRAINER_LABEL[tr] for tr in trainers]
        # A third column per trainer: how many languages went into that row's mean.
        subheads = [r"{\footnotesize train}", r"{\footnotesize eval}", r"{\footnotesize $n$}"]
        anchor = (r"\plainscheme{} is the mean absolute baseline in characters per token, "
                  r"every other cell the mean percentage change against it, ")
    else:
        lines, means = body(cells, langs, trainers, arms)
        columns = [LANG_LABEL[lg] for lg in langs] + ["Mean"]
        subheads = [r"{\footnotesize train}", r"{\footnotesize eval}"]
        anchor = (r"\plainscheme{} is absolute characters per token on each corpus, every "
                  r"other cell the percentage change against it within the same block, ")
    width = len(subheads)
    header = [
        r"\begin{tabular}{l" + (" " + "r" * width) * len(columns) + "}",
        r"\toprule",
        "Scheme " + "".join(rf"& \multicolumn{{{width}}}{{c}}{{{c}}} " for c in columns) + r"\\",
        " ".join(rf"\cmidrule(lr){{{2 + width * i}-{1 + width * (i + 1)}}}"
                 for i in range(len(columns))),
        " & " + " & ".join(subheads * len(columns)) + r" \\",
        r"\midrule",
    ]
    # Only a variant that loses a round trip its own baseline keeps is this table's
    # business, and it never has. The shared counts are the encoding, not the markers.
    lossy = excess_roundtrip_failures(cells, langs, trainers, arms)
    caption = (
        r"\caption{Compression, " + setting + r". " + anchor
        + r"positive meaning better compression. "
        + (r"\bnds{w} $<$ \bnds{wp} $<$ \bnds{wpd} throughout. " if ordered else "")
        + (rf"Per-language numbers in Table~\ref{{{detail}}}. " if layout == "mean"
           else MISSING + r"~marks cells the grid has not reached. ")
        + (r"Each mean covers the languages that have that scheme; $n$ is how many. "
           if layout == "mean" else
           r"Means cover the languages present in that row, and a train mean is shown "
           r"only where it covers the same ones as the eval mean beside it. ")
        + (rf"{lossy} documents fail to round-trip under a variant but not under its "
           rf"baseline. " if lossy > 0 else "")
    ).rstrip() + r"}"
    tex = "\n".join([
        "% Generated by marker_experiments/make_intrinsic_table.py. Do not edit.",
        r"% Requires booktabs and the paper's \bnds and \plainscheme macros.",
        f"% source: {os.path.relpath(results, os.path.dirname(HERE))} ({corpus} corpus)",
        r"\centering",
        # Thirteen columns of a per-language table need \tiny to fit the page width; the
        # main table has four and stays readable.
        r"\tiny" if layout == "languages" else r"\small",
        *header, *lines,
        r"\bottomrule",
        r"\end{tabular}",
        caption,
        rf"\label{{{label}}}",
        "",
    ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(tex)
    total = len(langs) * len(trainers) * (1 + len(arms))
    filled = sum(1 for lg in langs for tr in trainers for arm in ["plain", *arms]
                 if (lg, arm, tr) in cells)
    print(f"[tex] {out}")
    print(f"[tex] {filled}/{total} cells: {len(langs)} language(s), "
          f"trainer(s) {', '.join(trainers)}, scheme(s) plain, {', '.join(arms)}")


if __name__ == "__main__":
    app()
