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

Both corpus columns are a percentage change against the same trainer's `plain`, and both
are signed so that positive means better compression. On the evaluation corpus that is
the change in characters per token directly; on the training corpus, where the character
count is not recorded but is identical across arms of one language, it is the equivalent
change derived from the token counts.

Emitted as a `table*` body: thirteen columns do not fit a single ACL column.

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


def body(cells, langs, trainers, arms):
    """One block per trainer: the absolute `plain` anchors, then a row per variant."""
    lines, means = [], {}
    for i, tr in enumerate(trainers):
        # The mean covers the languages complete for the whole block, so every row in it
        # averages the same set and the rows stay comparable. Below two languages there is
        # nothing to average: the column would silently repeat a language already shown.
        block = [lg for lg in langs
                 if all((lg, arm, tr) in cells for arm in ["plain", *arms])]
        block = block if len(block) > 1 else []
        means[tr] = block
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + 2 * (len(langs) + 1)}}}{{l}}"
                     rf"{{\emph{{{TRAINER_LABEL[tr]}}}}} \\")

        anchors = []
        for lg in langs:
            base = cells.get((lg, "plain", tr))
            anchors.append(MISSING if base is None or "train_tokens" not in base
                           else f"{base['train_tokens'] / 1e9:.3f}")
            anchors.append(MISSING if base is None else f"{base['eval_chars_per_token']:.4f}")
        lines.append(f"{PLAIN_LABEL} & " + " & ".join([*anchors, MISSING, MISSING]) + r" \\")

        for arm in arms:
            row, totals = [], ([], [])
            for lg in langs:
                train, ev = deltas(cells, lg, arm, tr)
                row += [fmt(train), fmt(ev)]
                if lg in block:
                    for acc, value in zip(totals, (train, ev)):
                        if value is not None:
                            acc.append(value)
            row += [fmt(sum(a) / len(a)) if a else MISSING for a in totals]
            lines.append(f"{ARM_LABEL[arm]} & " + " & ".join(row) + r" \\")
    return lines, means


def mean_body(cells, langs, trainers, arms):
    """The main-paper layout: one row per scheme, one column pair per trainer, means only.

    The \\plainscheme row averages the absolute baselines, which no single language has but
    which gives the rest of the column its scale; per-language baselines are in the
    appendix table.
    """
    blocks = {tr: [lg for lg in langs
                   if all((lg, arm, tr) in cells for arm in ["plain", *arms])]
              for tr in trainers}
    blocks = {tr: block if len(block) > 1 else [] for tr, block in blocks.items()}

    anchors = []
    for tr in trainers:
        bases = [cells[(lg, "plain", tr)] for lg in blocks[tr]]
        for field, digits in (("train_tokens", 3), ("eval_chars_per_token", 4)):
            values = [b[field] for b in bases if field in b]
            scale = 1e9 if field == "train_tokens" else 1
            anchors.append(MISSING if not values
                           else f"{sum(values) / len(values) / scale:.{digits}f}")
    lines = [f"{PLAIN_LABEL} & " + " & ".join(anchors) + r" \\"]

    for arm in arms:
        row = []
        for tr in trainers:
            totals = ([], [])
            for lg in blocks[tr]:
                for acc, value in zip(totals, deltas(cells, lg, arm, tr)):
                    if value is not None:
                        acc.append(value)
            row += [fmt(sum(a) / len(a)) if a else MISSING for a in totals]
        lines.append(f"{ARM_LABEL[arm]} & " + " & ".join(row) + r" \\")
    return lines, blocks


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
    detail: str = "tab:intrinsic",
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
        anchor = (r"\plainscheme{} is the mean absolute baseline (training-corpus tokens "
                  r"in billions; evaluation characters per token), every other cell the "
                  r"mean percentage change against it, ")
    else:
        lines, means = body(cells, langs, trainers, arms)
        columns = [LANG_LABEL[lg] for lg in langs] + ["Mean"]
        anchor = (r"\plainscheme{} is absolute (training-corpus tokens in billions; "
                  r"evaluation characters per token), every other cell the percentage "
                  r"change against it within the same block, ")
    header = [
        r"\begin{tabular}{l" + " rr" * len(columns) + "}",
        r"\toprule",
        "Scheme " + "".join(rf"& \multicolumn{{2}}{{c}}{{{c}}} " for c in columns) + r"\\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(columns))),
        " & " + " & ".join([r"{\footnotesize train}", r"{\footnotesize eval}"] * len(columns))
        + r" \\",
        r"\midrule",
    ]
    covered = (
        "; ".join(f"{TRAINER_LABEL[tr]} {len(block)}"
                  for tr, block in means.items() if block)
        if any(means.values()) else None
    )
    # Only a variant that loses a round trip its own baseline keeps is this table's
    # business, and it never has. The shared counts are the encoding, not the markers.
    lossy = excess_roundtrip_failures(cells, langs, trainers, arms)
    caption = (
        r"\caption{Compression, " + setting + r". " + anchor
        + r"positive meaning better compression. "
        + (r"\bnds{w} $<$ \bnds{wp} $<$ \bnds{wpd} throughout. " if ordered else "")
        + (rf"Per-language numbers in Table~\ref{{{detail}}}. " if layout == "mean"
           else MISSING + r"~marks cells the grid has not reached. ")
        + (rf"Means over the languages complete for a trainer ({covered}). " if covered else "")
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
