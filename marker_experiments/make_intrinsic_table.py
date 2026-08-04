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
fifteen columns of `\tiny` in a `table*`. `--layout mean` is the main-paper one: the
compression means beside MorphScore, from `morphscore.json`, under both of its settings for
words the tokenizer leaves whole. MorphScore covers English alone rather than the full
grid: gold exists for five of the six languages, but only the English gold is trusted.

In both layouts the best cell in a **column** is bold and the runner-up underlined, ties
sharing a place. The column is the comparison the table is for: one scheme against the
others, everything else held fixed. Nothing is emphasised along a row -- a row spans
languages whose absolute compression differs by a factor of two.

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
from collections import namedtuple

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(HERE, "paper", "generated")
MORPHSCORE = os.path.join(GENERATED, "morphscore.json")

LANG_LABEL = {"en": "English", "de": "German", "fi": "Finnish",
              "ru": "Russian", "ar": "Arabic", "ko": "Korean"}
LANG_ORDER = ["en", "de", "fi", "ru", "ar", "ko"]

# Every arm the grid trains, in display order: each boundary scope followed by its
# case-handling form. `_extcaps` puts the canonical case form outside the markers and is
# the form that stays, so it gets the plain `caps` notation; `bnd_wpd_caps` puts it inside
# them and exists only as the placement ablation.
# The case-code arm is `_extcapsfix`, which tests `istitle() or isupper()` before coding a
# span. `_extcaps` skipped that test, so it coded every span of a script with no case at
# all -- 33,183 of 33,579 Arabic word spans -- and cost Arabic 2.3% of its tokens. It is
# the same scheme wherever case exists, and the two are identical to the digit on en, de,
# fi and ru, so `_extcapsfix` carries the plain `caps` notation and `_extcaps` is not
# reported. `bnd_w_extcaps` and `bnd_wp_extcaps` still predate the fix and are still wrong
# on Arabic and Korean; they stay only until their fixed cells are trained.
ARM_ORDER = ["bnd_w", "bnd_w_extcaps", "bnd_wp", "bnd_wp_extcaps",
             "bnd_wpd", "bnd_wpd_extcapsfix"]
ARM_LABEL = {
    "bnd_w": r"\bnds{w}",
    "bnd_wp": r"\bnds{wp}",
    "bnd_wpd": r"\bnds{wpd}",
    "bnd_w_extcaps": r"\bnds{wcaps}",
    "bnd_wp_extcaps": r"\bnds{wpcaps}",
    "bnd_wpd_extcapsfix": r"\bnds{wpdcaps}",
}
# What the main table argues: the three boundary scopes, and case handling once. Case
# handling at the narrower scopes answers a question the scopes do not raise and says
# nothing the appendix does not. Kept out of the main table, kept in the appendix one.
MAIN_ARMS = ["bnd_w", "bnd_wp", "bnd_wpd", "bnd_wpd_extcapsfix"]
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


# A rendered cell, kept apart from its own emphasis: which cells are bold or underlined is
# a property of the column they sit in, so it cannot be decided while the cell is built.
# A cell is one or more parts, each a (text, rank) pair; the first is the cell's own
# figure and any further part is a parenthesised alternative measurement of the same thing.
# `rank` of None keeps a part out of the ranking, which is what the parenthesised figures
# take: they are measured under a different probe from the rest of the column, so a bold
# mark landing on one says "best in this column" about a number the column is not
# comparable to. They are set in grey instead, and the marks rank the primary figures only.
Cell = namedtuple("Cell", "parts math")
GREY = "black!55"


def cell(text, rank=None, extra=None, math=False):
    return Cell(((text, rank),) + ((extra,) if extra else ()), math)


MISSING_CELL = cell(MISSING)


def render(c, marks):
    """One cell as LaTeX. Each mark is `best`, `second` or None, one per part."""
    out = []
    for i, ((text, _), mark) in enumerate(zip(c.parts, marks)):
        if mark == "best":
            text = rf"\{'mathbf' if c.math else 'textbf'}{{{text}}}"
        elif mark == "second":
            text = rf"\underline{{{text}}}"
        out.append(text if i == 0 else rf"\,\textcolor{{{GREY}}}{{({text})}}")
    joined = "".join(out)
    return f"${joined}$" if c.math else joined


def emphasise(rows):
    """Bold the best and underline the runner-up in each column of one block.

    By column, because that is the comparison the table exists to support: one scheme
    against the others on the same language, corpus and trainer. A row spans languages
    whose absolute compression differs by a factor of two and metrics on unrelated scales,
    so ranking across one says nothing.

    Ties share a mark and consume a place, so two cells tied for best leave no runner-up.
    That is deliberate -- promoting the third-best would claim a separation the numbers do
    not have.
    """
    marks = [[[None] * len(c.parts) for c in cells] for _, cells in rows]
    for col in range(max((len(cells) for _, cells in rows), default=0)):
        ranked = [(part[1], i, j)
                  for i, (_, cells) in enumerate(rows) if col < len(cells)
                  for j, part in enumerate(cells[col].parts) if part[1] is not None]
        best = sorted({r for r, _, _ in ranked}, reverse=True)[:2]
        for value, i, j in ranked:
            if value in best:
                marks[i][col][j] = "best" if value == best[0] else "second"
    return [(label, [render(c, m) for c, m in zip(cells, row_marks)])
            for (label, cells), row_marks in zip(rows, marks)]


def chars_per_token(entry, corpus):
    """Absolute characters per token on one corpus, or None where the count is missing.

    The evaluation figure is measured directly. The training one divides the corpus
    character count -- the sampler's own `selected_chars`, recorded per cell -- by the
    token count the trainer left behind, so it costs nothing to compute but is only
    available where that corpus was built.
    """
    if entry is None:
        return None
    if corpus == "eval":
        return entry["eval_chars_per_token"]
    return None if entry.get("train_chars") is None else entry["train_chars"] / entry["train_tokens"]


def absolute_cell(value, digits=4, rank=None):
    """The \\plainscheme anchor: printed absolute, but ranked as the zero it is.

    Every other cell in the column is a percentage against this one, so the baseline sits
    at exactly 0 and belongs in the ranking at that value. Leaving it out marked the least
    bad variant as best in columns where nothing beat the baseline at all -- which is most
    of them.
    """
    return MISSING_CELL if value is None else cell(f"{value:.{digits}f}", rank)


def delta_cell(value, digits=2):
    # Ranked on the rounded value, not the raw one: two cells printed as $-1.67$ are the
    # same number as far as the page is concerned, and marking one of them better claims a
    # separation the reader cannot see.
    return MISSING_CELL if value is None else cell(
        f"{value:+.{digits}f}", round(value, digits), math=True)


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


def matched_means(pairs, absolute=False, digits=3):
    """The (train, eval) means of a row, with train withheld unless it covers the same
    languages as eval.

    Only the appendix layout uses this. It has no room for a count column, so a train mean
    over one language sitting beside an evaluation mean over six -- which is what the
    unbuilt training corpora produce here -- would read as a comparison and is not one.
    """
    (train, train_set), (ev, ev_set) = (row_mean([p[i] for p in pairs]) for i in (0, 1))
    if train_set != ev_set:
        train = None
    if absolute:
        return [absolute_cell(train, digits, rank=0.0), absolute_cell(ev, digits, rank=0.0)]
    return [delta_cell(train), delta_cell(ev)]


def morph_scores(path, statistic):
    """(lang, arm, trainer, in_context) -> score, from one morphology metric's result file.

    Both files carry the two probes side by side, the in-context one keyed with a `@ctx`
    suffix, so a cell knows which of the two it is.
    """
    with open(path) as f:
        data = json.load(f)
    return {(v["lang"], v["arm"], v["trainer"], key.endswith("@ctx")): statistic(v)
            for key, v in data.items()}


def morphscore_statistic(record, field="credit_single_tok"):
    """MorphScore F$_1$ under one of the metric's two settings for unsplit words.

    The harmonic mean of the two macro means, which is how `morphscore.py` forms its own
    `macro_f1`. Recall alone would reward a scheme for splitting everywhere, and the
    marker schemes split less than the baseline, not more.

    `credit_single_tok` scores every gold word, giving a word the tokenizer kept whole a
    point; `exclude_single_tok` drops those words and scores only the ones it split. The
    switch decides the sign of this table's morphology comparison, because the first
    setting is dominated by how often a word is one token -- across the 118 measured cells
    it tracks the single-token share at r = 0.95 -- and keeping words whole is exactly what
    a boundary marker buys. Both are therefore reported rather than one being chosen.
    """
    d = record[field]
    r, p = d["recall"], d["precision"]
    return 2 * r * p / (r + p) if r + p else 0.0


def morphscore_split_statistic(record):
    """MorphScore over the words the tokenizer actually split."""
    return morphscore_statistic(record, "exclude_single_tok")


def morph_cell(scores, langs, arm, trainer, digits=3):
    """One morphology cell: the metric as published, trailed by the in-context variant.

    The headline figure segments each gold word on its own, which is how the metric is
    computed everywhere it has been reported. That makes it the number that compares to the
    literature, so it is the one shown and the one ranked. Segmenting inside a carrier
    phrase is ours; it follows in grey parentheses where it differs, which is only for
    \\plainscheme -- the marker schemes delimit a span the same way either side, so the two
    agree for them and the cell shows one number. It is deliberately unranked: it is the
    only cell in its column measured that way, so a mark on it would compare it against
    numbers from the other probe.
    """
    def mean(in_context):
        got = [scores[(lg, arm, trainer, in_context)] for lg in langs
               if (lg, arm, trainer, in_context) in scores]
        return sum(got) / len(got) if got else None

    bare, ctx = mean(False), mean(True)
    if bare is None:
        return MISSING_CELL
    shown = f"{bare:.{digits}f}"
    if ctx is None or f"{ctx:.{digits}f}" == shown:
        return cell(shown, round(bare, digits))
    return cell(shown, round(bare, digits), extra=(f"{ctx:.{digits}f}", None))


def body(cells, langs, trainers, arms):
    """One block per trainer: the absolute `plain` anchors, then a row per variant."""
    blocks = []
    for tr in trainers:
        rows = []
        anchors, absolute = [], []
        for lg in langs:
            base = cells.get((lg, "plain", tr))
            pair = tuple(chars_per_token(base, c) for c in ("train", "eval"))
            # Three decimals, not four: the widest cell sets the column width, and in most
            # of these columns that is this row rather than a two-digit percentage below
            # it. The fourth digit is worth less than the page width it costs, and four
            # significant figures still reconstruct any arm from its percentage.
            anchors += [absolute_cell(v, 3, rank=0.0) for v in pair]
            absolute.append(pair)
        rows.append((PLAIN_LABEL, anchors + matched_means(absolute, absolute=True)))

        for arm in arms:
            pairs = [deltas(cells, lg, arm, tr) for lg in langs]
            row = [delta_cell(value) for pair in pairs for value in pair]
            rows.append((ARM_LABEL[arm], row + matched_means(pairs)))
        blocks.append((TRAINER_LABEL[tr], rows))
    return blocks


def mean_body(cells, langs, trainers, arms, score_langs):
    """The main-paper layout: one row per scheme, four columns per trainer, means only.

    Compression is the mean over every language the grid has. MorphScore is not: gold exists
    for five of them, but the non-English segmentations are not trusted -- a baseline scoring
    0.70 on English and 0.21 on German says more about the German gold than about the
    tokenizer -- so it is reported on English alone.

    The \\plainscheme row's compression is the absolute baseline, which no single language
    has but which gives the rest of the column its scale; its morphology cells are real
    scores on the same footing as every other row, and are ranked with them.
    """
    # Two decimals on the excluding column, three on the crediting one. It runs an order of
    # magnitude lower, on the minority of words a marker scheme splits at all -- a
    # thousand-odd English words -- and a third decimal there is smaller than the shift from
    # one word leaving the scored set, which is all that separates its two probes.
    sources = [(morph_scores(MORPHSCORE, morphscore_statistic), score_langs, 3),
               (morph_scores(MORPHSCORE, morphscore_split_statistic), score_langs, 2)]

    rows = []
    anchors = []
    for tr in trainers:
        bases = [cells.get((lg, "plain", tr)) for lg in langs]
        for corpus in ("train", "eval"):
            # Two decimals here, four in the appendix: this row is the scale the
            # percentages below it are read against, not the number anyone reproduces
            # against, and the fourth digit is a hundredth of the smallest delta in the
            # column.
            anchors.append(absolute_cell(
                row_mean([chars_per_token(b, corpus) for b in bases])[0], digits=2, rank=0.0))
        # Two decimals on this row alone: it is the only one carrying a second figure, and
        # a third digit on both halves makes it much the widest cell in the table.
        anchors += [morph_cell(src, lgs, "plain", tr, digits=2) for src, lgs, _ in sources]
    rows.append((PLAIN_LABEL, anchors))

    for arm in arms:
        row = []
        for tr in trainers:
            pairs = [deltas(cells, lg, arm, tr) for lg in langs]
            row += [delta_cell(row_mean([p[i] for p in pairs])[0]) for i in (0, 1)]
            row += [morph_cell(src, lgs, arm, tr, digits) for src, lgs, digits in sources]
        rows.append((ARM_LABEL[arm], row))
    return [(None, rows)]


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
    morphscore_langs: str = "en",
    arms: str | None = None,
) -> None:
    """Write the compression table for one grid.

    Args:
        results: `eval_goldfish.json`, or any file with the same per-cell keys.
        out: LaTeX file to write, \\input{} from a table* environment. Defaults to
            `table_intrinsic[_quick|_main].tex` beside the other generated artifacts.
        corpus: `full` or `quick`, for sources holding both. `quick` is the
            read-until-full sample -- useful for iterating, not for reported results.
        layout: `languages` for the per-language appendix table, or `mean` for the
            main-paper one: means and MorphScore.
        setting: How the corpora and vocabulary are described in the caption. Defaults to
            the FineWeb grid's description.
        label: LaTeX label. Defaults to `tab:intrinsic[-quick|-main]`.
        detail: Label the `mean` layout points at for the per-language numbers.
        morphscore_langs: Comma-separated languages the MorphScore columns average. Gold
            exists for en, de, fi, ru and ko, but only English is reported: the baseline
            scores 0.70 there and 0.21 on German, which is a statement about the German
            gold rather than about the tokenizer.
        arms: Comma-separated schemes to show. Defaults to everything the source has for
            the appendix layout and to MAIN_ARMS for the main one.
    """
    if corpus not in ("full", "quick"):
        raise SystemExit(f"--corpus must be full or quick, not {corpus!r}")
    if layout not in ("languages", "mean"):
        raise SystemExit(f"--layout must be languages or mean, not {layout!r}")
    with open(results) as f:
        data = json.load(f)
    cells = read_cells(data, corpus)
    langs, trainers, available = select(cells)
    if not langs:
        raise SystemExit(f"no usable cells in {results} ({corpus} corpus)")
    wanted = ([x.strip() for x in arms.split(",") if x.strip()] if arms
              else MAIN_ARMS if layout == "mean" else available)
    if unknown := [a for a in wanted if a not in ARM_LABEL]:
        raise SystemExit(f"unknown scheme(s): {', '.join(unknown)}")
    arms = [a for a in ARM_ORDER if a in wanted and a in available]

    suffix = "_main" if layout == "mean" else ("_quick" if corpus == "quick" else "")
    out = out or os.path.join(GENERATED, f"table_intrinsic{suffix}.tex")
    label = label or f"tab:intrinsic{suffix.replace('_', '-')}"
    # The per-language table of the same corpus, which is where the main table's reader
    # goes for the numbers behind a mean.
    detail = detail or f"tab:intrinsic{'-quick' if corpus == 'quick' else ''}"
    setting = setting or (
        "trained on 5\\,GB of FineWeb per language"
        + (" (\\texttt{quick} non-uniform sample)" if corpus == "quick" else "")
        # The grid matches the *total* at 34,685, which leaves the learned counts differing
        # by at most three (32,975 for plain, 32,974 with a marker, 32,972 with the case
        # codes) as the atomic alphabet grows. That is roundoff, so the caption says the
        # round number the paper says.
        + ", 32k matched learned tokens, evaluated on held-out Goldfish"
    )

    # What a mean is over, stated only when it is the same everywhere -- a caption is not
    # the place to narrate which cells a running grid has not filled yet, and a table with
    # a ragged mean should not be the one that goes in the paper.
    complete = all((lg, arm, tr) in cells
                   for lg in langs for tr in trainers for arm in ["plain", *arms])
    spelled = {4: "four", 5: "five", 6: "six"}.get(len(langs), str(len(langs)))
    coverage = (rf" averaged over the {spelled} languages"
                if complete and layout == "mean" else "")

    split = lambda v: [x.strip() for x in v.split(",") if x.strip()]  # noqa: E731
    score_langs = split(morphscore_langs)
    if layout == "mean":
        blocks = mean_body(cells, langs, trainers, arms, score_langs)
        columns = [TRAINER_LABEL[tr] for tr in trainers]
        # A middle header tier naming the metric, because `train` and `eval` alone name the
        # corpus and not the quantity, and because MorphScore's two settings have to read as
        # one metric measured twice rather than as two metrics.
        groups = [("Compression", 2), ("MorphScore", 2)]
        subheads = ["train", "eval", "credit", "exclude"]
        anchor = (r"\plainscheme{} compression is characters per token" + coverage
                  + r", every other compression cell the percentage change against it, ")
    else:
        blocks = body(cells, langs, trainers, arms)
        columns = [LANG_LABEL[lg] for lg in langs] + ["Mean"]
        groups = None
        subheads = ["train", "eval"]
        anchor = (r"\plainscheme{} is absolute characters per token on each corpus, every "
                  r"other cell the percentage change against it within the same block, ")
    width = len(subheads)
    span = 1 + width * len(columns)
    lines = []
    for i, (heading, rows) in enumerate(blocks):
        if heading is not None:
            if i:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{span}}}{{l}}{{\emph{{{heading}}}}} \\")
        # Ranked inside the block, since a delta is against that block's own baseline.
        lines += [f"{label_} & " + " & ".join(cells_) + r" \\"
                  for label_, cells_ in emphasise(rows)]
    def rule(spans):
        """A \\cmidrule under each group, from a list of (start column, width)."""
        return " ".join(rf"\cmidrule(lr){{{a}-{a + w - 1}}}" for a, w in spans)

    header = [
        r"\begin{tabular}{l" + (" " + "r" * width) * len(columns) + "}",
        r"\toprule",
        "Scheme " + "".join(rf"& \multicolumn{{{width}}}{{c}}{{{c}}} " for c in columns) + r"\\",
        rule([(2 + width * i, width) for i in range(len(columns))]),
    ]
    if groups:
        header.append(" " + "".join(
            rf"& \multicolumn{{{w}}}{{c}}{{{name}}} " for _ in columns for name, w in groups
        ) + r"\\")
        starts, at = [], 2
        for _ in columns:
            for _name, w in groups:
                starts.append((at, w))
                at += w
        header.append(rule(starts))
    header += [
        " & " + " & ".join(subheads * len(columns)) + r" \\",
        r"\midrule",
    ]
    # Only a variant that loses a round trip its own baseline keeps is this table's
    # business, and it never has. The shared counts are the encoding, not the markers.
    lossy = excess_roundtrip_failures(cells, langs, trainers, arms)
    def listed(names):
        labels = [LANG_LABEL[lg] for lg in names]
        return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + f" and {labels[-1]}"

    morph_note = (
        rf"MorphScore is over {listed(score_langs)}, higher better, under both of its "
        r"settings for words the tokenizer leaves whole: \emph{credit} scores them as "
        r"correct, \emph{exclude} drops them and scores only the words that were split. "
        r"For \plainscheme{}, both are also shown in grey with the gold word segmented "
        r"with a leading space, which significantly affects scores. "
    )
    caption = (
        r"\caption{" + (r"Intrinsic evaluation results. " if layout == "mean"
                        else r"Compression, " + setting + r". ") + anchor
        + r"higher is better. "
        + (morph_note if layout == "mean" else "")
        + r"\textbf{Bold} is best in a column and \underline{underline} runner-up, "
        + (r"grey figures excluded, " if layout == "mean" else "")
        + r"counting \plainscheme{} as zero in the compression columns. "
        + (r"Per-language numbers in \Cref{app:allthetokenizers}. "
           if layout == "mean" else "")
        + (rf"{lossy} documents fail to round-trip under a variant but not under its "
           rf"baseline. " if lossy > 0 else "")
    ).rstrip() + r"}"
    tex = "\n".join([
        "% Generated by marker_experiments/make_intrinsic_table.py. Do not edit.",
        r"% Requires booktabs, xcolor and the paper's \bnds and \plainscheme macros.",
        f"% source: {os.path.relpath(results, os.path.dirname(HERE))} ({corpus} corpus)",
        r"\centering",
        # Fifteen columns of a per-language table need \tiny to fit the page width; the
        # main table has seven and stays readable.
        r"\tiny" if layout == "languages" else r"\small",
        # \tiny alone overfills: the gutters are set in points and do not shrink with the
        # font, so 15 columns carry 180pt of padding at the 6pt default. 3pt returns 90pt
        # of it. 4pt was enough until the column emphasis went in -- bold digits are wider
        # than upright ones, and the anchor row is bold in most columns. Scoped to the
        # float this is \input into.
        *([r"\setlength{\tabcolsep}{3pt}"] if layout == "languages" else []),
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
