#!/usr/bin/env python3
r"""Generate the worked pre-tokenization example.

One sentence, pre-tokenized under every scheme in the main table, with the counts the
caption quotes computed rather than typed. This is the *pre-tokenizer's* output, before any
vocabulary is learned, so it needs no trained tokenizer and no corpus: it runs on a clean
checkout in under a second.

The sentence is chosen to exercise each scheme at the point where it differs from the one
above it -- a space between two word spans, a digit span with a space either side, a
sentence-final period, a mixed-case word, and an all-caps word:

    Ash caught 3 SolidGoldMagikarp. WOW!

Changing `--text` recomputes the table and the counts, but the caption argues about *this*
sentence's features; pass `--caption` as well, or expect prose that no longer matches.

Needs `booktabs` and the paper's macros:

    \newcommand{\mk}{\textcolor{bndcol}{\textbf{|}}}      % boundary marker
    \newcommand{\shf}{\textcolor{bndcol}{\textbf{$\uparrow$}}}    % title-case code
    \newcommand{\cps}{\textcolor{bndcol}{\textbf{$\Uparrow$}}}    % all-caps code
    \newcommand{\tsp}{\textvisiblespace}
    \newcommand{\tokens}[1]{\texttt{#1}}
    \newcommand{\pretokens}[1]{\texttt{#1}}               % comma-separated

    uv run python paper_utils/boundary/make_example_table.py
"""

import os

import cyclopts

from paper_utils.boundary.make_intrinsic_table import MAIN_ARMS
from paper_utils.boundary.utils import GENERATED

DEFAULT_TEXT = "Ash caught 3 SolidGoldMagikarp. WOW!"

# The prose macro, not the `\bnds` table one the numeric tables use: this column is read as
# a name in running text rather than scanned down as a key.
SCHEME_LABEL = {
    "plain": r"\plainscheme",
    "bnd_w": r"\bnd{w}",
    "bnd_wp": r"\bnd{wp}",
    "bnd_wpd": r"\bnd{wpd}",
    "bnd_wpd_caps": r"\bnd{wpdcaps}",
}
# Rows of the main table, in its order. `plain` first, then whatever the main table argues.
SCHEMES = ["plain", *MAIN_ARMS]

# How each control sequence separates itself from literal text that follows. LaTeX needs
# something there -- `\mkAsh` is an undefined control sequence -- and the two forms are not
# interchangeable in output width, so they are stated rather than picked.
SEPARATOR = {r"\mk": " ", r"\shf": " ", r"\cps": " ", r"\tsp": None}  # None means brace it

CAPTION = r"""Pretokens under each scheme, before vocab entries are learned. Under
\bnd{w}, the space between two marked word spans is removed, but the spaces
bordering \texttt{%(digit)s} and the space after the period remain, raising the count
from %(n_plain)d to %(n_w)d.
\bnd{wp} marks the period on its right only, where a space was removed. \bnd{wpd} marks the digit span as well, and no space remains.
\bnd{wpdcaps} places \shf{} or \cps{} before the span's opening marker, so the
count is unchanged and \tokens{\mk %(lowered)s\mk} is a pretoken that largely overlaps with
the lower-case form.
Mixed case is not restorable from a single
marker, so \texttt{%(mixed)s} is left as-is."""

app = cyclopts.App()


def escape(text):
    r"""Literal text as LaTeX. The example is ASCII prose, but a `_` in a chosen sentence
    would otherwise compile as a subscript rather than fail."""
    for ch in "\\&%$#_{}":
        text = text.replace(ch, "\\" + ch)
    return text.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")


def build(pt, text):
    """One scheme's pre-tokens, as a list of LaTeX strings."""
    # The baseline has none of these; getattr rather than a branch on the scheme name, so
    # this stays right if an arm ever enables one code and not the other.
    macro = {getattr(pt, name, None): m for name, m in
             (("marker_token_id", r"\mk"), ("shift_token_id", r"\shf"), ("caps_token_id", r"\cps"))}
    macro.pop(None, None)

    out = []
    for chunk in pt.pretokenize(text):
        parts, run = [], []

        def flush(run=None):
            if run:
                decoded = pt.try_decode_strict(run)
                assert decoded is not None, f"undecodable run in {chunk}"
                parts.append(("text", escape(decoded)))

        for tok in chunk:
            if tok in macro:
                flush(run)
                run = []
                parts.append(("macro", macro[tok]))
            else:
                run.append(tok)
        flush(run)
        out.append(render(parts))
    return out


def render(parts):
    r"""Join (kind, value) pairs, separating a control sequence from following text.

    A space renders `\mk Ash`; braces render `{\tsp}caught`. Two control sequences need
    nothing between them, so `\shf\mk` is written as it stands.
    """
    s = ""
    for i, (kind, value) in enumerate(parts):
        if kind != "macro":
            # A literal space inside a pre-token is the baseline's leading space. It takes
            # the same treatment as the macros: braced when text follows it, bare at the end.
            s += "".join(r"{\tsp}" if ch == " " and j + 1 < len(value) else
                         (r"\tsp" if ch == " " else ch)
                         for j, ch in enumerate(value))
            continue
        text_follows = i + 1 < len(parts) and parts[i + 1][0] == "text"
        sep = SEPARATOR[value]
        s += (value + sep) if (text_follows and sep) else (f"{{{value}}}" if text_follows else value)
    return s


@app.default
def main(
    text: str = DEFAULT_TEXT,
    out: str = os.path.join(GENERATED, "table_example.tex"),
    caption: str | None = None,
    label: str = "tab:example",
) -> None:
    """Write the worked pre-tokenization example.

    Args:
        text: Sentence to pre-tokenize. The default is the paper's.
        out: LaTeX file to write.
        caption: Overrides the caption, which otherwise argues about the default sentence.
        label: LaTeX label.
    """
    from script_bpe.pretokenize import get_pretokenizer

    from paper_utils.boundary.boundary_pretokenizer import get_boundary_pretokenizer

    rows, counts = [], {}
    for arm in SCHEMES:
        pt = get_pretokenizer("scriptenc3_cb") if arm == "plain" else get_boundary_pretokenizer(arm)
        # An example that does not round-trip is not an example of this scheme.
        flat = [t for chunk in pt.pretokenize(text) for t in chunk]
        assert pt.decode(flat) == pt.normalize(text), f"{arm} does not round-trip {text!r}"
        pretokens = build(pt, text)
        # `\pretokens{}` takes a comma-separated list, so a pre-token that IS a comma is
        # indistinguishable from the separator: `.,,,\mk` reads as three separators. The
        # paper's sentence has none. Warn rather than emit it silently, and rather than
        # brace every pre-token, which would change the output for the sentence that works.
        if any("," in t for t in pretokens):
            print(f"  WARNING: {arm} has a pre-token containing a comma, which collides with "
                  f"the \\pretokens separator. Choose a sentence without one, or brace the "
                  f"list items in the macro.")
        counts[arm] = len(pretokens)
        rows.append(rf"{SCHEME_LABEL[arm]} & \pretokens{{{','.join(pretokens)}}} & {len(pretokens)} \\")

    if caption is None:
        # Facts the default caption names, pulled out of the sentence rather than typed, so
        # a changed sentence produces a caption that is wrong loudly rather than quietly.
        digit = next((w for w in text.split() if w.isdigit()), "the digit")
        # Mixed case means a cased word that is none of lower, upper or title -- `isalpha`
        # first, or a digit qualifies by failing all three.
        mixed = next((w for w in (x.strip(".,!?") for x in text.split())
                      if w.isalpha() and not (w.islower() or w.isupper() or w.istitle())),
                     "the mixed-case word")
        lowered = next((w.strip(".,!?").lower() for w in text.split() if w.istitle()), "the word")
        caption = CAPTION % dict(digit=escape(digit), mixed=escape(mixed), lowered=escape(lowered),
                                 n_plain=counts["plain"], n_w=counts["bnd_w"])

    lines = [
        "% Generated by paper_utils/boundary/make_example_table.py. Do not edit.",
        r"% Requires booktabs and the paper's \mk, \shf, \cps, \tsp, \pretokens and \tokens macros.",
        f"% sentence: {text}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}l l r@{}}",
        r"\toprule",
        r"Scheme & Pre-tokens & \# \\",
        r"\midrule",
        # \\[3pt] between rows, not after the last: the extra leading separates the schemes,
        # and after the last row it would only pad the bottom rule.
        *[r + (r"[3pt]" if i < len(rows) - 1 else "") for i, r in enumerate(rows)],
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        "",
    ]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"[tex] {out}")
    print("      " + "  ".join(f"{a}={counts[a]}" for a in SCHEMES))


if __name__ == "__main__":
    app()
