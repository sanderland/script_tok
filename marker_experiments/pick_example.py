#!/usr/bin/env python3
"""Find one real sentence whose tokenization reproduces the corpus-level result.

A worked example is only worth printing if it is honest: an invented sentence can be
tuned to make any scheme look good. This searches the held-out English slice for a
sentence whose per-scheme token counts land closest to the deltas measured over the whole
slice, so the example a reader sees is a fair sample of what the tables report rather than
a flattering one.

Candidates must contain punctuation and a digit, since those are exactly what \\bnd{wp}
and \\bnd{wpd} change, and must be short enough to print.

Writes the chosen sentence's tokenization under every scheme, both to stdout and to
paper/generated/table_example.tex.

    uv run python marker_experiments/pick_example.py
"""

import json
import os
import re

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(HERE, "paper", "generated")
TOKENIZERS = os.path.join(HERE, "tokenizers")

# All five share one 250M-character English corpus and 32,768 additional vocabulary; the
# caps cell was trained by the caps grid, which used the same corpus (both report the
# baseline bnd_wpd at 3.8849 chars/token).
SCHEMES = [
    ("plain", "mg250_en_plain_bpe_32k.json.gz", None),
    ("bnd_w", "mg250_en_bnd_w_bpe_32k.json.gz", "bnd_w"),
    ("bnd_wp", "mg250_en_bnd_wp_bpe_32k.json.gz", "bnd_wp"),
    ("bnd_wpd", "mg250_en_bnd_wpd_bpe_32k.json.gz", "bnd_wpd"),
    ("bnd_wpd_caps", "caps250_en_bnd_wpd_caps_bpe_32k.json.gz", "bnd_wpd_caps"),
]

app = cyclopts.App()


def load_all():
    from script_bpe.pretokenize import get_pretokenizer
    from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer

    from marker_experiments.boundary_pretokenizer import get_boundary_pretokenizer

    out = []
    for name, filename, variant in SCHEMES:
        path = os.path.join(TOKENIZERS, filename)
        if not os.path.exists(path):
            print(f"[example] missing {filename}; {name} omitted")
            continue
        pt = get_pretokenizer("scriptenc3_cb") if variant is None else get_boundary_pretokenizer(variant)
        out.append((name, BPETokenizer.load(path), pt))
    return out


def render(tokenizer, pt, text):
    """The token sequence as printable strings, in order."""
    marker = getattr(pt, "marker_token_id", None)
    shift = getattr(pt, "shift_token_id", None)
    caps = getattr(pt, "caps_token_id", None)
    label = {marker: "<|>", shift: "<^>", caps: "<^^>"}
    special = {k for k in (marker, shift, caps) if k is not None}
    pieces = []
    for tid in tokenizer.encode(text):
        ids = list(tokenizer.tokens[tid].atomic_tokens)
        s, run = "", []
        for i in ids:
            if i in special:
                if run:
                    s += pt.try_decode_strict(run) or "?"
                    run = []
                s += label[i]
            else:
                run.append(i)
        if run:
            s += pt.try_decode_strict(run) or "?"
        pieces.append(s.replace(" ", "␣"))  # open box, so a leading space is visible
    return pieces


def sentences(texts, lo, hi):
    seen = set()
    for doc in texts:
        for line in doc.split("\n"):
            for s in re.split(r"(?<=[.!?]) +", line.strip()):
                s = s.strip()
                if not (lo <= len(s) <= hi) or s in seen:
                    continue
                seen.add(s)
                if not re.search(r"\d", s) or not re.search(r"[,.;:()]", s):
                    continue
                if s != s.strip() or "|" in s or s.count('"') % 2:
                    continue
                yield s


@app.default
def main(
    eval_texts: str = os.path.join(HERE, "eval_texts", "en.json"),
    min_chars: int = 70,
    max_chars: int = 130,
    out: str = os.path.join(GENERATED, "table_example.tex"),
    show: int = 3,
) -> None:
    """Pick and print the most representative sentence.

    Args:
        eval_texts: JSON list of held-out documents to search.
        min_chars: Shortest candidate sentence.
        max_chars: Longest candidate sentence, so the tokenization still prints.
        out: LaTeX table to write.
        show: How many runners-up to list.
    """
    loaded = load_all()
    if len(loaded) < 2:
        raise SystemExit("need at least the baseline and one variant")
    texts = json.load(open(eval_texts))

    # The target each candidate is scored against: the delta over the WHOLE slice, which
    # is the number the tables report.
    totals = {name: sum(len(t.encode(doc)) for doc in texts) for name, t, _ in loaded}
    # chars/token relative to plain, which is what the tables report. Over a fixed text
    # that reduces to plain_tokens / n - 1, so no character count is needed.
    target = {name: 100 * (totals["plain"] / n - 1) for name, n in totals.items()}
    chars = sum(map(len, texts))
    print("over the whole held-out slice:")
    for name, _, _ in loaded:
        print(f"  {name:<14} {totals[name]:>8,} tokens  {chars/totals[name]:.4f} ch/tok  "
              f"{target[name]:+.2f}%")

    scored = []
    for s in sentences(texts, min_chars, max_chars):
        counts = {name: len(t.encode(s)) for name, t, _ in loaded}
        delta = {n: 100 * (counts["plain"] / c - 1) for n, c in counts.items()}
        err = sum(abs(delta[n] - target[n]) for n, _, _ in loaded if n != "plain")
        scored.append((err, s, counts, delta))
    if not scored:
        raise SystemExit("no candidate sentence matched the filters")
    scored.sort(key=lambda x: x[0])

    err, sentence, counts, delta = scored[0]
    print(f"\nsearched {len(scored):,} candidate sentences; best total error {err:.2f}pp\n")
    print(f"  {sentence!r}\n")
    print(f"  {'scheme':<14} {'tokens':>6} {'ch/tok vs plain':>16} {'slice':>9}")
    for name, _, _ in loaded:
        print(f"  {name:<14} {counts[name]:>6} {delta[name]:>+15.2f}% {target[name]:>+8.2f}%")

    print("\ntokenization (␣ is a space inside a token):\n")
    rendered = {}
    for name, t, pt in loaded:
        pieces = render(t, pt, sentence)
        rendered[name] = pieces
        assert t.decode(t.encode(sentence)) == sentence, f"{name} does not round-trip"
        print(f"{name} ({len(pieces)}):")
        print("   " + " │ ".join(pieces) + "\n")

    if show:
        print(f"runners-up:")
        for err, s, c, _ in scored[1:1 + show]:
            print(f"  {err:5.2f}pp  {s[:90]}")

    def esc(t):
        """LaTeX-safe, with the three marker glyphs handed to the paper's macros."""
        for a, b in [("\\", r"\textbackslash "), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                     ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
                     ("~", r"\textasciitilde "), ("^", r"\textasciicircum ")]:
            t = t.replace(a, b)
        return (t.replace("<|>", r"\mk{}").replace("<\\textasciicircum \\textasciicircum >", r"\capsc{}")
                 .replace("<\\textasciicircum >", r"\shift{}")
                 .replace("\u2423", r"\textvisiblespace{}"))

    rows = []
    for name, _, _ in loaded:
        cells = " ".join(
            r"\fbox{\strut " + p.replace("\\", r"\textbackslash ").replace("_", r"\_")
            .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
            .replace("$", r"\$").replace("<|>", r"\mk{}").replace("<^^>", r"\capsc{}")
            .replace("<^>", r"\shift{}").replace("␣", r"\textvisiblespace{}") + "}"
            for p in rendered[name]
        )
        rows.append(rf"\texttt{{{name.replace('_', chr(92) + '_')}}} & {len(rendered[name])} & {cells} \\")
    tex = "\n".join([
        "% Generated by marker_experiments/pick_example.py. Do not edit.",
        r"% Requires \mk, \shift, \capsc and \textvisiblespace.",
        f"% sentence chosen from {os.path.relpath(eval_texts, os.path.dirname(HERE))}",
        r"\centering\small",
        # Tight boxes: 30 tokens across five rows, so the default rule and padding would
        # dominate the figure.
        r"\setlength{\fboxsep}{1pt}\setlength{\fboxrule}{0.3pt}",
        r"\begin{tabular}{llp{0.72\textwidth}}",
        r"\toprule",
        r"Scheme & $n$ & Tokens \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        # The sentence is set in \texttt so its straight quotes render as themselves.
        r"\caption{Tokenization of one held-out sentence, \texttt{" + esc(sentence) + r"}. "
        rf"Chosen automatically as the sentence whose per-scheme token counts land closest "
        rf"to the deltas measured over the whole held-out slice, so it is a representative "
        rf"sample rather than a flattering one.}}",
        r"\label{tab:example}",
        "",
    ])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(tex)
    print(f"\n[tex] {out}")


if __name__ == "__main__":
    app()
