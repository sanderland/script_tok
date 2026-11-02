"""Render `results.typ` for the notes from the result JSONs under results/morphology.

    uv run python paper_utils/morphology/notes/make_results.py && uv run python paper_utils/morphology/notes/build.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../../../results/morphology")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "(left," + ",".join(["right"] * (len(header) - 1)) + ")"
    cells = ", ".join(f"[{h}]" for h in header)
    for r in rows:
        cells += ",\n  " + ", ".join(f"[{c}]" for c in r)
    return f"#table(columns: {len(header)}, align: {align}, stroke: 0.4pt,\n  {cells})\n"


def band_row(name: str, rows: dict) -> list[str]:
    out = [name]
    for band in ("0k-1k", "1k-10k", "10k-50k"):
        v = rows[band]
        out.append(f"{v['F1']:.2f} ({v['P']:.2f}/{v['R']:.2f})")
    return out


def main() -> None:
    parts = []
    exp = load("experiments_eng.json")["results"]
    sweep = load("morfessor_sweep_eng.json")
    parts.append("""=== Classical segmenters against the SIGMORPHON reference

English, Goldfish frequency bands. Cells are boundary F1 (precision/recall) against
SIGMORPHON 2022 projected to surface. Supervised methods train on SIGMORPHON words outside the
top 50k. `CELEX` is not a method: it is the second human-derived gold, scored against the first,
so its row is the agreement between two references.
""")
    rows = [band_row("CELEX vs SIGMORPHON", exp["celex_vs_sig"]),
            band_row("no split", exp["no_split"]),
            band_row("Morfessor, $alpha = 1$", exp["morfessor"])]
    rows += [band_row(f"Morfessor, $alpha = {a}$", sweep[a]) for a in ("0.3", "0.1", "0.03")]
    rows.append(band_row("Morfessor semi-sup. (1k)", exp["morfessor_semi1k"]))
    for n in ("1000", "10000", "50000"):
        rows.append(band_row(f"CRF ({int(n):,})", exp[f"crf{n}"]))
        rows.append(band_row(f"CRF + Morfessor ({int(n):,})", exp[f"crf{n}_morf"]))
    parts.append(table(["", "top 1k", "1k-10k", "10k-50k"], rows))
    parts.append("""
Unsupervised Morfessor tops out at 0.67 F1 at its best corpus weight, which is the range the
literature reports. A CRF with Ruokolainen features learns from labelled words steadily (0.71
at 1k, 0.85 at 50k); Morfessor features buy precision for recall and no F1. The two human-derived
references agree with each other at only 0.67 F1 in the 1k-10k band, so 0.85 against one of them
is close to what any method can show on this yardstick.
""")

    ag = load("agreement_eng_xfamily.json")
    llm = ["haiku_typed", "haiku_typed_r2", "sonnet_typed", "gemini-3.8-flash_typed", "gpt-6-astra_typed"]
    label = {"haiku_typed": "Haiku", "haiku_typed_r2": "Haiku (run 2)", "sonnet_typed": "Sonnet",
             "gemini-3.8-flash_typed": "Gemini 3.8 Flash", "gpt-6-astra_typed": "GPT-6 Astra",
             "SIGMORPHON": "SIGMORPHON", "CELEX": "CELEX", "CRF-50k": "CRF (50k)"}
    names = llm + ["SIGMORPHON", "CRF-50k", "CELEX"]

    def pair(a, b):
        if a == b:
            return "--"
        k = f"{a}|{b}" if f"{a}|{b}" in ag["pairs"] else f"{b}|{a}"
        return f"{ag['pairs'][k]['F1']:.2f}"

    parts.append(f"""
=== LLM annotation

Prompt: typed surface segmentation (`llm_segment.TYPED_PROMPT`), boundaries marked `+`
inflection, `-` derivation, `=` compound, with explicit conventions for depth (split all the
way down), bound roots (split an affix only if the base is a real related word) and spelling
changes (affixes keep their dictionary form, the stem absorbs the change: `ignor+ed`,
`patch+es`). An earlier untyped prompt left depth open and Sonnet split only the outer layer
(F1 0.78 against SIGMORPHON, against 0.85 typed).

Pairwise boundary F1 on {ag['n']} words drawn from the top 10k and present in SIGMORPHON. F1 is
symmetric, so each cell reads as agreement. Claude models run through `claude -p`, the others
through the Copilot CLI.
""")
    parts.append(table([""] + [label[n] for n in names],
                       [[label[a]] + [pair(a, b) for b in names] for a in names]))
    t = ag["by_type"]
    parts.append("""
LLMs agree with each other at 0.93-0.97, across model families as much as within one, and with
either human-derived reference at 0.70-0.86. Two runs of the same model agree at 0.95-0.97, so
sampling noise is a small part of the disagreement. By boundary type, inflection is almost
settled and derivation is where the remaining disagreement lives:
""")
    def mean_type(kind):
        vals = [v["F1"] for k, v in t.items() if k.startswith(kind + ":")]
        return f"{sum(vals) / len(vals):.2f} ({min(vals):.2f}-{max(vals):.2f})"
    parts.append(table(["boundary type", "mean pairwise F1 (range)"],
                       [["inflection", mean_type("infl")], ["derivation", mean_type("deriv")],
                        ["compound", mean_type("comp")]]))
    parts.append("""
Of the 256 words where all four models (Haiku, Sonnet, Gemini, GPT; three families) agree exactly, 29 differ from SIGMORPHON. Most
are reference errors (`alex|a`, `gen|ome`, `line|n`, `los|t`, `give|a|way`) or artefacts of
projecting canonical gold to surface (`ignore|d`, `patche|s`, `consume|r|s`, where the LLM
convention is `ignor+ed`, `patch+es`, `consum-er+s`). A few go the other way and are debatable
deep Latinate splits (`typ-ic-al`, `athlet-ic-s`). The 35 words where the LLMs disagree among
themselves are bound roots and depth (`cognit-ive`, `re-mind+ed`, `proto-type`), which a
per-boundary vote resolves.
""")

    mini = load("agreement_eng_minimal.json")["pairs"]
    mnames = ["haiku_minimal", "sonnet_minimal", "gemini-3.8-flash_minimal", "gpt-6-astra_minimal",
              "SIGMORPHON", "CELEX"]
    mlabel = {"haiku_minimal": "Haiku", "sonnet_minimal": "Sonnet", "gemini-3.8-flash_minimal": "Gemini",
              "gpt-6-astra_minimal": "GPT", "SIGMORPHON": "SIGMORPHON", "CELEX": "CELEX"}

    def mpair(a, b):
        if a == b:
            return "--"
        k = f"{a}|{b}" if f"{a}|{b}" in mini else f"{b}|{a}"
        return f"{mini[k]['F1']:.2f}"

    parts.append("""
==== Does the rulebook make the agreement?

The typed prompt fixes depth, bound roots and spelling conventions, so models that follow it
will agree partly because of the rules. Rerun with a minimal prompt ("split each word into its
morphemes, pieces must concatenate to the word", `llm_segment.MINIMAL_PROMPT`):
""")
    parts.append(table([""] + [mlabel[n] for n in mnames],
                       [[mlabel[a]] + [mpair(a, b) for b in mnames] for a in mnames]))
    parts.append("""
Agreement drops from 0.93-0.97 to 0.82-0.93: the rulebook accounts for roughly 0.08 of it. The
remaining minimal-prompt disagreements are the same convention questions the rulebook settles,
not different morphology: etymological roots (Haiku's `con|tain`, `in|sert`, `e|merge`),
e-drop (`ignore|d` against `ignor|ed`), consonant doubling (`fit|ting` against `fitt|ing`) and
depth (`behavior|al` against `behav|ior|al`). Gemini is the strongest minimal-prompt annotator
(0.85 against SIGMORPHON, the same as typed); Haiku is the weakest (0.78) because it splits
Latinate roots.

Caveat: agreement is not accuracy. The models may share biases, the human references use
different conventions from each other and from the rulebook, and the SIGMORPHON reference here
is a projection that adds its own errors. A hand check of a random sample is the only direct
accuracy measurement and has not been done yet.
""")

    silver = os.path.join(RES, "silver", "eng_top10000.tsv")
    if os.path.exists(silver):
        with open(silver, encoding="utf-8") as f:
            rows = [line.rstrip("\n").split("\t") for line in f][1:]
        n = len(rows)
        unan = sum(r[4] == "1.00" for r in rows)
        split = sum(any(c in r[3] for c in "+-=") for r in rows)
        parts.append(f"""
=== English top-10k silver gold

`build_silver.py`: Haiku, Gemini 3.8 Flash and Sonnet annotate every letters-only word of two or
more characters in the top 10k; a boundary is kept when at least two of the three place it.
{n:,} words, {unan:,} ({100 * unan / n:.0f}%) unanimous, {split:,} ({100 * split / n:.0f}%) with at
least one boundary. File: `results/morphology/silver/eng_top10000.tsv`.
""")
    with open(os.path.join(HERE, "results.typ"), "w") as f:
        f.write("\n".join(parts))
    print(os.path.join(HERE, "results.typ"))


if __name__ == "__main__":
    main()
