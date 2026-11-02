"""Pairwise boundary F1 between every annotator on the LLM sample, and per boundary type.

F1 between two annotators is symmetric, so it reads as agreement. The table says whether the
LLM runs agree with each other more than any of them agrees with the human-derived references,
which is the case for using them as silver gold.

    uv run python -m paper_utils.morphology.agreement --lang eng
"""

import itertools
import json
import os

import cyclopts

from paper_utils.morphology.evaluate import prf
from paper_utils.morphology.gold import celex_en, sigmorphon

OUT = os.path.join(os.path.dirname(__file__), "../../results/morphology")
app = cyclopts.App()


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


@app.default
def main(lang: str = "eng", runs: str = "haiku_typed,haiku_typed_r2,sonnet_typed,sonnet_typed_r2",
         out: str = "") -> None:
    typed = {r: load(f"llm_{lang}_{r}.json") for r in runs.split(",")}
    ann = {r: {w: [b for b, _ in v] for w, v in t.items()} for r, t in typed.items()}
    ann["SIGMORPHON"] = {w: b for w, (b, _) in sigmorphon(lang).items()}
    if lang == "eng":
        ann["CELEX"] = celex_en()
    crf = load(f"experiments_{lang}.json")["preds"]["crf50000"]
    ann["CRF-50k"] = crf
    sample = sorted(set.intersection(*(set(typed[r]) for r in typed)))
    names = list(ann)
    matrix = {}
    for a, b in itertools.product(names, names):
        if a < b:
            r = prf(ann[a], ann[b], sample)
            matrix[f"{a}|{b}"] = r
    print(f"{len(sample)} words in every LLM run")
    for k, v in sorted(matrix.items(), key=lambda kv: -kv[1]["F1"]):
        print(f"  {k:<34} F1 {v['F1']:.3f}  word agreement {v['word_acc']:.2f}  n {v['n']}")

    # per boundary type: agreement between the two runs of each model, and across models
    by_type = {}
    for t in ("infl", "deriv", "comp"):
        only = {r: {w: [b for b, ty in v if ty == t] for w, v in typed[r].items()} for r in typed}
        for a, b in itertools.combinations(typed, 2):
            by_type[f"{t}:{a}|{b}"] = prf(only[a], only[b], sample)
    print("per boundary type")
    for k, v in by_type.items():
        print(f"  {k:<42} F1 {v['F1']:.3f}")
    if out:
        with open(out, "w") as f:
            json.dump({"pairs": matrix, "by_type": by_type, "n": len(sample)}, f, indent=1)


if __name__ == "__main__":
    app()
