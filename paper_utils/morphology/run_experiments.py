"""Compare segmenters against projected SIGMORPHON surface gold on the most frequent words.

Training gold never includes an evaluation word: the evaluation set is the top-50k
frequency list, and supervised methods train on SIGMORPHON words outside it.

    uv run python -m paper_utils.morphology.run_experiments --lang eng
"""

import json
import os
import random

import cyclopts

from paper_utils.morphology import methods
from paper_utils.morphology.evaluate import report, top_words
from paper_utils.morphology.gold import celex_en, sigmorphon

OUT = os.path.join(os.path.dirname(__file__), "../../results/morphology")
GOLDFISH = {"eng": "eng_latn", "rus": "rus_cyrl"}

app = cyclopts.App()


@app.default
def main(lang: str = "eng", train_sizes: str = "1000,10000,50000", seed: int = 0) -> None:
    sig = sigmorphon(lang)
    ref = {w: b for w, (b, _exact) in sig.items()}
    with open(os.path.join(OUT, "counts", f"{GOLDFISH[lang]}.json")) as f:
        counts = json.load(f)["counts"]
    ranked = list(counts)[:50_000]
    eval_set = set(ranked)
    pool = sorted(w for w in ref if w not in eval_set and w.isalpha())
    random.Random(seed).shuffle(pool)
    results, preds = {}, {}

    if lang == "eng":
        results["celex_vs_sig"] = report("CELEX (reference agreement)", celex_en(), ref, ranked)
    preds["no_split"] = {w: [] for w in ranked}
    results["no_split"] = report("no split", preds["no_split"], ref, ranked)

    morf_counts = dict(list(counts.items())[:200_000])
    model = methods.morfessor_model(morf_counts)
    preds["morfessor"] = methods.morfessor_segment(model, ranked)
    results["morfessor"] = report("Morfessor (unsupervised)", preds["morfessor"], ref, ranked)

    semi = methods.morfessor_model(morf_counts, {w: ref[w] for w in pool[:1000]})
    preds["morfessor_semi1k"] = methods.morfessor_segment(semi, ranked)
    results["morfessor_semi1k"] = report("Morfessor semi-sup (1k)", preds["morfessor_semi1k"], ref, ranked)

    morf_bounds = {w: set(b) for w, b in methods.morfessor_segment(model, pool[:50_000] + ranked).items()}
    for n in [int(x) for x in train_sizes.split(",")]:
        train = {w: ref[w] for w in pool[:n]}
        crf = methods.crf_train(train)
        preds[f"crf{n}"] = methods.crf_segment(crf, ranked)
        results[f"crf{n}"] = report(f"CRF ({n:,})", preds[f"crf{n}"], ref, ranked)
        crf = methods.crf_train(train, extra=lambda w: [morf_bounds[w]])
        preds[f"crf{n}_morf"] = methods.crf_segment(crf, ranked, extra=lambda w: [morf_bounds[w]])
        results[f"crf{n}_morf"] = report(f"CRF + Morfessor ({n:,})", preds[f"crf{n}_morf"], ref, ranked)

    with open(os.path.join(OUT, f"experiments_{lang}.json"), "w") as f:
        json.dump({"results": results, "preds": preds}, f, ensure_ascii=False)


if __name__ == "__main__":
    app()
