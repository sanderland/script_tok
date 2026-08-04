#!/usr/bin/env python3
r"""MorphScore for the boundary-marker arms, with and without credit for whole words.

MorphAlign (`marker_experiments/morphalign.py`) reports the marker schemes at roughly half
the baseline's score, and the per-word evidence says that number is narrower than it looks:
it is an IBM1 bag-of-segments model, so it has no notion of *where* a boundary falls -- a
word-initial `s` in `s|ulk|s` collects the same credit as the plural suffix -- and it can
only reward a word by finding a frequent affix among its pieces. A tokenizer that keeps
`leads` whole scores zero on it, which is indistinguishable from splitting it wrongly.

MorphScore (`script_bpe/analysis/morphscore.py`, Arnett's dataset) answers both objections:

  * it checks the gold boundary *index* against the predicted boundary indices, so
    position matters and a stray leading consonant earns nothing
  * its precision term is (gold boundaries found) / (boundaries predicted), so
    over-splitting is penalised rather than rewarded
  * `exclude_single_tok` decides what happens to a word the tokenizer did not split:
    excluded from scoring (the default), or credited with a point

That last switch is the one that matters here. Under the markers a fifth to a quarter of
gold words are a single token, against a twentieth for the baseline, so the default silently
scores the two arms on different subsets -- and the subset it hands the marker schemes is
the harder one, since the words they kept whole are exactly the ones they handled best.
Both settings are reported.

The segmentation comes from `morphalign.segment`, which strips markers and case codes and
restores the casing, so what is scored is the same character spans the tokenizer produced.
MorphScore's own tokenization path is bypassed: it decodes atomic tokens straight to text,
which would leave `<|>` in the string and shift every boundary index.

    uv run python marker_experiments/morphscore_boundary.py
    uv run python marker_experiments/morphscore_boundary.py --langs en --arms plain,bnd_wpd
"""

import json
import os

import cyclopts
import numpy as np

from script_bpe.analysis.morphscore import MorphScore
from script_bpe.tokenizers.load import load_tokenizer

from marker_experiments.morphalign import segment

# Registers the boundary pretokenizers, without which the tokenizers do not load.
import marker_experiments.downstream.boundary_tokenizer  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZERS = os.path.join(HERE, "downstream", "tokenizers")
GENERATED = os.path.join(HERE, "paper", "generated")

# MorphScore's language codes for the six the grid trains. Arabic has no entry.
LANGS = {"en": "eng_latn", "de": "deu_latn", "fi": "fin_latn",
         "ru": "rus_cyrl", "ko": "kor_hang"}
ARMS = ["plain", "bnd_w", "bnd_wp", "bnd_wpd", "bnd_wpd_caps", "bnd_wpd_extcaps"]

app = cyclopts.App()


def score_arm(tokenizer, dataset, scorer):
    """Mean recall and precision over the dataset, plus what share of words went unsplit."""
    recall, precision, single, dropped = [], [], 0, 0
    for _, row in dataset.iterrows():
        wordform = row["wordform"]
        if not isinstance(wordform, str) or not wordform.strip():
            continue
        # A part counts only when it is a non-empty string. The dataset carries both NaN
        # and None for "absent", and a handful of rows have no stem at all; taking len()
        # of those is what MorphScore's own loop would crash on.
        parts = [row["preceding_part"], row["stem"], row["following_part"]]
        if not isinstance(row["stem"], str) or not row["stem"]:
            dropped += 1
            continue
        morphemes = [p for p in parts if isinstance(p, str) and p]

        tokens = segment(tokenizer, wordform)
        if tokens is None:
            dropped += 1
            continue
        single += len(tokens) == 1
        r, p = scorer.morph_eval(morphemes, tokens)
        recall.append(r)
        precision.append(p)
    return {
        "recall": float(np.nanmean(recall)),
        "precision": float(np.nanmean(precision)),
        "scored": int(np.sum(~np.isnan(recall))),
        "words": len(recall),
        "single_token_share": single / len(recall) if recall else 0.0,
        "dropped": dropped,
    }


@app.default
def main(
    langs: str = ",".join(LANGS),
    arms: str = ",".join(ARMS),
    trainers: str = "bpe,mingram",
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    out: str = os.path.join(GENERATED, "morphscore.json"),
) -> None:
    """Score each arm both ways: whole words excluded, and whole words credited.

    Args:
        langs: Comma-separated languages; each needs an entry in LANGS.
        arms: Comma-separated arms to score.
        trainers: Comma-separated trainers.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        out: JSON of scores, keyed by tokenizer name.
    """
    scores = json.load(open(out)) if os.path.exists(out) else {}
    for lang in [x.strip() for x in langs.split(",") if x.strip()]:
        # One filtered dataset per language, shared by every arm, so all arms are scored on
        # exactly the same words.
        excl = MorphScore(language_subset=[LANGS[lang]], stem_eq_lemma=True, exclude_single_tok=True)
        incl = MorphScore(language_subset=[LANGS[lang]], stem_eq_lemma=True, exclude_single_tok=False)
        dataset = excl._get_filtered_dataset()
        print(f"\n{lang} ({LANGS[lang]}): {len(dataset):,} words")
        print(f"  {'arm':<18}{'trainer':<9}{'excl R':>8}{'excl P':>8}{'incl R':>8}{'incl P':>8}"
              f"{'1-token':>9}{'scored':>8}")
        for arm in [x.strip() for x in arms.split(",") if x.strip()]:
            for trainer in [x.strip() for x in trainers.split(",") if x.strip()]:
                key = f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}"
                path = os.path.join(TOKENIZERS, f"{key}.json.gz")
                if not os.path.exists(path):
                    continue
                tokenizer = load_tokenizer(path)
                a = score_arm(tokenizer, dataset, excl)
                b = score_arm(tokenizer, dataset, incl)
                scores[key] = {"lang": lang, "arm": arm, "trainer": trainer,
                               "exclude_single_tok": a, "credit_single_tok": b}
                print(f"  {arm:<18}{trainer:<9}{a['recall']:>8.4f}{a['precision']:>8.4f}"
                      f"{b['recall']:>8.4f}{b['precision']:>8.4f}"
                      f"{100 * a['single_token_share']:>8.1f}%{a['scored']:>8,}", flush=True)
                with open(out, "w") as f:
                    json.dump(scores, f, indent=2, sort_keys=True)
    print(f"\n[json] {out}: {len(scores)} cell(s)")


if __name__ == "__main__":
    app()
