#!/usr/bin/env python3
r"""How many common words survive as one token, per scheme.

The morphology metrics penalise the marker schemes for keeping words whole; this measures
the same property from the other side. A word that is one token is a word the model never
has to reassemble, which is the property the boundary markers exist to buy.

Words come from `wordfreq` (Speer et al.), frequency-ranked per language, so "common" is
an external judgement rather than one taken from the training corpus.

Counted in a carrier, not in isolation
--------------------------------------
A bare word is the wrong probe. The baseline's vocabulary is split between `the` and
` the`, and which of the two a bare word hits is an artefact of how the entry was learned
rather than of how the tokenizer behaves in text -- measuring bare words credits the
markers for duplication they removed and the baseline cannot use. The cost is therefore
the marginal one: the tokens a carrier phrase gains when the word is inserted into it,
space included. The carrier is built from that language's own two commonest words, so it
never introduces a script change.

Frequency weighting is reported beside the plain count: a scheme that keeps `the` whole
and loses a rare compound is not the same as the reverse, and the unweighted rate cannot
tell them apart.

    uv run --with wordfreq python marker_experiments/single_token_rate.py
    uv run --with wordfreq python marker_experiments/single_token_rate.py --top 5000
"""

import json
import os

import cyclopts

from script_bpe.tokenizers.load import load_tokenizer

# Registers the boundary pretokenizers, without which the tokenizers do not load.
import marker_experiments.downstream.boundary_tokenizer  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZERS = os.path.join(HERE, "downstream", "tokenizers")
GENERATED = os.path.join(HERE, "paper", "generated")

LANGS = ["en", "de", "fi", "ru", "ar", "ko"]
ARMS = ["plain", "bnd_w", "bnd_wp", "bnd_wpd", "bnd_wpd_caps", "bnd_wpd_extcaps"]

app = cyclopts.App()


def word_costs(tokenizer, words, carrier):
    """Marginal token cost of each word inside the carrier, space included."""
    left, right = carrier
    empty = len(tokenizer.encode(left.rstrip() + right))
    return [len(tokenizer.encode(left + w + right)) - empty for w in words]


@app.default
def main(
    langs: str = ",".join(LANGS),
    arms: str = ",".join(ARMS),
    trainers: str = "bpe,mingram",
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    top: int = 10_000,
    out: str = os.path.join(GENERATED, "single_token_rate.json"),
) -> None:
    """Report the share of the top-N words each arm encodes as one token.

    Args:
        langs: Comma-separated languages, each needing a wordfreq list.
        arms: Comma-separated arms to compare.
        trainers: Comma-separated trainers.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        top: How many of the most frequent words to test.
        out: JSON of rates, keyed by tokenizer name.
    """
    from wordfreq import top_n_list, word_frequency

    scores = json.load(open(out)) if os.path.exists(out) else {}
    for lang in [x.strip() for x in langs.split(",") if x.strip()]:
        words = top_n_list(lang, top)
        freqs = [word_frequency(w, lang) for w in words]
        total_freq = sum(freqs)
        # The two commonest words of the language itself, so the carrier stays in script.
        carrier = (f"{words[0]} ", f" {words[1]}")
        print(f"\n{lang}: top {len(words):,} words, carrier {carrier[0]!r}...{carrier[1]!r}")
        print(f"  {'arm':<18}{'trainer':<9}{'1 token':>9}{'freq-wtd':>10}{'mean toks':>11}")
        for arm in [x.strip() for x in arms.split(",") if x.strip()]:
            for trainer in [x.strip() for x in trainers.split(",") if x.strip()]:
                key = f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}"
                path = os.path.join(TOKENIZERS, f"{key}.json.gz")
                if not os.path.exists(path):
                    continue
                costs = word_costs(load_tokenizer(path), words, carrier)
                single = sum(1 for c in costs if c == 1)
                wtd = sum(f for c, f in zip(costs, freqs) if c == 1) / total_freq
                scores[key] = {
                    "lang": lang, "arm": arm, "trainer": trainer, "top": top,
                    "single_token_rate": single / len(words),
                    "freq_weighted_rate": wtd,
                    "mean_tokens": sum(costs) / len(costs),
                }
                print(f"  {arm:<18}{trainer:<9}{100 * single / len(words):>8.1f}%"
                      f"{100 * wtd:>9.1f}%{sum(costs) / len(costs):>11.3f}", flush=True)
                with open(out, "w") as f:
                    json.dump(scores, f, indent=2, sort_keys=True)
    print(f"\n[json] {out}: {len(scores)} cell(s)")


if __name__ == "__main__":
    app()
