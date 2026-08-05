#!/usr/bin/env python3
r"""MorphScore for the boundary-marker arms, with and without credit for whole words.

MorphAlign is the metric the MinGram paper reports, and it is a poor fit here: it is an
IBM1 bag-of-segments model, so it has no notion of *where* a boundary falls -- a
word-initial `s` in `s|ulk|s` collects the same credit as the plural suffix -- and it can
only reward a word by finding a frequent affix among its pieces. A tokenizer that keeps
`leads` whole scores zero on it, which is indistinguishable from splitting it wrongly.
That is exactly what the boundary schemes do, so MorphScore is what the paper reports.

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

The segmentation comes from `segment` below, which strips markers and case codes and
restores the casing, so what is scored is the same character spans the tokenizer produced.
MorphScore's own tokenization path is bypassed: it decodes atomic tokens straight to text,
which would leave `<|>` in the string and shift every boundary index.

    uv run python paper_utils/boundary/morphscore_boundary.py
    uv run python paper_utils/boundary/morphscore_boundary.py --langs en --arms plain,bnd_wpd
"""

import json
import os

import cyclopts
import numpy as np

from script_bpe.analysis.morphscore import MorphScore
from script_bpe.tokenizers.load import load_tokenizer

# Registers the boundary pretokenizers, without which the tokenizers do not load.
import paper_utils.boundary.downstream.boundary_tokenizer  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZERS = os.path.join(HERE, "downstream", "tokenizers")
GENERATED = os.path.join(HERE, "paper", "generated")

# MorphScore's language codes for the six the grid trains. Arabic has no entry.
LANGS = {"en": "eng_latn", "de": "deu_latn", "fi": "fin_latn",
         "ru": "rus_cyrl", "ko": "kor_hang"}
ARMS = ["plain", "bnd_w", "bnd_w_caps", "bnd_wp", "bnd_wp_caps",
        "bnd_wpd", "bnd_wpd_caps"]

app = cyclopts.App()


def segment(tokenizer, word):
    """The word's segments under this tokenizer, markers and case codes removed.

    Returns None when the segments cannot be made to spell the word -- a character the
    script encoding does not cover, or a case transform that is not invertible. Those
    words are dropped rather than guessed at, and counted.
    """
    pt = tokenizer.pretokenizer
    marker = getattr(pt, "marker_token_id", None)
    shift, caps = getattr(pt, "shift_token_id", None), getattr(pt, "caps_token_id", None)

    pieces, code = [], None
    for token_id in tokenizer.encode(word):
        ids = [int(i) for i in tokenizer.tokens[token_id].atomic_tokens]
        if shift is not None and shift in ids:
            code = "shift"
        if caps is not None and caps in ids:
            code = "caps"
        core = [i for i in ids if i not in (marker, shift, caps)]
        if not core:
            continue                      # a marker or code alone spells nothing
        text = pt.try_decode_strict(core)
        if text is None:
            return None
        pieces.append(text)

    joined = "".join(pieces)
    if code == "shift":
        joined = joined[:1].upper() + joined[1:]
    elif code == "caps":
        joined = joined.upper()
    if joined != word:
        return None
    # Re-split the recased string at the original offsets: the segmentation is the
    # tokenizer's and must not shift because the casing was restored.
    out, at = [], 0
    for piece in pieces:
        out.append(joined[at:at + len(piece)])
        at += len(piece)
    return out


def segment_in_context(tokenizer, word, carrier=("the ", " to")):
    """The word's segments as the tokenizer would produce them *in text*.

    Both MorphScore and MorphAlign encode the gold word on its own, which is not a neutral
    choice: under a leading-space convention the entry is ` leads`, so a bare `leads`
    cannot match it and comes back as `lead|s` -- a split the tokenizer never makes in
    running text. That phantom split is then scored as morphological insight. The marker
    schemes are unaffected, since their span is delimited the same way either side, so the
    bare-word probe silently compares one scheme's real behaviour against another's
    artefact.

    Encoding inside a carrier and keeping the tokens that cover the word's characters
    measures both schemes as they actually behave. Tokens cannot straddle a word boundary
    under either scheme -- the baseline's chunking splits on the space, and the markers
    forbid a merge across the elided-space point -- so the word's tokens are exactly those
    covering its span.
    """
    left, right = carrier
    text = left + word + right
    start, end = len(left), len(left) + len(word)

    pieces, at, code, seen, pending = [], 0, None, [], None
    pt = tokenizer.pretokenizer
    marker = getattr(pt, "marker_token_id", None)
    shift, caps = getattr(pt, "shift_token_id", None), getattr(pt, "caps_token_id", None)
    for token_id in tokenizer.encode(text):
        ids = [int(i) for i in tokenizer.tokens[token_id].atomic_tokens]
        # Offsets have to come from decoding the sequence so far, not from this token
        # alone: the marker scheme reconstructs an elided space from a *pair* of touching
        # markers, so per-token lengths do not sum to the text and the span drifts.
        seen.extend(ids)
        span = (at, len(pt.decode(seen)))
        at = span[1]
        inside = span[1] > start and span[0] < end
        here = ("caps" if caps is not None and caps in ids else
                "shift" if shift is not None and shift in ids else None)
        if not inside:
            # Under `_caps` the case code sits outside the marker pair, so it is its own
            # token sitting just before the word rather than part of it. It cannot be
            # placed by offset: the pair of touching markers only reconstructs the elided
            # space when both have decoded, so the code token ends one character short of
            # where the word begins and no boundary test matches it. Carry it forward
            # instead -- a code applies to the next character, so the last one still
            # unspent when the word starts is the word's. Anything that spells characters
            # spends it. Without this every capitalised word came back lowercased, failed
            # the spelling check and was dropped, which is exactly what the 28 words lost
            # under `_caps` and no other arm had in common.
            pending = here or (None if span[1] > span[0] else pending)
            continue
        code = here or pending or code
        pending = None
        core = [i for i in ids if i not in (marker, shift, caps)]
        if not core:
            continue
        text_piece = pt.try_decode_strict(core)
        if text_piece is None:
            return None
        pieces.append(text_piece)

    if pieces:                                    # the leading space rides on the first token
        pieces[0] = pieces[0].lstrip(" ")
    joined = "".join(pieces)
    if code == "shift":
        joined = joined[:1].upper() + joined[1:]
    elif code == "caps":
        joined = joined.upper()
    if joined != word:
        return None
    out, at = [], 0
    for piece in pieces:
        out.append(joined[at:at + len(piece)])
        at += len(piece)
    return [p for p in out if p]


def score_arm(tokenizer, dataset, scorer, segmenter=segment):
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

        tokens = segmenter(tokenizer, wordform)
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
    in_context: bool = False,
    out: str = os.path.join(GENERATED, "morphscore.json"),
    force: bool = False,
) -> None:
    """Score each arm both ways: whole words excluded, and whole words credited.

    Args:
        langs: Comma-separated languages; each needs an entry in LANGS.
        arms: Comma-separated arms to score.
        trainers: Comma-separated trainers.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        in_context: Segment the word inside a carrier phrase rather than on its own, so a
            leading-space vocabulary is actually reachable. Without it the baseline is
            scored on splits it never makes in running text.
        out: JSON of scores, keyed by tokenizer name, with `@ctx` appended in that mode so
            both probes live in one file.
        force: Re-score cells already present.
    """
    segmenter = segment_in_context if in_context else segment
    # The two probes are different measurements of the same tokenizer and both are worth
    # keeping, so they need different keys -- without the suffix an in-context run silently
    # overwrites the bare one and the file no longer says which it holds.
    suffix = "@ctx" if in_context else ""
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
                name = f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}"
                path = os.path.join(TOKENIZERS, f"{name}.json.gz")
                if not os.path.exists(path) or (name + suffix in scores and not force):
                    continue
                tokenizer = load_tokenizer(path)
                a = score_arm(tokenizer, dataset, excl, segmenter)
                b = score_arm(tokenizer, dataset, incl, segmenter)
                scores[name + suffix] = {"lang": lang, "arm": arm, "trainer": trainer,
                                         "in_context": in_context,
                                         "exclude_single_tok": a, "credit_single_tok": b}
                print(f"  {arm:<18}{trainer:<9}{a['recall']:>8.4f}{a['precision']:>8.4f}"
                      f"{b['recall']:>8.4f}{b['precision']:>8.4f}"
                      f"{100 * a['single_token_share']:>8.1f}%{a['scored']:>8,}", flush=True)
                with open(out, "w") as f:
                    json.dump(scores, f, indent=2, sort_keys=True)
    print(f"\n[json] {out}: {len(scores)} cell(s)")


if __name__ == "__main__":
    app()
