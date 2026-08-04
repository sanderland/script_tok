#!/usr/bin/env python3
r"""MorphAlign for the boundary-marker arms, on the three UniMorph-evaluated languages.

The hybrid paper reports MorphAlign as IBM1 alignment at threshold 0.01 times 100
(`paper_utils/hybrid/generate_morphalign_scatter.py`). That pipeline segments each gold
word with the tokenizer and hands the segmentation to `eval/morph-tok-eval/align.py`,
which asserts the segments concatenate back to the word. Boundary tokenizers break that
assert twice over: their tokens carry `<|>` markers, and under the caps schemes a cased
word is stored lowercased with a separate case code, so the pieces spell `laufen` where
the gold word is `Laufen`.

Stripping is therefore a preprocessing step, not a detail:

  * marker and case-code atomic tokens are dropped from each token's core
  * the case the codes stood for is reapplied to the assembled string, then re-split at
    the original segment offsets, so the segmentation is unchanged and only the casing
    moves
  * a token that was nothing but markers or a code contributes no characters and is
    dropped -- a segment boundary with no text either side is not a boundary

What that leaves is comparable to the plain arm's segmentation, because it is the same
sequence of character spans. It is worth knowing what the codes do to it: `plain` splits
`Laufen` as `L|aufen`, since the capital and the lowercase stem are different vocabulary
entries, while `bnd_wpd_extcaps` emits one segment plus a code. A spurious boundary after
the first letter is exactly the kind of thing MorphAlign is sensitive to.

    uv run python marker_experiments/morphalign.py
    uv run python marker_experiments/morphalign.py --langs de --arms plain,bnd_wpd_extcaps
"""

import importlib.util
import json
import os

import cyclopts

from script_bpe.tokenizers.load import load_tokenizer

# Registers the boundary pretokenizers, without which the tokenizers do not load.
import marker_experiments.downstream.boundary_tokenizer  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOKENIZERS = os.path.join(HERE, "downstream", "tokenizers")
GENERATED = os.path.join(HERE, "paper", "generated")
SEGMENTED = os.path.join(REPO, "results", "marker_morphalign")
MORPH_TOK_EVAL = os.path.join(REPO, "eval", "morph-tok-eval")

# The hybrid paper's settings, so the numbers sit on the same axis as its table.
THRESHOLDS = [0.01]
ITERATIONS = 10
MODEL = "IBM1"
METRIC = "test-morpho-score-mean-split-0.01-IBM1"
SCALE = 100.0

GOLD = {
    "en": "eng-unimorph2uniseg_CELEX.tsv",
    "de": "deu-unimorph2uniseg_CELEX.tsv",
    "fi": "fin-unimorph2uniseg_morphynet.tsv",
}
ARMS = ["plain", "bnd_w", "bnd_w_extcaps", "bnd_wp", "bnd_wp_extcaps",
        "bnd_wpd", "bnd_wpd_extcaps", "bnd_wpd_caps"]

app = cyclopts.App()


def _align_module():
    spec = importlib.util.spec_from_file_location(
        "morph_tok_eval_align", os.path.join(MORPH_TOK_EVAL, "align.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    pieces, at, code, seen = [], 0, None, []
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
        if span[1] <= start or span[0] >= end:
            continue
        if shift is not None and shift in ids:
            code = "shift"
        if caps is not None and caps in ids:
            code = "caps"
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


def write_segmented(tokenizer, gold_path, out_path, segmenter=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    segmenter = segmenter or segment
    kept = dropped = 0
    with open(gold_path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            word, tag, _segments = line.rstrip("\n").split("\t")
            pieces = segmenter(tokenizer, word)
            if pieces is None:
                dropped += 1
                continue
            kept += 1
            print(word, tag, "|".join(pieces), sep="\t", file=f_out)
    return kept, dropped


def gold_subset(gold_path, segmented_path, out_path):
    """The gold rows matching what was segmented, so both sides align row for row."""
    with open(segmented_path, encoding="utf-8") as f:
        words = {line.split("\t")[0] for line in f}
    with open(gold_path, encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.split("\t")[0] in words:
                f_out.write(line)
    return out_path


@app.default
def main(
    langs: str = ",".join(GOLD),
    arms: str = ",".join(ARMS),
    trainers: str = "bpe,mingram",
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    in_context: bool = False,
    out: str = os.path.join(GENERATED, "morphalign.json"),
    force: bool = False,
) -> None:
    """Score every available arm, caching by cell so a re-run only adds what is new.

    Args:
        langs: Comma-separated languages; each needs a gold file in GOLD.
        arms: Comma-separated arms to score.
        trainers: Comma-separated trainers.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        in_context: Segment each gold word inside a carrier phrase instead of on its own.
            Without it a leading-space vocabulary cannot match, so the baseline is scored
            on splits it never makes in running text -- see segment_in_context.
        out: JSON of scores, keyed by tokenizer name, with `@ctx` appended in this mode so
            both probes live in one file.
        force: Re-score cells already present.
    """
    segmenter = segment_in_context if in_context else segment
    suffix = "@ctx" if in_context else ""
    align = _align_module()
    scores = json.load(open(out)) if os.path.exists(out) else {}

    for lang in [x.strip() for x in langs.split(",") if x.strip()]:
        gold_path = os.path.join(MORPH_TOK_EVAL, "data", "morpho", GOLD[lang])
        for arm in [x.strip() for x in arms.split(",") if x.strip()]:
            for trainer in [x.strip() for x in trainers.split(",") if x.strip()]:
                name = f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}"
                key = name + suffix
                path = os.path.join(TOKENIZERS, f"{name}.json.gz")
                if not os.path.exists(path) or (key in scores and not force):
                    continue
                seg_path = os.path.join(SEGMENTED, f"{key.replace('@', '_')}.tsv")
                kept, dropped = write_segmented(load_tokenizer(path), gold_path, seg_path, segmenter)
                subset = gold_subset(gold_path, seg_path, seg_path.replace(".tsv", ".gold.tsv"))
                results, _ = align.evaluate_segmentations(
                    subset, seg_path, THRESHOLDS, ITERATIONS, MODEL, skip_gold_train=True)
                scores[key] = {
                    "lang": lang, "arm": arm, "trainer": trainer, "in_context": in_context,
                    "morphalign": float(results[METRIC]) * SCALE,
                    "words": kept, "dropped": dropped,
                }
                print(f"{key:<52} MorphAlign={scores[key]['morphalign']:7.4f} "
                      f"({kept:,} words, {dropped} dropped)", flush=True)
                with open(out, "w") as f:
                    json.dump(scores, f, indent=2, sort_keys=True)

    print(f"\n[json] {out}: {len(scores)} cell(s)")


if __name__ == "__main__":
    app()
