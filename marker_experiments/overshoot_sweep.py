#!/usr/bin/env python3
r"""Does a bigger MinGram init make the marker vocabulary more morphological?

The marker schemes buy vocabulary back by not storing ` the` and `the` separately, and
spend it on pieces that sit inside a word rather than at its edge. Those pieces ought to
be more morpheme-shaped than the leading-space entries they replaced -- but every
morphology metric so far says the opposite. One explanation is that the pieces exist and
are simply not the ones a morpheme boundary needs, and the knob that controls how much
choice the trainer had is MinGram's BPE-init overshoot: the init is trained to
`f * additional_vocab_size` and the EM pass prunes back to target, so a larger `f` means
more candidates competing for each slot.

This sweeps `f` for one cell -- English, `bnd_wpd_extcaps`, MinGram -- and reports
compression beside both morphology metrics, so a trade can be seen if there is one.

Both metrics use the in-context segmentation (`morphalign.segment_in_context`), since the
bare-word probe scores a leading-space tokenizer on splits it never makes; and MorphScore
credits an unsplit word rather than dropping it, so the arms are scored on the same words.

f=1.15 is the repo default and reuses the grid's own tokenizer rather than retraining it.

    uv run python marker_experiments/overshoot_sweep.py
    uv run python marker_experiments/overshoot_sweep.py --factors 1.15,1.25 --workers 6
"""

import json
import os
import time

import cyclopts

from script_bpe.analysis.morphscore import MorphScore
from script_bpe.tokenizers.load import load_tokenizer

from marker_experiments.downstream.eval_goldfish import build_slice
from marker_experiments.downstream.train_matched import train_one
from marker_experiments.morphalign import (
    GOLD, ITERATIONS, METRIC, MODEL, MORPH_TOK_EVAL, SCALE, THRESHOLDS,
    _align_module, gold_subset, segment_in_context, write_segmented,
)
from marker_experiments.morphscore_boundary import LANGS as MS_LANGS, score_arm

# Registers the boundary pretokenizers, without which the tokenizers do not load.
import marker_experiments.downstream.boundary_tokenizer  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GRID_TOKENIZERS = os.path.join(HERE, "downstream", "tokenizers")
OUT_DIR = os.path.join(REPO, "results", "marker_overshoot")
GENERATED = os.path.join(HERE, "paper", "generated")

DEFAULT_F = 1.15          # the grid's setting, whose tokenizer is reused rather than retrained
TOTAL_VOCAB = 34_685

app = cyclopts.App()


def tokenizer_for(factor, lang, arm, corpus, total_vocab, workers):
    """The trained tokenizer for one overshoot factor, training it if it is not on disk."""
    if factor == DEFAULT_F:
        grid = os.path.join(GRID_TOKENIZERS, f"{corpus}_{arm}_mingram_v{total_vocab}.json.gz")
        if os.path.exists(grid):
            print(f"[f={factor}] reusing the grid's tokenizer", flush=True)
            return load_tokenizer(grid)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{corpus}_{arm}_mingram_f{factor}_v{total_vocab}.json.gz")
    if os.path.exists(path):
        print(f"[f={factor}] already trained", flush=True)
        return load_tokenizer(path)

    print(f"[f={factor}] training, init vocabulary {int(factor * 32975):,}", flush=True)
    t = time.time()
    tokenizer, info = train_one(arm, "mingram", corpus, total_vocab, workers, factor, None, None)
    tokenizer.save(path)
    print(f"[f={factor}] trained in {round(time.time() - t)}s, vocab {info['total_vocab']:,}", flush=True)
    return load_tokenizer(path)


def compression(tokenizer, lang):
    docs = build_slice(lang)
    chars = toks = 0
    for doc in docs:
        chars += len(doc)
        toks += len(tokenizer.encode(doc))
    return chars / toks


def morphalign(tokenizer, lang, tag, align):
    gold = os.path.join(MORPH_TOK_EVAL, "data", "morpho", GOLD[lang])
    seg_path = os.path.join(OUT_DIR, f"morphalign_{tag}.tsv")
    kept, dropped = write_segmented(tokenizer, gold, seg_path, segment_in_context)
    subset = gold_subset(gold, seg_path, seg_path.replace(".tsv", ".gold.tsv"))
    results, _ = align.evaluate_segmentations(
        subset, seg_path, THRESHOLDS, ITERATIONS, MODEL, skip_gold_train=True)
    return float(results[METRIC]) * SCALE, kept, dropped


@app.default
def main(
    lang: str = "en",
    arm: str = "bnd_wpd_extcaps",
    corpus: str = "fineweb_{lang}_5gb_quick",
    factors: str = "1.15,1.25,1.35,1.44",
    total_vocab: int = TOTAL_VOCAB,
    workers: int = 8,
    out: str = os.path.join(GENERATED, "overshoot_sweep.json"),
) -> None:
    """Train one cell at several overshoot factors and score each three ways.

    Args:
        lang: Language of the cell.
        arm: Arm to sweep.
        corpus: Corpus name pattern, `{lang}` substituted.
        factors: Comma-separated MinGram overshoot factors.
        total_vocab: Matched total vocabulary; every factor ends at this size.
        workers: Trainer processes.
        out: JSON of results, keyed by factor.
    """
    corpus_name = corpus.format(lang=lang)
    align = _align_module()
    scorer = MorphScore(language_subset=[MS_LANGS[lang]], stem_eq_lemma=True,
                        exclude_single_tok=False)
    dataset = scorer._get_filtered_dataset()
    rows = json.load(open(out)) if os.path.exists(out) else {}

    for factor in [float(x) for x in factors.split(",") if x.strip()]:
        tokenizer = tokenizer_for(factor, lang, arm, corpus_name, total_vocab, workers)
        ch_tok = compression(tokenizer, lang)
        ma, kept, dropped = morphalign(tokenizer, lang, f"{corpus_name}_{arm}_f{factor}", align)
        ms = score_arm(tokenizer, dataset, scorer, segment_in_context)
        rows[str(factor)] = {
            "lang": lang, "arm": arm, "overshoot": factor,
            "eval_chars_per_token": ch_tok, "morphalign": ma,
            "morphscore_recall": ms["recall"], "morphscore_precision": ms["precision"],
            "single_token_share": ms["single_token_share"],
            "morphalign_words": kept, "morphalign_dropped": dropped,
        }
        with open(out, "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
        print(f"[f={factor}] ch/tok={ch_tok:.4f}  MorphAlign={ma:.4f}  "
              f"MorphScore R={ms['recall']:.4f} P={ms['precision']:.4f}", flush=True)

    print(f"\n{'f':>6}{'ch/tok':>10}{'MorphAlign':>12}{'MS recall':>11}{'MS prec':>10}{'1-token':>10}")
    for key in sorted(rows, key=float):
        r = rows[key]
        print(f"{r['overshoot']:>6}{r['eval_chars_per_token']:>10.4f}{r['morphalign']:>12.4f}"
              f"{r['morphscore_recall']:>11.4f}{r['morphscore_precision']:>10.4f}"
              f"{100 * r['single_token_share']:>9.1f}%")
    print(f"\n[json] {out}")


if __name__ == "__main__":
    app()
