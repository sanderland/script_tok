#!/usr/bin/env python3
"""Pretokenize the corpus for a cell a running grid job has not reached yet.

A cell is a corpus build followed by a training run, and the build is the slower half for
a language like Korean: ~11 GB of sampled text, once per arm, because each arm's
pretokenizer produces a different chunking and therefore a different cache entry. A job
that runs its arms in sequence pays that cost serially even when the machine has cores to
spare.

The cache is keyed by (corpus, pretokenizer hash) and nothing else, so a second process
can build a future arm's corpus while the job works on the current one. When the job gets
there it finds the entry and goes straight to training.

Which arm: the LAST one the job still needs, in the order it runs them. Building the arm
it is about to reach would race it -- both write through a staging directory and rename,
so the loser's work is discarded rather than corrupted, but it is still an hour wasted.
The last one is the furthest from being contended.

    uv run python marker_experiments/downstream/prebuild_corpus.py --lang ko
    uv run python marker_experiments/downstream/prebuild_corpus.py --lang ko --dry-run
"""

import os
import sys
import time

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from script_bpe.corpus.base import PretokenizedCorpus  # noqa: E402
from script_bpe.corpus.registry import load_corpus_by_name  # noqa: E402

from marker_experiments.downstream.train_matched import make_pretokenizer  # noqa: E402
from marker_experiments.downstream.train_multilang import DEFAULT_ARMS, DEFAULT_TRAINERS, TOTAL_VOCAB  # noqa: E402

TOKENIZERS = os.path.join(HERE, "tokenizers")

app = cyclopts.App()


def missing_arms(lang, corpus, arms, trainers, total_vocab):
    """Arms with a trainer still to run, in the order the grid job runs them."""
    return [
        arm for arm in arms
        if not all(os.path.exists(os.path.join(TOKENIZERS, f"{corpus}_{arm}_{tr}_v{total_vocab}.json.gz"))
                   for tr in trainers)
    ]


@app.default
def main(
    lang: str = "ko",
    quick: bool = True,
    arms: str = DEFAULT_ARMS,
    trainers: str = DEFAULT_TRAINERS,
    total_vocab: int = TOTAL_VOCAB,
    workers: int = 0,
    dry_run: bool = False,
) -> None:
    """Build the pretokenized corpus for the last arm this language still needs.

    Args:
        lang: Language whose grid job is being helped.
        quick: Use the `_quick` corpus, matching the running jobs.
        arms: Comma-separated arms, in the order the job runs them.
        trainers: Comma-separated trainers; an arm counts as done only with all of them.
        total_vocab: Matched total vocabulary, for reading the tokenizer filenames.
        workers: Pretokenizing processes. 0 means one per core.
        dry_run: Report the arm and whether its corpus is already cached, then stop.
    """
    corpus = f"fineweb_{lang}_5gb" + ("_quick" if quick else "")
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    trainer_list = [t.strip() for t in trainers.split(",") if t.strip()]

    todo = missing_arms(lang, corpus, arm_list, trainer_list, total_vocab)
    if not todo:
        print(f"[prebuild] {lang}: every arm has its tokenizers, nothing to build")
        return

    # Latest first, so the job is furthest from contending for it, and skipping whatever
    # is already cached -- otherwise a second run finds only the arm the first one built
    # and stops, leaving the middle of the queue cold. The first missing arm is the one
    # the job is on or about to start, so it is left alone unless it is all that is left.
    candidates = list(reversed(todo[1:])) or todo
    for arm in candidates:
        pt = make_pretokenizer(arm)
        cached = os.path.join(PretokenizedCorpus.DEFAULT_BASE_PATH, corpus, pt.hash())
        if not os.path.exists(os.path.join(cached, "metadata.json")):
            break
        print(f"[prebuild] {lang}/{arm}: corpus {pt.hash()} already cached, looking earlier")
    else:
        print(f"[prebuild] {lang}: every buildable arm is cached, nothing to do")
        return

    print(f"[prebuild] {lang}: {len(todo)} arm(s) left {todo}, building {arm} ({pt.hash()})",
          flush=True)
    if dry_run:
        return

    t = time.time()
    built = load_corpus_by_name(corpus, pt, num_workers=workers or (os.cpu_count() or 4))
    print(f"[prebuild] {lang}/{arm}: {built.metadata.get('unique_chunks'):,} unique chunks "
          f"in {round(time.time() - t)}s -- the grid job will skip straight to training")


if __name__ == "__main__":
    app()
