#!/usr/bin/env python3
"""Train the vocabulary-matched tokenizer grid on FineWeb, one language at a time.

One grid serves both halves of the paper: the intrinsic compression numbers and the
tokenizers the downstream runs pretrain on. Every language, every arm, 5 GB of FineWeb per
language, total vocabulary 34,685 -- so a compression figure and the bits-per-byte figure
beside it describe the same tokenizer.

A cell is skipped when its tokenizer file already exists, so a run that is interrupted
resumes where it stopped and languages can be split across machines. Neither the
pretokenized corpus nor the sampled-text cache is committed; both are large and both are
gitignored. The first cell of a language pays the source scan and every later cell reuses
it.

    # one language, BPE (the default)
    uv run python paper_utils/boundary/downstream/train_multilang.py --lang en

    # both trainers
    uv run python paper_utils/boundary/downstream/train_multilang.py \\
        --lang en --trainers bpe,mingram

    # the quick corpus sample: minutes instead of hours, and what the paper reports.
    # Writes to a separate corpus name, so it can never share a cache or a tokenizer path
    # with a full-sample run
    uv run python paper_utils/boundary/downstream/train_multilang.py --lang en --quick

    # the whole grid, in the order the corpora are cheapest to reuse
    uv run python paper_utils/boundary/downstream/train_multilang.py --lang en,de,fi,ru,ar,ko

Every file a cell writes is named after that cell -- its tokenizer, and its own manifest
fragment under paper/generated/manifest_parts -- so concurrent machines never write the
same path. Fold the fragments into manifest.json at the end:

    uv run python paper_utils/boundary/downstream/merge_manifests.py \
        --parts 'paper_utils/boundary/paper/generated/manifest_parts/*.json'

Evaluation compression is optional and off the critical path. `--eval-texts` takes a
pattern containing `{lang}`; a language with no such file trains and records nothing else.
"""

import json
import os
import sys
import time

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

from paper_utils.boundary.downstream.train_matched import eval_compression, train_one  # noqa: E402

TOKENIZERS = os.path.join(HERE, "tokenizers")
GENERATED = os.path.join(REPO, "paper_utils", "boundary", "paper", "generated")
MANIFEST_PARTS = os.path.join(GENERATED, "manifest_parts")

# Every arm of the grid: the three boundary scopes, each with and without the case codes,
# plus the baseline. Ordered so the cheapest cells finish first, and so an arm's two
# trainers run back to back while its pretokenized corpus is still warm.
DEFAULT_ARMS = (
    "plain,"
    "bnd_w,bnd_w_caps,"
    "bnd_wp,bnd_wp_caps,"
    "bnd_wpd,bnd_wpd_caps"
)
# Both trainers by default: the grid is the intrinsic comparison as well as the source of
# downstream tokenizers, and a BPE-only grid cannot answer whether the variant ordering
# survives a change of trainer. Cells are ordered arm-major, so an arm's two trainers run
# back to back while its pretokenized corpus is still warm.
DEFAULT_TRAINERS = "bpe,mingram"
DEFAULT_LANGS = "en,de,fi,ru,ar,ko"
TOTAL_VOCAB = 34_685


app = cyclopts.App()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def record(key, info):
    """Write this cell's manifest entry to its own file, and return that path.

    One file per cell, not one shared manifest. Languages are meant to run on several
    machines that all push to this branch, and a single manifest rewritten by
    read-modify-write is the one path two of them would both touch -- so every rebase
    would conflict on it and one machine's entries would be lost. Fold the parts into
    manifest.json afterwards with merge_manifests.py.
    """
    os.makedirs(MANIFEST_PARTS, exist_ok=True)
    path = os.path.join(MANIFEST_PARTS, f"{key}.json")
    with open(path, "w") as f:
        json.dump({key: info}, f, indent=2, sort_keys=True)
    return path


@app.default
def main(
    lang: str = "en",
    arms: str = DEFAULT_ARMS,
    trainers: str = DEFAULT_TRAINERS,
    quick: bool = False,
    total_vocab: int = TOTAL_VOCAB,
    workers: int = 0,
    overshoot: float = 1.15,
    eval_texts: str = os.path.join(REPO, "paper_utils", "boundary", "eval_texts", "{lang}.json"),
) -> None:
    """Train every (language, arm, trainer) cell that has no tokenizer yet.

    Args:
        lang: Comma-separated languages. Each maps to the registry corpus fineweb_<lang>_5gb.
        arms: Comma-separated arms. `plain` is the SCRIPT-v3 baseline; the rest are
            boundary variants.
        trainers: Comma-separated trainers: bpe, mingram, or both. Both by default;
            pass `--trainers bpe` for the cheaper half.
        quick: Sample by reading until full instead of scanning the whole source, which
            takes minutes rather than the ~1.9 hours a full pass costs. The sample is not
            uniform over the source, so this is for iterating, not for reported results.
            Writes to fineweb_<lang>_5gb_quick, a separate corpus and separate tokenizer
            filenames, so a quick run never overwrites or is confused with a full one.
        total_vocab: Matched total vocabulary. Every cell ends at exactly this size.
        workers: Trainer and corpus-build worker processes. 0 means one per core.
        overshoot: MinGram BPE-init overshoot factor (ignored for bpe).
        eval_texts: Pattern for the held-out slice, containing `{lang}`. Missing files are
            skipped, so a language with no slice still trains.
    """
    os.makedirs(TOKENIZERS, exist_ok=True)
    n_workers = workers or (os.cpu_count() or 4)
    langs = [x.strip() for x in lang.split(",") if x.strip()]
    arm_list = [x.strip() for x in arms.split(",") if x.strip()]
    trainer_list = [x.strip() for x in trainers.split(",") if x.strip()]

    todo = [
        (lg, arm, tr)
        for lg in langs
        for arm in arm_list          # arm outer, trainer inner: both trainers of an arm
        for tr in trainer_list       # share one pretokenized corpus, which is not cached
    ]                                # across a working-tree wipe.
    log(f"{len(todo)} cell(s), fineweb 5gb{' quick' if quick else ''}, vocab {total_vocab:,}, "
        f"{n_workers} workers, trainer(s) {'+'.join(trainer_list)}")

    done = skipped = 0
    for lg, arm, tr in todo:
        corpus = f"fineweb_{lg}_5gb" + ("_quick" if quick else "")
        key = f"{corpus}_{arm}_{tr}_v{total_vocab}"
        path = os.path.join(TOKENIZERS, f"{key}.json.gz")
        if os.path.exists(path):
            log(f"{key}: exists, skipping")
            skipped += 1
            continue

        log(f"{key}: starting")
        t0 = time.time()
        tokenizer, info = train_one(
            arm, tr, corpus, total_vocab, n_workers, overshoot, None, None
        )
        tokenizer.save(path)
        info["lang"] = lg
        info["path"] = os.path.relpath(path, os.path.dirname(HERE))

        slice_path = eval_texts.format(lang=lg)
        if os.path.exists(slice_path):
            info["eval_slice"] = os.path.relpath(slice_path, REPO)
            info.update(eval_compression(tokenizer, slice_path))

        record(key, info)
        cpt = info.get("eval_chars_per_token")
        log(f"{key}: vocab={info['total_vocab']:,} {round(time.time() - t0)}s"
            + (f" ch/tok={cpt:.4f} rt={info['roundtrip_failures']}" if cpt else " (no eval slice)"))
        done += 1

    log(f"DONE: {done} trained, {skipped} already present")


if __name__ == "__main__":
    app()
