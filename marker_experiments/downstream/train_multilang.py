#!/usr/bin/env python3
"""Train the vocabulary-matched tokenizer grid on FineWeb, one language at a time.

The compression grids and the downstream runs currently use different tokenizers: the
grids train on FineWiki at a fixed `additional_vocab_size`, the downstream arms on 5 GB of
FineWeb English at a fixed *total* vocabulary of 34,685. So the compression numbers the
paper reports and the bits-per-byte numbers beside them do not describe the same
tokenizers. This trains one grid that both can read: every language, every arm, FineWeb
5 GB, total vocabulary 34,685.

Cells are committed as they finish
----------------------------------
A cell is skipped when its tokenizer file already exists, and every finished cell is
committed and pushed before the next one starts. Tokenizers are therefore tracked, not
gitignored: at roughly 1.1 MB each the whole 6x5 grid is about 33 MB, which is worth
paying so that no cell is ever trained twice. This also survives the container losing its
working tree mid-run, which is the normal case for long builds here -- on restart the
committed cells are simply skipped.

    # one language, BPE (the default)
    uv run python marker_experiments/downstream/train_multilang.py --lang en

    # both trainers
    uv run python marker_experiments/downstream/train_multilang.py \\
        --lang en --trainers bpe,mingram

    # minutes instead of hours, for iterating; writes to a separate corpus so the full
    # run's tokenizers are never touched
    uv run python marker_experiments/downstream/train_multilang.py --lang en --quick

    # the whole grid, in the order the corpora are cheapest to reuse
    uv run python marker_experiments/downstream/train_multilang.py --lang en,de,fi,ru,ar,ko

Running languages on several machines at once is the intended use. Every file a cell
commits is named after that cell -- its tokenizer, and its own manifest fragment under
paper/generated/manifest_parts -- so concurrent machines never write the same path and a
push rejected as non-fast-forward always rebases cleanly. Fold the fragments into
manifest.json at the end:

    uv run python marker_experiments/downstream/merge_manifests.py \
        --parts 'marker_experiments/paper/generated/manifest_parts/*.json'

Neither the pretokenized corpus nor the sampled-text cache is committed; both are large
and both are gitignored. On an ordinary machine that is all you need -- the first cell of
a language pays the source scan and every later cell reuses it. In a container whose
working tree is periodically restored from a snapshot, note that the restore removes
anything created after that snapshot whether it is gitignored or not, so the caches are
lost too and only the committed tokenizers carry progress forward. A cell that takes
longer than the interval between restores can never finish there; run it somewhere with a
persistent disk.

Evaluation compression is optional and off the critical path. `--eval-texts` takes a
pattern containing `{lang}`; a language with no such file trains and records nothing else.
This is deliberately loose because the evaluation slice is expected to move from the
held-out FineWiki documents to Goldfish, and the tokenizers should not need retraining
when it does.
"""

import json
import os
import subprocess
import sys
import time

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from marker_experiments.downstream.train_matched import eval_compression, train_one  # noqa: E402

TOKENIZERS = os.path.join(HERE, "tokenizers")
GENERATED = os.path.join(REPO, "marker_experiments", "paper", "generated")
MANIFEST_PARTS = os.path.join(GENERATED, "manifest_parts")
BRANCH = "claude/fineweb-space-neighbors-k10ufw"

# The four pretokenizers of the compression grid plus the caps variant the downstream runs
# use, so one grid covers both. Ordered so the cheapest cells finish first.
# bnd_wpd_caps is kept alongside bnd_wpd_extcaps as an ablation: the two differ only in
# whether the caps code sits inside or outside the span's markers, which decides whether
# the lowercase entry can be reused at all. Likely dropped once that is measured.
DEFAULT_ARMS = "plain,bnd_w,bnd_wp,bnd_wpd,bnd_wpd_caps,bnd_wpd_extcaps"
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


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)


def commit_cell(key, paths):
    """Commit and push one finished cell. Never fatal: a cell is worth keeping locally
    even when the remote is unreachable, and the next cell should still run.

    Rebases before retrying, because languages are meant to run on several machines at
    once and they all push here. A push rejected as non-fast-forward is then the normal
    case, not a network blip, and retrying the same push cannot fix it. Every file a cell
    commits is named after that cell, so concurrent machines never touch the same path and
    the rebase is always clean.
    """
    git("add", *paths)
    if not git("diff", "--cached", "--quiet").returncode:
        return  # nothing staged
    if git("commit", "-q", "-m", f"FineWeb 5GB matched grid: {key}").returncode:
        log(f"  commit failed for {key}")
        return
    for delay in (2, 4, 8, 16, 0):
        if git("push", "-q", "-u", "origin", BRANCH).returncode == 0:
            return
        pull = git("pull", "--rebase", "-q", "origin", BRANCH)
        if pull.returncode:
            git("rebase", "--abort")
            log(f"  rebase failed for {key}: {pull.stderr.strip()[:160]}")
            break
        if delay:
            time.sleep(delay)
    log(f"  WARNING: push failed for {key}; commit is local only")


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
    eval_texts: str = os.path.join(REPO, "marker_experiments", "eval_texts", "{lang}.json"),
    commit: bool = True,
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
        commit: Commit and push each finished cell. Turn off for a local dry run.
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

        part = record(key, info)
        cpt = info.get("eval_chars_per_token")
        log(f"{key}: vocab={info['total_vocab']:,} {round(time.time() - t0)}s"
            + (f" ch/tok={cpt:.4f} rt={info['roundtrip_failures']}" if cpt else " (no eval slice)"))
        if commit:
            commit_cell(key, [path, part])
        done += 1

    log(f"DONE: {done} trained, {skipped} already present")


if __name__ == "__main__":
    app()
