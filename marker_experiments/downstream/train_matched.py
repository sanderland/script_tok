#!/usr/bin/env python3
"""Train vocabulary-matched tokenizers for the downstream LM comparison.

Why this exists instead of reusing `marker_experiments/tokenizers/*`
-------------------------------------------------------------------
The compression grid fixed `additional_vocab_size=32768` for every arm, so the arms
end up with *different* total vocabularies, because the boundary marker and the caps
codes are extra atomic tokens:

    plain          1710 atomic + 32768 = 34478
    bnd_wpd        1711 atomic + 32768 = 34479
    bnd_wpd_caps   1713 atomic + 32768 = 34481

That is the right control for a compression measurement -- each arm gets the same
number of *learned* merges -- but it is the wrong control downstream, where the
vocabulary size sets the embedding and unembedding shapes, hence the parameter count,
hence nanochat's compute-optimal token horizon. Downstream we match the *total*, and
let the learned budget absorb the difference:

    additional_vocab_size = total_vocab - len(pretokenizer.atomic_tokens)

The default total is 34,685, which is what the MinGram downstream table used, so these
runs sit on the same axis as that table.

Corpus
------
Default `fineweb_en_5gb`, the corpus MinGram's downstream tokenizers were trained on,
and web text like the ClimbMix corpus the LM trains on. `finewiki_en_1gb` reproduces
the domain of our compression grid instead; expect slightly different absolute
chars/token from the numbers in the paper, because the grid withheld 500 evaluation
documents from training and the registry corpus does not.

Usage
-----
    uv run python marker_experiments/downstream/train_matched.py \
        --arms plain,bnd_wpd,bnd_wpd_caps --trainer bpe --workers 90

    uv run python marker_experiments/downstream/train_matched.py \
        --arms plain,bnd_wpd --trainer mingram --workers 90
"""

import json
import os
import sys
import time

import cyclopts

from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig

from marker_experiments.boundary_pretokenizer import ALL_VARIANTS, get_boundary_pretokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "tokenizers")
MANIFEST = os.path.join(HERE, "manifest.json")

# 34,685 is the matched vocabulary of the MinGram downstream table.
DEFAULT_TOTAL_VOCAB = 34_685
# f=1.15 is the repo default and the setting the overshoot sweep settled on.
DEFAULT_OVERSHOOT = 1.15

app = cyclopts.App()


def make_pretokenizer(arm: str):
    """`plain` is the stock SCRIPT-v3 baseline; everything else is a boundary variant."""
    if arm == "plain":
        return get_pretokenizer("scriptenc3_cb")
    return get_boundary_pretokenizer(arm)


def build_corpus(corpus_name, pt, base_dir, text_file, workers=None):
    """Registry corpus, or -- with `text_file` -- a throwaway corpus from one local file.

    The local-file path exists so the vocabulary-matching logic and the full
    train -> save -> load -> adapter chain can be exercised in seconds, before
    committing to a multi-GB download. It is not a research setting: at this size the
    vocabulary cannot be filled, so `total_vocab` is not reached and the matched-size
    assertion is skipped.

    `workers` applies to pretokenizing the corpus as well as to training. Without it the
    registry falls back to CORPUS_BUILD_DEFAULT_WORKERS (16), which is what made a
    fineweb_en_5gb build take 8.6 hours on a 288-core node.
    """
    if not text_file:
        # `--corpus-base-dir` is optional and documented as defaulting to the repo's
        # cache. Forwarding base_dir=None would override load_corpus_by_name's own
        # default rather than fall back to it, and PretokenizedCorpus then fails with
        # `TypeError: expected str ... not NoneType` when it joins the path.
        if base_dir is None:
            return load_corpus_by_name(corpus_name, pt, num_workers=workers)
        return load_corpus_by_name(corpus_name, pt, base_dir=base_dir, num_workers=workers)
    from script_bpe.corpus.base import PretokenizedCorpus
    from script_bpe.corpus.registry import normalize_whitespace

    with open(text_file, encoding="utf-8") as f:
        text = normalize_whitespace(f.read())
    return PretokenizedCorpus.from_text_batches(
        name=corpus_name,
        base_path=base_dir or os.path.join(HERE, "corpora"),
        pretokenizer=pt,
        text_batches=iter([[text]]),
        num_workers=1,
    )


def train_one(arm, trainer_name, corpus_name, total_vocab, workers, overshoot, base_dir, text_file):
    pt = make_pretokenizer(arm)
    additional = total_vocab - len(pt.atomic_tokens)
    assert additional > 0, f"{arm}: total_vocab {total_vocab} below {len(pt.atomic_tokens)} atomic tokens"

    corpus = build_corpus(corpus_name, pt, base_dir, text_file, workers=workers)

    t = time.time()
    if trainer_name == "bpe":
        cfg = BPETrainerConfig(additional_vocab_size=additional, num_workers=workers)
        tokenizer = BPETrainer(pt, corpus, cfg).train()
    elif trainer_name == "mingram":
        cfg = MinGramTrainerConfig(
            additional_vocab_size=additional, num_workers=workers, overshoot_factor=overshoot
        )
        tokenizer = MinGramTrainer(pt, corpus, cfg).train()
    else:
        raise ValueError(f"unknown trainer {trainer_name!r}")
    seconds = time.time() - t

    return tokenizer, {
        "arm": arm,
        "trainer": trainer_name,
        "corpus": corpus_name,
        "atomic_vocab": len(pt.atomic_tokens),
        "additional_vocab_size": additional,
        "total_vocab": len(tokenizer.tokens),
        "train_seconds": round(seconds),
        "overshoot_factor": overshoot if trainer_name == "mingram" else None,
    }


def eval_compression(tokenizer, eval_path):
    """chars/token and round-trip failures on the compression grid's held-out slice."""
    if not os.path.exists(eval_path):
        return {}
    with open(eval_path) as f:
        texts = json.load(f)
    chars = toks = fails = 0
    for text in texts:
        ids = tokenizer.encode(text)
        chars += len(text)
        toks += len(ids)
        if tokenizer.decode(ids) != text:
            fails += 1
    return {
        "eval_chars": chars,
        "eval_tokens": toks,
        "eval_chars_per_token": chars / toks,
        "roundtrip_failures": fails,
    }


@app.default
def main(
    arms: str = "plain,bnd_wpd,bnd_wpd_caps",
    trainer: str = "bpe",
    corpus: str = "fineweb_en_5gb",
    total_vocab: int = DEFAULT_TOTAL_VOCAB,
    workers: int = 8,
    overshoot: float = DEFAULT_OVERSHOOT,
    out_dir: str = OUT_DIR,
    corpus_base_dir: str | None = None,
    eval_texts: str | None = None,
    text_file: str | None = None,
    force: bool = False,
    manifest_path: str | None = None,
) -> None:
    """Train one vocabulary-matched tokenizer per arm and record a manifest.

    Args:
        arms: Comma-separated arms. `plain` is the SCRIPT-v3 baseline; the rest are
            boundary variants (bnd_w, bnd_wp, bnd_wpd and their _caps forms).
        trainer: `bpe` or `mingram`.
        corpus: Registry corpus name for tokenizer training (fineweb_en_5gb, finewiki_en_1gb, ...).
        total_vocab: Matched total vocabulary. Every arm ends at exactly this size.
        workers: Trainer worker processes. Set this to roughly the core count.
        overshoot: MinGram BPE-init overshoot factor (ignored for `bpe`).
        out_dir: Where the .json.gz tokenizers land.
        corpus_base_dir: Pretokenized-corpus cache dir (defaults to the repo's).
        eval_texts: Optional JSON list of held-out texts for a chars/token sanity check.
            `marker_experiments/eval_texts/en.json` is the compression grid's slice.
        manifest_path: Where to record the trained arms. Defaults to manifest.json next
            to this script. Running one arm per job needs a separate file per job: the
            manifest is rewritten by read-modify-write, so concurrent jobs would drop each
            other's entries. Merge them afterwards with merge_manifests.py.
        text_file: Train on this local text file instead of a registry corpus. Seconds
            rather than hours; for pipeline checks only, and vocabulary matching is not
            enforced because the vocabulary cannot be filled at that size.
        force: Retrain arms whose output file already exists.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest_file = manifest_path or MANIFEST
    manifest = json.load(open(manifest_file)) if os.path.exists(manifest_file) else {}
    if text_file:
        corpus = f"tiny_{os.path.basename(text_file).split('.')[0]}"

    trained = []
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        assert arm == "plain" or arm in ALL_VARIANTS, f"unknown arm {arm!r}"
        key = f"{corpus}_{arm}_{trainer}_v{total_vocab}"
        path = os.path.join(out_dir, f"{key}.json.gz")
        if os.path.exists(path) and not force:
            print(f"[train] {key}: exists, skipping", flush=True)
            continue

        tokenizer, info = train_one(
            arm, trainer, corpus, total_vocab, workers, overshoot, corpus_base_dir, text_file
        )
        trained.append(key)
        tokenizer.save(path)
        info["path"] = os.path.relpath(path, os.path.dirname(HERE))
        if eval_texts:
            info.update(eval_compression(tokenizer, eval_texts))
        manifest[key] = info
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        cpt = info.get("eval_chars_per_token")
        print(
            f"[train] {key}: vocab={info['total_vocab']:,} "
            f"(atomic {info['atomic_vocab']} + {info['additional_vocab_size']:,}) "
            f"{info['train_seconds']}s"
            + (f" ch/tok={cpt:.4f} rt_fail={info['roundtrip_failures']}" if cpt else ""),
            flush=True,
        )

    # A mismatch here silently biases the downstream comparison, so fail loudly.
    sizes = {
        k: v["total_vocab"]
        for k, v in manifest.items()
        if v["trainer"] == trainer and v["corpus"] == corpus
    }
    if text_file and len(set(sizes.values())) > 1:
        # A tiny corpus may not contain enough distinct pairs to fill the vocabulary,
        # in which case the arms legitimately stop at different sizes. Report, don't fail.
        print(f"[train] tiny corpus did not fill the vocabulary in every arm: {sizes}")
    elif len(set(sizes.values())) > 1:
        raise SystemExit(f"vocabulary sizes are NOT matched across arms: {sizes}")
    else:
        print(f"[train] {len(sizes)} {trainer} arms on {corpus}, "
              f"all at vocab {next(iter(sizes.values()), None)}")


if __name__ == "__main__":
    app()
    # Everything this script produces (the tokenizers and the manifest) is on disk by
    # now. Interpreter shutdown is not reliable here: multiprocessing's atexit handling
    # can block on workers the forkserver never reaped, which left one job idle for 18
    # minutes after its work had finished and would hold a node until its walltime.
    # Flush and leave rather than wait for that.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
