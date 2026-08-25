"""Raw, *order-preserving* text sources for n-gram training and evaluation.

`PretokenizedCorpus` cannot be used here. It stores a `Counter` of pretokenized chunks,
so document order and cross-chunk order are gone by construction -- fine for BPE/Unigram
training, which only ever needs chunk frequencies, and fatal for an n-gram model, whose
whole subject is sequence. So this module streams raw text instead.

Three source specs:

    sampled:<corpus_name>   the registry's own `_sampled_text` block cache, which already
                            holds the exact documents a `fineweb_*` corpus was built from,
                            in order. Free if the corpus was ever built on this machine.
    file:<path>             .jsonl (one JSON string per line, the block-cache format) or
                            .txt (one document per line).
    hf:<dataset>[/<config>][#<split>]     streamed from the Hub.

Train and eval slices are contiguous and disjoint: eval takes the first `eval_chars` of
the stream, train the next `train_chars`. Contiguous rather than interleaved on purpose --
adjacent documents in a web corpus are often near-duplicates, and striding them across the
split would leak eval text into training and flatter every tokenizer equally but wrongly.
"""

import glob
import json
import os
from typing import Iterator

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.registry import SAMPLE_CACHE_DIRNAME


def _iter_jsonl(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_sampled(corpus_name: str, base_dir: str) -> Iterator[str]:
    """Documents from the registry's sampled-text block cache, in sampling order."""
    root = os.path.join(base_dir, SAMPLE_CACHE_DIRNAME)
    matches = sorted(glob.glob(os.path.join(root, f"{corpus_name.replace(':', '_')}_*")))
    matches = [m for m in matches if os.path.exists(os.path.join(m, "manifest.json"))]
    if not matches:
        raise FileNotFoundError(
            f"no sampled-text cache for {corpus_name!r} under {root}. It is written when a "
            f"fineweb_* corpus is built; build the corpus first, or pass an hf:/file: spec."
        )
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} sampled-text caches match {corpus_name!r} ({', '.join(map(os.path.basename, matches))}); "
            f"they differ in sample size or seed, so pick one with file: explicitly."
        )
    with open(os.path.join(matches[0], "manifest.json")) as f:
        blocks = json.load(f)["blocks"]
    for name in blocks:
        yield from _iter_jsonl(os.path.join(matches[0], name))


def _iter_hf(spec: str) -> Iterator[str]:
    from datasets import load_dataset

    body, _, split = spec.partition("#")
    dataset_name, _, config = body.partition("/")
    kwargs = {"name": config} if config else {}
    dataset = load_dataset(dataset_name, split=split or "train", streaming=True, **kwargs)
    for row in dataset:
        yield row["text"]


def iter_documents(spec: str, base_dir: str = PretokenizedCorpus.DEFAULT_BASE_PATH) -> Iterator[str]:
    """Yield documents from a source spec, in a stable order."""
    kind, _, rest = spec.partition(":")
    if kind == "sampled":
        yield from _iter_sampled(rest, base_dir)
    elif kind == "file":
        if rest.endswith(".jsonl"):
            yield from _iter_jsonl(rest)
        else:
            with open(rest, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield line.rstrip("\n")
    elif kind == "hf":
        yield from _iter_hf(rest)
    else:
        raise ValueError(f"unknown text source spec {spec!r}; expected sampled:/file:/hf: prefix")


def take_split(spec: str, *, eval_chars: int, train_chars: int,
               base_dir: str = PretokenizedCorpus.DEFAULT_BASE_PATH) -> tuple[list[str], list[str]]:
    """Read one pass of `spec` into disjoint (eval_docs, train_docs) slices.

    Documents are never split, so the realized sizes overshoot the budgets by at most one
    document each. Raises if the source runs dry before both budgets are met -- a silently
    short training corpus would look like a real result.
    """
    eval_docs: list[str] = []
    train_docs: list[str] = []
    seen_eval = seen_train = 0
    for doc in iter_documents(spec, base_dir=base_dir):
        if not doc:
            continue
        if seen_eval < eval_chars:
            eval_docs.append(doc)
            seen_eval += len(doc)
        elif seen_train < train_chars:
            train_docs.append(doc)
            seen_train += len(doc)
        else:
            break
    if seen_eval < eval_chars or seen_train < train_chars:
        raise ValueError(
            f"{spec!r} yielded {seen_eval:,} eval + {seen_train:,} train chars, short of the "
            f"requested {eval_chars:,} + {train_chars:,}. Use a larger source or smaller budgets."
        )
    return eval_docs, train_docs
