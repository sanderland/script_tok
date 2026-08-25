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
    parquet:<path-or-glob>[#<column>]     parquet shards, `text` column by default. Covers
                            both a hand-downloaded FineWeb shard and nanochat's own data
                            directory, and is the practical way to read a Hub dataset whose
                            file list is too large for `load_dataset` to enumerate quickly.
    hf:<dataset>[/<config>][#<split>]     streamed from the Hub. Convenient, but on a
                            repo with tens of thousands of shards the streaming resolver
                            can take longer to list files than the eval takes to run; reach
                            for `parquet:` on those.

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


def _iter_parquet(spec: str) -> Iterator[str]:
    """Documents from parquet shards, in filename order then row order.

    Read in record batches via pyarrow rather than with a whole-file reader: a FineWeb
    shard is ~2 GB and the caller normally wants a slice off the front, so materializing
    the column would cost more memory than the n-gram counting it feeds.
    """
    import pyarrow.parquet as pq

    pattern, _, column = spec.partition("#")
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        raise FileNotFoundError(f"no parquet files match {pattern!r}")
    column = column or "text"
    for path in paths:
        parquet = pq.ParquetFile(path)
        if column not in parquet.schema_arrow.names:
            raise ValueError(f"{path} has no column {column!r}; found {parquet.schema_arrow.names}")
        for batch in parquet.iter_batches(batch_size=2048, columns=[column]):
            yield from batch.column(0).to_pylist()


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
    elif kind == "parquet":
        yield from _iter_parquet(rest)
    elif kind == "hf":
        yield from _iter_hf(rest)
    else:
        raise ValueError(f"unknown text source spec {spec!r}; expected sampled:/file:/parquet:/hf: prefix")


def take_split(spec: str, *, eval_chars: int, train_chars: int, skip_chars: int = 0,
               base_dir: str = PretokenizedCorpus.DEFAULT_BASE_PATH) -> tuple[list[str], list[str]]:
    """Read one pass of `spec` into disjoint (eval_docs, train_docs) slices.

    Documents are never split, so the realized sizes overshoot the budgets by at most one
    document each. Raises if the source runs dry before both budgets are met -- a silently
    short training corpus would look like a real result.

    `skip_chars` discards that much text off the front first, which is how to get a
    replicate on disjoint text. That matters more here than it might look: the differences
    between tokenizer arms are small at higher orders, so the honest question is whether a
    ranking survives being measured on a different sample, and a replicate is the only way
    to answer it.
    """
    eval_docs: list[str] = []
    train_docs: list[str] = []
    seen_eval = seen_train = skipped = 0
    for doc in iter_documents(spec, base_dir=base_dir):
        if not doc:
            continue
        if skipped < skip_chars:
            skipped += len(doc)
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
            f"{spec!r} yielded {seen_eval:,} eval + {seen_train:,} train chars after skipping "
            f"{skipped:,}, short of the requested {eval_chars:,} + {train_chars:,}. "
            f"Use a larger source or smaller budgets."
        )
    return eval_docs, train_docs
