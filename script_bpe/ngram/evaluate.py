"""Held-out bits-per-byte for a tokenizer, under an n-gram language model.

The measurement, end to end:

    train docs --encode--> token ids --fit--> modified Kneser-Ney LM
    eval  docs --encode--> token ids --score--> total bits / true UTF-8 bytes

`n` interpolates between the two kinds of tokenizer evaluation we already have. At n=1 it
is close to the familiar intrinsic metrics: unigram entropy over the tokenizer's own
distribution, priced per byte. Every increment of n lets the model use sequential
structure that only a trained LM would otherwise see. So the family spans intrinsic to
extrinsic, and where along it the ranking stabilises is itself the interesting result.

**The denominator is the true UTF-8 length of the held-out text**, measured before
tokenization. That is deliberately not what nanochat does: nanochat divides by the summed
byte length of each target token decoded *in isolation*, which under-counts for any
scheme that reconstructs a character from two touching tokens, and pynanochat has to
correct for it with a separately measured `byte_factor` (see `runner.measure_byte_factor`).
Scoring against the real text sidesteps that entirely -- there is no factor to measure and
nothing to get wrong. It does mean a *lossy* tokenizer would be flattered by having less
to predict against an unchanged denominator, so `roundtrip_ok` is reported alongside.

What this metric is not: a substitute for training a model. Context here is measured in
tokens, so a trigram over a high-compression vocabulary sees several times more bytes of
history than one over a byte-level vocabulary. That mirrors how a fixed-context
transformer rewards compression, which is part of why this tracks downstream results
better than compression alone -- but it does structurally favour long tokens, and an
n-gram can never exploit within-token composition the way a transformer partly does.
"""

import hashlib
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from script_bpe.ngram.counts import build_stream
from script_bpe.ngram.kn import KneserNeyLM
from script_bpe.utils import create_logger, mp_ctx

CACHE_DIRNAME = "ngram_encoded"

_WORKER_MODEL = None


@dataclass
class VocabGeometry:
    """Where BOS/EOS live relative to the tokenizer's own id space.

    Specials are allocated *above* the tokenizer's ids rather than reusing whatever
    specials a tokenizer happens to define, so every tokenizer family is measured under
    the same convention and no tokenizer is charged for a special another one lacks.
    """

    max_id: int          # one past the largest real token id
    alphabet_size: int   # distinct emittable symbols: real tokens + EOS (BOS is never predicted)
    eos_id: int
    bos_id: int
    radix: int

    @classmethod
    def of(cls, model) -> "VocabGeometry":
        max_id = max(model.tokens) + 1
        return cls(max_id=max_id, alphabet_size=len(model.tokens) + 1,
                   eos_id=max_id, bos_id=max_id + 1, radix=max_id + 2)


@dataclass
class NgramResult:
    tokenizer_id: str
    order: int
    vocab_size: int
    train_docs: int
    eval_docs: int
    train_tokens: int
    eval_tokens: int
    eval_bytes: int
    bits: float
    bpb: float                 # the headline: bits per true UTF-8 byte
    bits_per_token: float
    tokens_per_byte: float
    oov_token_rate: float      # eval tokens whose id never occurred in training
    roundtrip_ok: bool
    ngram_types: dict = field(default_factory=dict)
    # Bits and bytes per held-out document. The metric is deterministic given the text, so
    # there is no seed to average over the way the downstream sweep does -- the only honest
    # way to ask whether two tokenizers differ is to resample the documents. Keeping the
    # per-document split makes that a paired test: the same documents scored by both arms,
    # which removes document difficulty (by far the largest source of variance) from the
    # comparison. Excluded from `as_row`; written alongside the TSV instead.
    doc_bits: list[float] = field(default_factory=list, repr=False)
    doc_bytes: list[int] = field(default_factory=list, repr=False)

    def as_row(self) -> dict:
        row = asdict(self)
        row["ngram_types"] = ",".join(f"{k}:{v}" for k, v in sorted(self.ngram_types.items()))
        row.pop("doc_bits", None)
        row.pop("doc_bytes", None)
        return row


def _init_worker(tokenizer_path: str, tokenizer_class: str):
    global _WORKER_MODEL
    import importlib

    module_name, _, cls_name = tokenizer_class.rpartition(".")
    _WORKER_MODEL = getattr(importlib.import_module(module_name), cls_name).load(tokenizer_path)


def _encode_batch(texts: list[str]) -> list[np.ndarray]:
    return [np.asarray(_WORKER_MODEL.encode(t), dtype=np.int32) for t in texts]


def encode_documents(model, docs: list[str], *, tokenizer_path: str | None = None,
                     tokenizer_class: str | None = None, num_workers: int = 1,
                     batch_size: int = 256) -> list[np.ndarray]:
    """Encode documents to int32 id arrays, optionally across a worker pool.

    Workers reload the tokenizer from disk rather than receiving the live object: the
    models hold large mappings and are not guaranteed picklable, and reloading once per
    worker is cheap next to the encoding itself.
    """
    if num_workers <= 1 or len(docs) < batch_size:
        return [np.asarray(model.encode(t), dtype=np.int32) for t in docs]
    if not (tokenizer_path and tokenizer_class):
        raise ValueError("parallel encoding needs tokenizer_path and tokenizer_class for the workers")
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
    pool = mp_ctx.Pool(num_workers, initializer=_init_worker, initargs=(tokenizer_path, tokenizer_class))
    try:
        return [arr for batch in pool.map(_encode_batch, batches) for arr in batch]
    finally:
        from script_bpe.utils import shutdown_pool

        shutdown_pool(pool)


def _cache_key(tokenizer_path: str, spec: str, eval_chars: int, train_chars: int) -> str:
    with open(tokenizer_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    # Keyed on the tokenizer's *content*: retraining an arm rewrites the same filename, and
    # a path-keyed cache would hand the new tokenizer the old encoding.
    return hashlib.sha256(f"{digest}:{spec}:{eval_chars}:{train_chars}".encode()).hexdigest()[:16]


def _load_cached(path: str):
    data = np.load(path)
    def split(prefix):
        flat, offs = data[f"{prefix}_flat"], data[f"{prefix}_offsets"]
        return [flat[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
    return split("eval"), split("train")


def _save_cached(path: str, eval_ids, train_ids):
    payload = {}
    for prefix, arrays in (("eval", eval_ids), ("train", train_ids)):
        lengths = np.array([len(a) for a in arrays], dtype=np.int64)
        payload[f"{prefix}_flat"] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.int32)
        payload[f"{prefix}_offsets"] = np.concatenate([[0], np.cumsum(lengths)])
    # Written to a temp name and renamed: a sweep runs arms concurrently, and a reader must
    # never see a half-written cache. np.savez appends .npz unless the target is a file
    # object, so hand it one.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, path)


def evaluate_ngram_bpb(
    model,
    *,
    eval_docs: list[str],
    train_docs: list[str],
    orders: list[int],
    tokenizer_id: str = "tokenizer",
    tokenizer_path: str | None = None,
    tokenizer_class: str | None = None,
    num_workers: int = 1,
    cache_dir: str | None = None,
    spec: str = "",
    check_roundtrip_docs: int = 32,
    logger=None,
) -> list[NgramResult]:
    """Fit and score one n-gram LM per requested order, reusing a single encoding pass.

    Encoding dominates the wall clock (script_bpe encoders are pure Python), so it happens
    once and every order reuses it; the counting and scoring are NumPy and comparatively
    free. Orders are returned in the order requested.
    """
    logger = logger or create_logger("ngram")
    geom = VocabGeometry.of(model)

    cached = None
    if cache_dir and tokenizer_path:
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(tokenizer_path, spec, sum(map(len, eval_docs)), sum(map(len, train_docs)))
        cached = os.path.join(cache_dir, f"enc_{key}.npz")
    if cached and os.path.exists(cached):
        logger.info(f"reusing cached encoding {cached}")
        eval_ids, train_ids = _load_cached(cached)
    else:
        logger.info(f"encoding {len(eval_docs):,} eval + {len(train_docs):,} train docs on {num_workers} worker(s)")
        eval_ids = encode_documents(model, eval_docs, tokenizer_path=tokenizer_path,
                                    tokenizer_class=tokenizer_class, num_workers=num_workers)
        train_ids = encode_documents(model, train_docs, tokenizer_path=tokenizer_path,
                                     tokenizer_class=tokenizer_class, num_workers=num_workers)
        if cached:
            _save_cached(cached, eval_ids, train_ids)

    # A lossy tokenizer gets less to predict against an unchanged byte denominator, which
    # would look like an improvement. Spot-check rather than assume.
    roundtrip_ok = all(model.decode(ids.tolist()) == doc
                       for doc, ids in zip(eval_docs[:check_roundtrip_docs], eval_ids[:check_roundtrip_docs]))
    if not roundtrip_ok:
        logger.warning(f"{tokenizer_id}: encode/decode is not the identity on the eval sample; "
                       f"bpb is per true byte, so a lossy scheme is undercharged here")

    eval_bytes = sum(len(d.encode("utf-8")) for d in eval_docs)
    eval_tokens = int(sum(len(a) for a in eval_ids))
    train_tokens = int(sum(len(a) for a in train_ids))

    # An eval token whose id never occurred in training can only ever be priced by the
    # uniform floor, so this rate is the share of the text the model is blind on. It does
    # not depend on the order.
    seen_ids = np.zeros(geom.radix, dtype=bool)
    for arr in train_ids:
        seen_ids[arr] = True
    oov = sum(int((~seen_ids[arr]).sum()) for arr in eval_ids) / max(eval_tokens, 1)

    results = []
    for order in orders:
        train_stream, train_mask, train_pos = build_stream(train_ids, order, geom.bos_id, geom.eos_id)
        lm = KneserNeyLM.fit(train_stream, train_mask, train_pos, order=order, radix=geom.radix,
                             alphabet_size=geom.alphabet_size)
        eval_stream, eval_mask, eval_pos = build_stream(eval_ids, order, geom.bos_id, geom.eos_id)
        log2p = lm.log2_probs(eval_stream, eval_mask, eval_pos)
        bits = float(-log2p.sum())
        # Split the scored positions back out per document. Each contributes its tokens
        # plus one EOS, in stream order, so the boundaries are a cumulative sum.
        per_doc = np.add.reduceat(-log2p, np.concatenate([[0], np.cumsum(
            [len(a) + 1 for a in eval_ids[:-1]])])) if eval_ids else np.zeros(0)
        # Scored positions include one EOS per document; those cost bits and add no bytes.
        # The document count is identical across tokenizers, so it does not shift rankings.
        results.append(NgramResult(
            tokenizer_id=tokenizer_id,
            order=order,
            vocab_size=len(model.tokens),
            train_docs=len(train_docs),
            eval_docs=len(eval_docs),
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            eval_bytes=eval_bytes,
            bits=bits,
            bpb=bits / eval_bytes,
            bits_per_token=bits / max(eval_tokens + len(eval_docs), 1),
            tokens_per_byte=eval_tokens / eval_bytes,
            oov_token_rate=oov,
            roundtrip_ok=roundtrip_ok,
            doc_bits=[float(b) for b in per_doc],
            doc_bytes=[len(d.encode("utf-8")) for d in eval_docs],
            ngram_types={k + 1: len(t) for k, t in enumerate(lm.tables)},
        ))
        logger.info(f"{tokenizer_id} n={order}: bpb={results[-1].bpb:.4f} "
                    f"tokens/byte={results[-1].tokens_per_byte:.4f} types={results[-1].ngram_types}")
    return results
