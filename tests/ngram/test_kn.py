"""Correctness tests for the n-gram LM.

The metric is only meaningful if the model is a genuine probability distribution -- if it
leaked mass, "bits per byte" would not be a code length and a tokenizer could win by
leaking more. So the properties tested here are the ones the metric rests on: the
distribution normalizes, the counting matches brute force, and the interpolation weights
satisfy the identity they are defined by.
"""

import numpy as np
import pytest

from script_bpe.ngram.counts import build_stream, gram_ids
from script_bpe.ngram.kn import KneserNeyLM


def _docs(seed=0, n_docs=40, vocab=7, max_len=25):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, vocab, size=int(rng.integers(3, max_len))).astype(np.int32) for _ in range(n_docs)]


def _fit(docs, order, vocab=7):
    geom = dict(bos_id=vocab + 1, eos_id=vocab, radix=vocab + 2)
    stream, mask, pos = build_stream(docs, order, geom["bos_id"], geom["eos_id"])
    lm = KneserNeyLM.fit(stream, mask, pos, order=order, radix=geom["radix"], alphabet_size=vocab + 1)
    return lm, geom


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_counts_match_brute_force(order):
    """Dense-prefix-id counting must agree with a dict of tuples."""
    docs = _docs()
    vocab = 7
    stream, mask, pos = build_stream(docs, order, vocab + 1, vocab)
    ids, tables = gram_ids(stream, mask, pos, order, vocab + 2)

    for k in range(1, order + 1):
        # Types are every in-document k-window; counts accrue only at predicted positions.
        brute: dict[tuple, int] = {}
        for i in np.flatnonzero(pos >= k - 1):
            gram = tuple(stream[i - k + 1:i + 1].tolist())
            brute[gram] = brute.get(gram, 0) + int(mask[i])
        assert len(tables[k - 1]) == len(brute), f"order {k}: {len(tables[k - 1])} types vs {len(brute)}"
        for i in np.flatnonzero(pos >= k - 1):
            gid = ids[k - 1][i]
            assert gid >= 0
            assert tables[k - 1].counts[gid] == brute[tuple(stream[i - k + 1:i + 1].tolist())]


@pytest.mark.parametrize("order", [1, 2, 3])
def test_distribution_normalizes(order):
    """For every context in the training data, the next-symbol probabilities sum to 1.

    Enumerated over the whole emittable alphabet by scoring one synthetic position per
    symbol, which is exactly how a real eval position is scored.
    """
    vocab = 7
    docs = _docs(vocab=vocab)
    lm, geom = _fit(docs, order, vocab)

    rng = np.random.default_rng(3)
    for _ in range(12):
        history = rng.integers(0, vocab, size=order - 1).astype(np.int32) if order > 1 else np.zeros(0, np.int32)
        total = 0.0
        for symbol in list(range(vocab)) + [geom["eos_id"]]:
            stream = np.concatenate([
                np.full(order - 1, geom["bos_id"], dtype=np.int32), history,
                np.array([symbol], dtype=np.int32)]).astype(np.int32)
            mask = np.zeros(len(stream), dtype=bool)
            mask[-1] = True
            doc_pos = np.arange(len(stream), dtype=np.int32)
            total += float(2.0 ** lm.log2_probs(stream, mask, doc_pos)[0])
        assert total == pytest.approx(1.0, abs=1e-9), f"history {history} sums to {total}"


@pytest.mark.parametrize("order", [2, 3])
def test_gamma_is_the_discounted_mass(order):
    """gamma(h) must equal exactly what the discounts removed from h's continuations."""
    docs = _docs(vocab=7)
    lm, _ = _fit(docs, order, vocab=7)
    for k in range(2, order + 1):
        level = lm.levels[k]
        parent = lm.tables[k - 1].parent
        kept = np.maximum(level.count - level.discounts[np.clip(level.count, 0, 3)], 0.0)
        removed = np.bincount(parent, weights=level.count - kept, minlength=len(level.ctx_total))
        np.testing.assert_allclose(removed, level.gamma, rtol=1e-9, atol=1e-9)


def test_unseen_symbols_are_never_free_and_never_impossible():
    """Every symbol keeps positive probability, and an unseen one costs more than a seen one."""
    vocab = 7
    docs = [np.array([0, 1, 2, 0, 1, 2] * 6, dtype=np.int32) for _ in range(10)]
    lm, geom = _fit(docs, 3, vocab)
    stream = np.array([geom["bos_id"], 0, 1], dtype=np.int32)   # 5 never occurs in training
    unseen = np.array([geom["bos_id"], 0, 5], dtype=np.int32)
    mask = np.array([False, False, True])
    doc_pos = np.arange(3, dtype=np.int32)
    seen_bits = -lm.log2_probs(stream, mask, doc_pos)[0]
    unseen_bits = -lm.log2_probs(unseen, mask, doc_pos)[0]
    assert np.isfinite(unseen_bits) and unseen_bits > seen_bits


def test_higher_order_never_hurts_on_repetitive_text():
    """On text with real sequential structure, more context must buy bits."""
    vocab = 7
    docs = [np.array([0, 1, 2, 3, 4] * 20, dtype=np.int32) for _ in range(20)]
    stream_bits = []
    for order in (1, 2, 3):
        lm, geom = _fit(docs, order, vocab)
        stream, mask, pos = build_stream(docs[:3], order, geom["bos_id"], geom["eos_id"])
        stream_bits.append(float(-lm.log2_probs(stream, mask, pos).sum()))
    assert stream_bits[0] > stream_bits[1] > stream_bits[2]


def test_key_packing_survives_large_id_spaces():
    """Dense ids times the radix must be computed in int64.

    Regression test for a silent wrap: ids are int32, and under NEP 50 an int32 array
    multiplied by a Python int stays int32, so `id * radix` overflows once a corpus has a
    few hundred thousand n-gram types. It does not raise -- it wraps negative and merges
    unrelated n-grams into one count, which shows up only as a quietly wrong bpb. The
    vocabulary here is large enough that the order-2 ids alone push the product past 2^31.
    """
    vocab = 32768
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, vocab, size=400).astype(np.int32) for _ in range(500)]
    stream, mask, pos = build_stream(docs, 3, vocab + 1, vocab)
    _, tables = gram_ids(stream, mask, pos, 3, vocab + 2)

    assert len(tables[1]) * (vocab + 2) > np.iinfo(np.int32).max, "test corpus too small to trip the overflow"
    for k in (2, 3):
        brute = {tuple(stream[i - k + 1:i + 1].tolist()) for i in np.flatnonzero(pos >= k - 1)}
        assert len(tables[k - 1]) == len(brute), f"order {k}: {len(tables[k - 1])} types vs {len(brute)} brute force"
