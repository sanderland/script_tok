"""End-to-end: a real trained tokenizer through to a bits-per-byte number."""

import numpy as np
import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.ngram import evaluate_ngram_bpb
from script_bpe.ngram.text import take_split
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _docs(n_docs: int, seed: int) -> list[str]:
    """Words drawn i.i.d. -- there is no structure above the word to find."""
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, size=int(rng.integers(20, 60)))) for _ in range(n_docs)]


def _markov_docs(n_docs: int, seed: int) -> list[str]:
    """Words from a *second*-order chain over words.

    Second order, not first: this tokenizer packs each of these words into a single token,
    so a first-order chain would be fully captured by the bigram model and the trigram
    would have nothing left to find. Two words of history means each successive order has
    real structure to pick up.
    """
    v = len(WORDS)
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n_docs):
        a, b = int(rng.integers(v)), int(rng.integers(v))
        out = [WORDS[a], WORDS[b]]
        for _ in range(int(rng.integers(60, 100))):
            # 85% of the time the successor is a fixed function of the last two words.
            nxt = (a + 2 * b + 1) % v if rng.random() < 0.85 else int(rng.integers(v))
            out.append(WORDS[nxt])
            a, b = b, nxt
        docs.append(" ".join(out))
    return docs


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ngram_model")
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    corpus = PretokenizedCorpus.from_texts(
        "ngram_eval_smoke", texts=_docs(60, seed=1), pretokenizer=pretokenizer, base_path=str(tmp_path)
    )
    config = MinGramTrainerConfig(additional_vocab_size=32, overshoot_factor=1.1, verbose=False)
    return MinGramTrainer(pretokenizer, corpus, config).train()


def test_bpb_is_a_finite_number_per_true_byte(tiny_model):
    eval_docs = _docs(20, seed=3)
    results = evaluate_ngram_bpb(
        tiny_model, eval_docs=eval_docs, train_docs=_docs(200, seed=2), orders=[1, 2, 3], tokenizer_id="tiny"
    )
    assert [r.order for r in results] == [1, 2, 3]
    for r in results:
        assert np.isfinite(r.bpb) and r.bpb > 0
        # The denominator is the real text, not the sum of per-token decoded lengths.
        assert r.eval_bytes == sum(len(d.encode("utf-8")) for d in eval_docs)
        assert r.roundtrip_ok


def test_context_only_pays_when_there_is_structure(tiny_model):
    """Order must buy bits on a Markov source and must not on an i.i.d. one.

    The second half is the more interesting guarantee: a metric that always improved with
    n would just be rewarding memorization of the eval set. Here the i.i.d. source has no
    word-to-word structure, so the trigram model pays for its own sparsity -- which is the
    honest answer, and is why the order to report has to be chosen empirically rather than
    by taking the largest n that runs.
    """
    markov = evaluate_ngram_bpb(tiny_model, eval_docs=_markov_docs(20, seed=11),
                                train_docs=_markov_docs(300, seed=12), orders=[1, 2, 3],
                                tokenizer_id="markov")
    assert markov[0].bpb > markov[1].bpb > markov[2].bpb

    iid = evaluate_ngram_bpb(tiny_model, eval_docs=_docs(20, seed=3), train_docs=_docs(300, seed=2),
                             orders=[2, 3], tokenizer_id="iid")
    assert iid[1].bpb >= iid[0].bpb


def test_bpb_beats_a_pessimal_shuffle(tiny_model):
    """A model trained on unrelated text must score the eval text worse."""
    eval_docs = _docs(20, seed=3)
    real = evaluate_ngram_bpb(tiny_model, eval_docs=eval_docs, train_docs=_docs(200, seed=2),
                              orders=[3], tokenizer_id="real")[0]
    rng = np.random.default_rng(7)
    noise = ["".join(rng.choice(list("qwrtyp"), size=300)) for _ in range(200)]
    mismatched = evaluate_ngram_bpb(tiny_model, eval_docs=eval_docs, train_docs=noise,
                                    orders=[3], tokenizer_id="noise")[0]
    assert mismatched.bpb > real.bpb


def test_take_split_is_disjoint_and_ordered(tmp_path):
    path = tmp_path / "docs.jsonl"
    import json

    docs = [f"document number {i} " * 5 for i in range(200)]
    path.write_text("\n".join(json.dumps(d) for d in docs), encoding="utf-8")
    eval_docs, train_docs = take_split(f"file:{path}", eval_chars=500, train_chars=2000)
    assert not set(eval_docs) & set(train_docs)
    assert docs.index(train_docs[0]) > docs.index(eval_docs[-1])
    assert sum(map(len, eval_docs)) >= 500 and sum(map(len, train_docs)) >= 2000


def test_short_source_is_an_error_not_a_silent_truncation(tmp_path):
    import json

    path = tmp_path / "small.jsonl"
    path.write_text(json.dumps("only a little text"), encoding="utf-8")
    with pytest.raises(ValueError, match="short of the requested"):
        take_split(f"file:{path}", eval_chars=1000, train_chars=1000)
