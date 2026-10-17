import math
import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.utils import token_array
from script_bpe.pretokenize import PRETOKENIZER_REGISTRY, get_pretokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig


def taylor_swift_text():
    with open("tests/data/taylorswift.txt", "r") as f:
        return f.read()


def zeros_text():
    return "\n".join([(" " + "0" * i) * j for i in range(10) for j in range(5)])


@pytest.mark.parametrize(
    "pretokenizer_name,text_fixture",
    [
        *[
            pytest.param(
                pretokenizer_name,
                text_fixture,
                id=f"{pretokenizer_name}-{text_fixture.__name__}",
            )
            for pretokenizer_name in PRETOKENIZER_REGISTRY
            if "nosplit" not in pretokenizer_name
            for text_fixture in [taylor_swift_text, zeros_text]
        ]
    ],
)
def test_unigram_train_end_to_end(tmp_path, pretokenizer_name, text_fixture, x_tokens: int = 10):
    text = text_fixture()
    pretokenizer = get_pretokenizer(pretokenizer_name)

    corpus = PretokenizedCorpus.from_texts(
        f"test_unigram_train_{text_fixture.__name__}", texts=[text], pretokenizer=pretokenizer, base_path=str(tmp_path)
    )

    trainer = UnigramTrainer(pretokenizer, corpus, UnigramTrainerConfig(additional_vocab_size=x_tokens, verbose=True))
    model = trainer.train()

    # Basic shape
    assert isinstance(model, UnigramModel)
    # Finalization can keep <= requested additional tokens
    assert len(pretokenizer.atomic_tokens) <= len(model.tokens) <= len(pretokenizer.atomic_tokens) + x_tokens

    # Probabilities are a proper distribution over kept tokens (can be < 1 after pruning)
    prob_sum = sum(math.exp(t.log_prob) for t in model.tokens.values())
    assert 0.8 <= prob_sum <= 1.0001

    # Encode/decode roundtrip preserves pretokenization
    ids = model.encode(text)
    decoded = model.decode(ids)
    orig_atomic = sum(pretokenizer.pretokenize(text), token_array([]))
    dec_atomic = sum(pretokenizer.pretokenize(decoded), token_array([]))
    assert dec_atomic.tolist() == orig_atomic.tolist()

    # Metadata sanity
    assert "tokens/pretoken" in model.metadata
    assert isinstance(model.metadata.get("total_tokens"), int)

    # Save/load roundtrip
    path = model.save(str(tmp_path / "unigram_model.json"))
    loaded = UnigramModel.load(path)
    assert isinstance(loaded, UnigramModel)
    assert len(loaded.tokens) == len(model.tokens)
    assert loaded.encode(text) == ids


@pytest.fixture(scope="module")
def sparse_corpus(tmp_path_factory):
    pretokenizer = get_pretokenizer("scriptenc_cb")
    corpus = PretokenizedCorpus.from_texts(
        "test_unigram_sparse_l1",
        texts=[taylor_swift_text()],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path_factory.mktemp("sparse_corpus")),
    )
    return pretokenizer, corpus


def train_sparse(sparse_corpus, **kwargs):
    pretokenizer, corpus = sparse_corpus
    config = UnigramTrainerConfig(sparse_l1=True, additional_vocab_size=kwargs.pop("n", 100), **kwargs)
    return UnigramTrainer(pretokenizer, corpus, config).train()


def test_sparse_l1_reaches_target_and_roundtrips(sparse_corpus):
    pretokenizer, _ = sparse_corpus
    text = taylor_swift_text()
    model = train_sparse(sparse_corpus, n=100)
    assert len(model.tokens) == len(pretokenizer.atomic_tokens) + 100
    assert model.metadata["config"]["sparse_l1"] is True
    assert math.isclose(sum(math.exp(t.log_prob) for t in model.tokens.values()), 1.0, rel_tol=1e-6)

    decoded = model.decode(model.encode(text))
    orig_atomic = sum(pretokenizer.pretokenize(text), token_array([]))
    dec_atomic = sum(pretokenizer.pretokenize(decoded), token_array([]))
    assert dec_atomic.tolist() == orig_atomic.tolist()


def test_sparse_l1_target_size_scales_and_compresses(sparse_corpus):
    pretokenizer, _ = sparse_corpus
    small = train_sparse(sparse_corpus, n=30)
    large = train_sparse(sparse_corpus, n=200)
    assert len(small.tokens) == len(pretokenizer.atomic_tokens) + 30
    assert len(large.tokens) == len(pretokenizer.atomic_tokens) + 200
    assert large.metadata["total_tokens"] < small.metadata["total_tokens"]


def test_sparse_l1_debias_changes_probabilities_not_vocab(sparse_corpus):
    debiased = train_sparse(sparse_corpus, n=100, post_l1_debias=True)
    shrunk = train_sparse(sparse_corpus, n=100, post_l1_debias=False)
    assert {t.atomic_tokens.tobytes() for t in debiased.tokens.values()} == {
        t.atomic_tokens.tobytes() for t in shrunk.tokens.values()
    }
    by_token = {t.atomic_tokens.tobytes(): t.log_prob for t in shrunk.tokens.values()}
    assert any(not math.isclose(t.log_prob, by_token[t.atomic_tokens.tobytes()]) for t in debiased.tokens.values())
