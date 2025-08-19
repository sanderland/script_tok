from collections import Counter

import math
import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import PRETOKENIZER_REGISTRY, get_pretokenizer
from script_bpe.unigram import UnigramModel, train_unigram


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

    model = train_unigram(
        pretokenizer=pretokenizer,
        corpus=corpus,
        additional_vocab_size=x_tokens,
        verbose=True,
    )

    # Basic shape
    assert isinstance(model, UnigramModel)
    # Finalization can keep <= requested additional tokens
    assert len(pretokenizer.atomic_tokens) <= len(model.tokens) <= len(pretokenizer.atomic_tokens) + x_tokens

    # Probabilities are a proper distribution over kept tokens (can be < 1 after pruning)
    prob_sum = sum(math.exp(t.log_prob) for t in model.tokens)
    assert 0.8 <= prob_sum <= 1.0001

    # Encode/decode roundtrip preserves pretokenization
    ids = model.encode(text)
    decoded = model.decode(ids)
    assert pretokenizer.encode(decoded).tolist() == pretokenizer.encode(text).tolist()

    # Metadata sanity
    assert "tokens/pretoken" in model.metadata
    assert isinstance(model.metadata.get("total_tokens"), int)

    # Save/load roundtrip
    path = model.save(str(tmp_path / "unigram_model.json"))
    loaded = UnigramModel.load(path)
    assert isinstance(loaded, UnigramModel)
    assert len(loaded.tokens) == len(model.tokens)
    assert loaded.encode(text) == ids


