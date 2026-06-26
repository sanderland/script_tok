import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.mingram import MinGramModel, MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.unigram import UnigramModel
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig


def _tiny_text() -> str:
    return "\n".join(
        [
            "hello world 0011",
            "hello tokenizer world 223344",
            "tokenizer world hello 556677",
        ]
    )


@pytest.fixture
def tiny_scriptenc_corpus(tmp_path):
    pretokenizer = get_pretokenizer("scriptenc_cb")
    corpus = PretokenizedCorpus.from_texts(
        "tiny_reindex",
        texts=[_tiny_text()],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
    )
    return pretokenizer, corpus


@pytest.mark.parametrize("model_kind", ["unigram", "mingram"])
def test_reindexed_load_produces_dense_ids(tmp_path, tiny_scriptenc_corpus, model_kind):
    pretokenizer, corpus = tiny_scriptenc_corpus
    if model_kind == "unigram":
        model = UnigramTrainer(
            pretokenizer,
            corpus,
            UnigramTrainerConfig(additional_vocab_size=8, num_workers=1, verbose=False),
        ).train()
        load_fn = UnigramModel.load
    else:
        model = MinGramTrainer(
            pretokenizer,
            corpus,
            MinGramTrainerConfig(
                additional_vocab_size=8,
                overshoot_factor=1.25,
                num_em_iterations=1,
                num_workers=1,
                verbose=False,
            ),
        ).train()
        load_fn = MinGramModel.load

    path = model.save(str(tmp_path / f"{model_kind}.json.gz"))
    loaded = load_fn(path)
    reindexed = load_fn(path, reindex=True)
    text = _tiny_text()

    assert [tuple(token.atomic_tokens) for token in loaded.tokens.values()] == [
        tuple(token.atomic_tokens) for token in reindexed.tokens.values()
    ]
    assert [token.log_prob for token in loaded.tokens.values()] == [token.log_prob for token in reindexed.tokens.values()]
    assert set(reindexed.tokens) == set(range(len(reindexed.tokens)))
    assert max(reindexed.encode(text)) < len(reindexed.tokens)
    assert reindexed.decode(reindexed.encode(text)) == loaded.decode(loaded.encode(text))
