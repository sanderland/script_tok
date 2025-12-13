import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.utils import token_array
from script_bpe.pretokenize import PRETOKENIZER_REGISTRY, get_pretokenizer
from script_bpe.tokenizers.uniformgram import UniformGramModel
from script_bpe.tokenizers.uniformgram.trainer import UniformGramTrainer, UniformGramTrainerConfig


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
def test_uniformgram_train_end_to_end(tmp_path, pretokenizer_name, text_fixture, x_tokens: int = 10):
    """Test end-to-end training of UniformGram tokenizer."""
    text = text_fixture()
    pretokenizer = get_pretokenizer(pretokenizer_name)

    corpus = PretokenizedCorpus.from_texts(
        f"test_uniformgram_train_{text_fixture.__name__}",
        texts=[text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
    )

    trainer = UniformGramTrainer(
        pretokenizer, corpus, UniformGramTrainerConfig(additional_vocab_size=x_tokens, verbose=True)
    )
    model = trainer.train()

    # Basic shape
    assert isinstance(model, UniformGramModel)
    # Should have exactly the requested number of tokens (or fewer if corpus doesn't have enough patterns)
    assert len(pretokenizer.atomic_tokens) <= len(model.tokens) <= len(pretokenizer.atomic_tokens) + x_tokens

    # All tokens should have uniform probability (log_prob = 0.0)
    for token in model.tokens.values():
        assert token.log_prob == 0.0, f"Token {token.id} has log_prob {token.log_prob}, expected 0.0"

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
    path = model.save(str(tmp_path / "uniformgram_model.json"))
    loaded = UniformGramModel.load(path)
    assert isinstance(loaded, UniformGramModel)
    assert len(loaded.tokens) == len(model.tokens)
    assert loaded.encode(text) == ids


def test_uniformgram_iterative_pruning(tmp_path):
    """Test that iterative Viterbi counting and pruning works correctly."""
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    text = "hello world hello world hello world"

    corpus = PretokenizedCorpus.from_texts(
        "test_uniformgram_pruning", texts=[text], pretokenizer=pretokenizer, base_path=str(tmp_path)
    )

    # Train with a small vocab to ensure pruning happens
    config = UniformGramTrainerConfig(
        additional_vocab_size=10,
        initial_vocab_factor=5,  # Start with 50 extra tokens
        verbose=True,
    )
    trainer = UniformGramTrainer(pretokenizer, corpus, config)
    model = trainer.train()

    # Should have pruned down to the target size
    assert len(model.tokens) <= len(pretokenizer.atomic_tokens) + 10

    # All tokens should have log_prob = 0.0
    for token in model.tokens.values():
        assert token.log_prob == 0.0


def test_uniformgram_viterbi_counting(tmp_path):
    """Test that Viterbi counting correctly counts token usage."""
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    # Repetitive text to ensure some tokens are used more than others
    text = "aaa bbb aaa bbb aaa"

    corpus = PretokenizedCorpus.from_texts(
        "test_uniformgram_counting", texts=[text], pretokenizer=pretokenizer, base_path=str(tmp_path)
    )

    config = UniformGramTrainerConfig(additional_vocab_size=20, verbose=False)
    trainer = UniformGramTrainer(pretokenizer, corpus, config)
    model = trainer.train()

    # Count tokens in Viterbi segmentation
    token_count, total_tokens = trainer.viterbi_count_step(model)

    # Total tokens counted should match the sum of individual token counts
    assert sum(token_count.values()) == total_tokens

    # Some tokens should have non-zero counts
    non_zero_counts = [count for count in token_count.values() if count > 0]
    assert len(non_zero_counts) > 0

    # All counts should be non-negative
    for count in token_count.values():
        assert count >= 0
