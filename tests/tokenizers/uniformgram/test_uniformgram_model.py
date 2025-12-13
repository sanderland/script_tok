import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.uniformgram import UniformGramModel
from script_bpe.tokenizers.uniformgram.trainer import UniformGramTrainer, UniformGramTrainerConfig


def _tiny_text():
    return "hello 0011 world"


@pytest.mark.parametrize("pretokenizer_name", ["bytes_gpt4_cb", "scriptenc_cb"])
def test_uniformgram_stats_and_report(tmp_path, pretokenizer_name):
    """Test that UniformGram can generate stats and reports."""
    pre = get_pretokenizer(pretokenizer_name)
    corpus = PretokenizedCorpus.from_texts("uniformgram_misc", [_tiny_text()], pre, base_path=str(tmp_path))
    model = UniformGramTrainer(pre, corpus, UniformGramTrainerConfig(additional_vocab_size=8, verbose=False)).train()

    stats = model.stats()
    for k in [
        "num_tokens",
        "num_atomic_tokens",
        "num_multi_tokens",
        "num_undecodable",
        "avg_token_length_bt",
        "avg_char_length",
        "longest_tokens_by_atomic",
    ]:
        assert k in stats

    rep = model.report()
    assert isinstance(rep, str) and len(rep) > 0 and rep.startswith("# UniformGram Tokenizer Report")

    # save/load roundtrip
    path = str(tmp_path / "uniformgram.json")
    model.save(path)
    loaded = UniformGramModel.load(path)
    assert loaded.encode(_tiny_text()) == model.encode(_tiny_text())


@pytest.mark.parametrize("pretokenizer_name", ["bytes_gpt4_cb", "scriptenc_cb"])
def test_uniformgram_uniform_probabilities(tmp_path, pretokenizer_name):
    """Test that all tokens have uniform probability (log_prob = 0.0)."""
    pre = get_pretokenizer(pretokenizer_name)
    corpus = PretokenizedCorpus.from_texts("uniformgram_uniform", [_tiny_text()], pre, base_path=str(tmp_path))
    model = UniformGramTrainer(pre, corpus, UniformGramTrainerConfig(additional_vocab_size=8, verbose=False)).train()

    # All tokens should have log_prob = 0.0
    for token in model.tokens.values():
        assert token.log_prob == 0.0, f"Token {token.id} has log_prob {token.log_prob}, expected 0.0"


@pytest.mark.parametrize("pretokenizer_name", ["bytes_gpt4_cb", "scriptenc_cb"])
def test_uniformgram_greedy_longest_tokens(tmp_path, pretokenizer_name):
    """
    Test that UniformGram greedily selects the longest available tokens.

    With uniform probabilities, Viterbi should select the path with the fewest tokens.
    """
    pre = get_pretokenizer(pretokenizer_name)
    # Use repetitive text to encourage longer tokens
    text = "hello hello hello world world world"
    corpus = PretokenizedCorpus.from_texts("uniformgram_greedy", [text], pre, base_path=str(tmp_path))
    model = UniformGramTrainer(pre, corpus, UniformGramTrainerConfig(additional_vocab_size=20, verbose=False)).train()

    # Encode the text
    tokens = model.encode(text, return_tokens=True)

    # With uniform probabilities and sufficient vocab, we should get relatively few tokens
    # (compared to using only atomic tokens)
    atomic_only_encoding_length = sum(len(chunk) for chunk in pre.pretokenize(text))

    # The number of tokens should be less than or equal to atomic-only encoding
    assert len(tokens) <= atomic_only_encoding_length

    # All tokens should have log_prob = 0.0
    for token in tokens:
        assert token.log_prob == 0.0
