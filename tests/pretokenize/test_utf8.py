import pytest

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.utils import TokenSeq, token_array


@pytest.fixture
def regex_pretokenizer():
    return get_pretokenizer("bytes_gpt4")


def test_hash(regex_pretokenizer):
    assert isinstance(regex_pretokenizer.hash(), str)


def test_regex_tokenize(regex_pretokenizer):
    text = "Hello world!"
    tokenized = regex_pretokenizer.pretokenize(text)
    assert isinstance(tokenized, list)
    assert all(isinstance(group, TokenSeq) for group in tokenized)

    token_ids = sum(tokenized, token_array([]))
    detokenized = regex_pretokenizer.decode(token_ids)
    assert isinstance(detokenized, str)
    assert detokenized == text
