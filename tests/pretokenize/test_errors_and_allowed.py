import pytest

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.utils import token_array


@pytest.mark.parametrize("name", ["bytes_gpt4_cb", "scriptenc_cb"])
def test_token_allowed_and_errors(name):
    pt = get_pretokenizer(name)

    # Build an invalid sequence by cutting a token boundary
    seq = pt.pretokenize("ü")[0]
    if len(seq) >= 2:
        bad = [seq[1]]  # continuation without start
        assert not pt.token_allowed(bad)

        # strict should raise, backslashreplace should not
        with pytest.raises(ValueError):
            pt.decode(bad, errors="strict")
        assert isinstance(pt.decode(bad, errors="backslashreplace"), str)

    # valid full-token should be allowed
    assert pt.token_allowed(seq)


def test_regex_split_edges():
    # GPT-4o pattern groups numbers up to 3 digits
    pt = get_pretokenizer("bytes_gpt4o_cb")
    text = "abc 1234 xyz"
    chunks = pt.pretokenize(text)
    # should not collapse everything into one chunk
    assert len(chunks) >= 3
    assert pt.decode(sum(chunks, token_array([]))) == pt.normalize(text)
