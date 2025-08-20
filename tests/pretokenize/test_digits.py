import pytest

from script_bpe.pretokenize import UTF8Pretokenizer, UTF8PretokenizerConfig
from script_bpe.pretokenize.pretokenizer import group_digits


def _flatten_pretok(chunks):
    return [tid for chunk in chunks for tid in chunk]


@pytest.mark.parametrize(
    "number,expected",
    [
        ("", []),
        ("1", ["1"]),
        ("12", ["12"]),
        ("123", ["123"]),
        ("1234", ["1", "234"]),
        ("12345", ["12", "345"]),
        ("123456", ["123", "456"]),
        ("1234567", ["1", "234", "567"]),
        ("0078", ["0", "078"]),
    ],
)
def test_group_digits_rtl3(number, expected):
    assert group_digits(number, "RTL3") == expected


@pytest.mark.parametrize("digit_handling", [None, "SPLIT", "RTL3"])
def test_roundtrip_and_chunking(digit_handling):
    cfg = UTF8PretokenizerConfig(regex_pattern=None, digit_handling=digit_handling)
    ptok = UTF8Pretokenizer(cfg)

    text = "ab12c0078d ٩"  # includes ASCII digits and a non-ASCII digit-like char to ensure only 0-9 split
    chunks = ptok.pretokenize(text)

    # Roundtrip
    flat = _flatten_pretok(chunks)
    assert ptok.decode(flat) == ptok.normalize(text)

    if digit_handling is None:
        # No digit splitting: single chunk expected
        assert len(chunks) == 1
    elif digit_handling == "SPLIT":
        # Expect non-digit, 1,2, non-digit, 0,0,7,8, rest = 9 chunks
        assert len(chunks) == 9
        for idx in [1, 2, 4,5,6,7]:
            for tid in chunks[idx]:
                # Should be present in token_to_id values
                assert tid in ptok.token_to_id.values()
    elif digit_handling == "RTL3":
        # 6 chunks, with the two digit chunks grouped RTL3
        assert len(chunks) == 6
        # Validate grouping by decoding just the digit chunks roundtrip
        # and comparing with original digit substrings
        # chunks positions 1 and 3 are digit-only
        digit_subs = ["12", "0","078"]
        for idx, original_digits in zip([1, 3, 4], digit_subs):
            decoded = ptok.tokens_repr(chunks[idx])
            assert decoded == original_digits


def test_registered_digit_tokens_split():
    ptok = UTF8Pretokenizer(UTF8PretokenizerConfig(digit_handling="SPLIT"))
    for d in "0123456789":
        assert d in ptok.token_to_id


def test_registered_digit_tokens_rtl3_samples():
    ptok = UTF8Pretokenizer(UTF8PretokenizerConfig(digit_handling="RTL3"))
    # sample a few from each size to avoid large loops
    for d in ["0", "7", "9"]:
        assert d in ptok.token_to_id
    for dd in ["00", "12", "99"]:
        assert dd in ptok.token_to_id
    for ddd in ["000", "078", "999"]:
        assert ddd in ptok.token_to_id


