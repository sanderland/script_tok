import itertools
import random

import pytest
import regex

from script_bpe.pretokenize import SP_REGEX, get_pretokenizer
from script_bpe.pretokenize.sentencepiece_split import sentencepiece_regex


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello world", ["hello", " world"]),
        (" leading", [" leading"]),
        ("a  b", ["a", " ", " b"]),
        ("a   b  ", ["a", " ", " ", " b", " ", " "]),
        ("  ", [" ", " "]),
        ("tab\tsep\nline", ["tab\tsep\nline"]),
        ("a \n b", ["a", " \n", " b"]),
        ("你好 世界,是的", ["你好", " 世界,是的"]),
        ("abc123 4567.8", ["abc123", " 4567.8"]),
        ("x▁y z", ["x▁y", " z"]),
    ],
)
def test_sp_split_rule(text, expected):
    pt = get_pretokenizer("bytes_sp_cb")
    chunks = pt.pretokenize(text)
    assert [pt.decode(c) for c in chunks] == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello, world", ["Hello", ",", " world"]),
        ("in 1990s", ["in", " ", "1", "9", "9", "0", "s"]),
        ("a  नमस्ते", ["a", " ", " नमस्ते"]),
        ("a   b", ["a", "  ", " b"]),
        ("end  ", ["end", "  "]),
        ("東京タワーです。", ["東京タワーです", "。"]),
        ("ქართ ру", ["ქართ", " ру"]),
        ("q\u0301x!", ["q\u0301x", "!"]),
        ("x.\n(y)", ["x", ".\n(", "y", ")"]),
    ],
)
def test_sp_gemma_split(text, expected):
    pt = get_pretokenizer("bytes_sp_gemma_cb")
    assert [pt.decode(c) for c in pt.pretokenize(text)] == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("of the cat, sat", ["of the cat", ",", " sat"]),
        ("in 1990s", ["in", " ", "1", "9", "9", "0", "s"]),
        ("a  b  नमस्ते  ", ["a  b", "  नमस्ते", "  "]),
        ("Привет мир hi", ["Привет мир", " hi"]),
    ],
)
def test_sp_gemma_xws_split(text, expected):
    pt = get_pretokenizer("bytes_sp_gemma_xws_cb")
    assert [pt.decode(c) for c in pt.pretokenize(text)] == expected


FLAGS = [
    "split_by_whitespace",
    "split_by_unicode_script",
    "split_by_number",
    "split_digits",
    "allow_whitespace_only_pieces",
]


@pytest.mark.parametrize("values", list(itertools.product([True, False], repeat=len(FLAGS))))
def test_sentencepiece_regex_covers_text(values):
    flags = dict(zip(FLAGS, values))
    rng = random.Random(0)
    pool = [chr(cp) for cp in range(0x20, 0x3100) if not 0xD800 <= cp < 0xE000] + [" "] * 400 + list("0123456789")
    text = "".join(rng.choice(pool) for _ in range(20000))
    chunks = regex.findall(sentencepiece_regex(**flags), text)
    assert "".join(chunks) == text
    assert all(chunk for chunk in chunks)


def test_sentencepiece_regex_without_script_split_matches_sp_cb():
    pattern = sentencepiece_regex(split_by_unicode_script=False, split_by_number=False)
    text = "a   b  c,d 12 x\ty\n  z ▁w  "
    assert regex.findall(pattern, text) == regex.findall(SP_REGEX, text)


@pytest.mark.parametrize(
    "text,allowed",
    [("of the", True), ("of ", False), ("e ", False), (" the", True), ("  ", True), ("a  b", True)],
)
def test_sp_gemma_xws_forbids_trailing_space(text, allowed):
    xws, gemma = get_pretokenizer("bytes_sp_gemma_xws_cb"), get_pretokenizer("bytes_sp_gemma_cb")
    token = list(itertools.chain.from_iterable(xws.pretokenize(text)))
    assert xws.token_allowed(token) is allowed
    assert gemma.token_allowed(token)
