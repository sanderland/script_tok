from __future__ import annotations

from collections import Counter

import pytest

from script_bpe.tokenizers.unigram.init_corpus import (
    compute_substring_frequencies_corpus,
)


# Globals: corpora and expected outputs (as readable string -> int maps)
CORPUS1_WORDS: list[str] = [
    "grammy award",
    "grammy winning",
    "grammar",
]
CORPUS1_EXPECTED_STR: dict[str, int] = {
    "grammy ": 2,
    "gramm": 3,
    "g": 4,
    "rammy ": 2,
    "ramm": 3,
    "r": 5,
    "ammy ": 2,
    "amm": 3,
    "ar": 2,
    "a": 6,
    "mmy ": 2,
    "mm": 3,
    "my ": 2,
    "m": 6,
    "y ": 2,
    " ": 2,
    "w": 2,
    "in": 2,
    "n": 3,
}
CORPUS1_EXPECTED_INTERMEDIATE_STR: dict[str, int] = {
    "grammy": 2,
    "grammy ": 2,
    "gr": 3,
    "gra": 3,
    "gram": 3,
    "gramm": 3,
    "g": 4,
    "rammy": 2,
    "rammy ": 2,
    "ra": 3,
    "ram": 3,
    "ramm": 3,
    "r": 5,
    "ammy": 2,
    "ammy ": 2,
    "am": 3,
    "amm": 3,
    "ar": 2,
    "a": 6,
    "mmy": 2,
    "mmy ": 2,
    "mm": 3,
    "my": 2,
    "my ": 2,
    "m": 6,
    "y ": 2,
    "y": 2,
    " ": 2,
    "w": 2,
    "in": 2,
    "i": 2,
    "n": 3,
}

WEIGHTED_WORDS: list[str] = ["banana", "ananas", "asinine", "nine"]
WEIGHTED_FREQS: list[int] = [1, 2, 3, 9]
WEIGHTED_EXPECTED_STR: dict[str, int] = {
    "anana": 3,
    "ana": 6,
    "as": 5,
    "a": 12,
    "nana": 3,
    "na": 6,
    "ne": 12,
    "nine": 12,
    "n": 30,
    "s": 5,
    "ine": 12,
    "in": 15,
    "e": 12,
}
WEIGHTED_EXPECTED_INTERMEDIATE_STR: dict[str, int] = {
    "anan": 3,
    "anana": 3,
    "an": 6,
    "ana": 6,
    "as": 5,
    "a": 12,
    "nan": 3,
    "nana": 3,
    "na": 6,
    "ne": 12,
    "ni": 12,
    "nin": 12,
    "nine": 12,
    "n": 30,
    "s": 5,
    "in": 15,
    "ine": 12,
    "i": 15,
    "e": 12,
}


def encode_char(char: str) -> int:
    if char == ' ':
        return 0
    return ord(char) - ord('a') + 1


def _encode_key(s: str) -> tuple[int, ...]:
    return tuple(encode_char(c) for c in s)


def _encode_expected(expected_str_map: dict[str, int]) -> dict[tuple[int, ...], int]:
    return {_encode_key(k): v for k, v in expected_str_map.items()}


def _make_corpus(words: list[str], freqs: list[int] | None = None) -> list[tuple[tuple[int, ...], int]]:
    if freqs is None:
        freqs = [1] * len(words)
    return [
        (tuple(encode_char(c) for c in w), f) for w, f in zip(words, freqs)
    ]


@pytest.mark.parametrize(
    "words,freqs,max_len,intermediate,expected_str",
    [
        pytest.param(CORPUS1_WORDS, None, 10, False, CORPUS1_EXPECTED_STR, id="corpus1"),
        pytest.param(CORPUS1_WORDS, None, 10, True, CORPUS1_EXPECTED_INTERMEDIATE_STR, id="corpus1-int"),
        pytest.param(WEIGHTED_WORDS, WEIGHTED_FREQS, 10, False, WEIGHTED_EXPECTED_STR, id="weighted"),
        pytest.param(WEIGHTED_WORDS, WEIGHTED_FREQS, 10, True, WEIGHTED_EXPECTED_INTERMEDIATE_STR, id="weighted-int"),
    ],
)
def test_compute_substring_frequencies_exact_outputs(words, freqs, max_len, intermediate, expected_str):
    corpus = _make_corpus(words, freqs)
    result = compute_substring_frequencies_corpus(
        corpus, max_token_length=max_len, intermediate_patterns=intermediate
    )

    assert result == Counter(_encode_expected(expected_str))


