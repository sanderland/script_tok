from collections import Counter

import pytest

from script_bpe.tokenizers.unigram.init_algorithms import (
    compute_substring_frequencies_corpus,
    compute_substring_frequencies_simple,
    FlatCorpusT,
)
from script_bpe.pretokenize.pretokenizer import UTF8Pretokenizer, UTF8PretokenizerConfig

config = UTF8PretokenizerConfig(regex_pattern=None, digit_handling=None)
pretokenizer = UTF8Pretokenizer(config)


class NoMultiWordPretokenizerConfig(UTF8PretokenizerConfig):
    pass


class NoMultiWordPretokenizer(UTF8Pretokenizer, config_type=NoMultiWordPretokenizerConfig):
    def token_allowed(self, token: tuple[int, ...]) -> bool:
        return ord(" ") + self.config.starting_token_id not in token[1:]


pretokenizer_no_multi_word = NoMultiWordPretokenizer(config)


def _encode_key(s: str) -> tuple[int, ...]:
    char_encs = pretokenizer.encode_text(s)
    return tuple(tid for ce in char_encs for tid in ce.atomic_token_ids)


def _encode_expected(expected_str_map: dict[str, int]) -> dict[tuple[int, ...], int]:
    return {_encode_key(k): v for k, v in expected_str_map.items()}


def _make_corpus(words: list[str], freqs: list[int] | None = None) -> list[tuple[tuple[int, ...], int]]:
    if freqs is None:
        freqs = [1] * len(words)
    return [(_encode_key(w), f) for w, f in zip(words, freqs)]


# Globals: corpora and expected outputs (as readable string -> int maps)
CORPUS1_WORDS: list[str] = [
    "grammy award",
    "grammy winning",
    "grammar",
]
CORPUS1_EXPECTED_LONG = {
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
    "d": 1,
    "y": 2,
    "i": 2,
}
CORPUS1_EXPECTED_FALLBACK = CORPUS1_EXPECTED_LONG
CORPUS1_EXPECTED_INTERMEDIATE = {
    **CORPUS1_EXPECTED_FALLBACK,
    "grammy": 2,
    "gr": 3,
    "gra": 3,
    "gram": 3,
    "rammy": 2,
    "ra": 3,
    "ram": 3,
    "ammy": 2,
    "am": 3,
    "mmy": 2,
    "my": 2,
    "y": 2,
    "i": 2,
    "d": 1,
}

CORPUS1_EXPECTED_FALLBACK_NO_SPACES = {
    **{k: v for k, v in CORPUS1_EXPECTED_FALLBACK.items() if " " not in k or k == " "},
    "grammy": 2,
    "rammy": 2,
    "ammy": 2,
    "mmy": 2,
    "my": 2,
    "y": 2,
    "w": 2,
    "in": 2,
    "n": 3,
    "i": 2,
}

WEIGHTED_WORDS: list[str] = ["banana", "ananas", "asinine", "nine"]
WEIGHTED_FREQS: list[int] = [1, 2, 3, 9]
WEIGHTED_EXPECTED_LONG = {
    "ananas": 2,
    "asinine": 3,
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
    "b": 1,
    "i": 15,
}
WEIGHTED_EXPECTED_FALLBACK = WEIGHTED_EXPECTED_LONG
WEIGHTED_EXPECTED_INTERMEDIATE = {
    **WEIGHTED_EXPECTED_FALLBACK,
    "anan": 3,
    "an": 6,
    "nan": 3,
    "ni": 12,
    "nin": 12,
    "i": 15,
    "b": 1,
}


def _assert_is_equal_to_expected(corpus: FlatCorpusT, result: dict, expected: dict, pretokenizer: UTF8Pretokenizer):
    errors = []
    for k in expected.keys() | result.keys():
        decoded = pretokenizer.decode(k)
        if k not in expected:
            errors.append(f"Extra key {k} = {decoded!r} found in result")
        elif k not in result:
            errors.append(f"Key {k} = {decoded!r} not found in result")
        elif expected[k] != result[k]:
            errors.append(f"Key {k} = {decoded!r} has frequency {result[k]} but expected {expected[k]}")
    assert not errors, "\n" + "\n".join(errors)

    single_count = Counter()
    for ts, c in corpus:
        for t in ts:
            single_count[t] += c
    for k, c in single_count.items():
        if not pretokenizer.token_allowed((k,)):
            continue
        decoded = pretokenizer.decode([k], errors="backslashreplace")
        assert result.get((k,), 0) == c, f"Key {(k,)} = {decoded} has frequency {result.get((k,), 0)} but expected {c}"


@pytest.mark.parametrize(
    "words,freqs,max_len,strategy,expected",
    [
        pytest.param(CORPUS1_WORDS, None, 10, "long", CORPUS1_EXPECTED_LONG, id="corpus1-long"),
        pytest.param(CORPUS1_WORDS, None, 10, "fallback", CORPUS1_EXPECTED_FALLBACK, id="corpus1-fallback"),
        pytest.param(CORPUS1_WORDS, None, 10, "intermediate", CORPUS1_EXPECTED_INTERMEDIATE, id="corpus1-intermediate"),
        pytest.param(WEIGHTED_WORDS, WEIGHTED_FREQS, 10, "long", WEIGHTED_EXPECTED_LONG, id="weighted-long"),
        pytest.param(
            WEIGHTED_WORDS, WEIGHTED_FREQS, 10, "fallback", WEIGHTED_EXPECTED_FALLBACK, id="weighted-fallback"
        ),
        pytest.param(
            WEIGHTED_WORDS,
            WEIGHTED_FREQS,
            10,
            "intermediate",
            WEIGHTED_EXPECTED_INTERMEDIATE,
            id="weighted-intermediate",
        ),
    ],
)
def test_compute_substring_frequencies_exact_outputs(words, freqs, max_len, strategy, expected: dict[str, int]):
    corpus = _make_corpus(words, freqs)
    result = compute_substring_frequencies_corpus(pretokenizer, corpus, max_token_length=max_len, strategy=strategy)

    encoded_expected = _encode_expected(expected)
    _assert_is_equal_to_expected(corpus, result, encoded_expected, pretokenizer)


def test_fallback_strategy_no_spaces():
    corpus = _make_corpus(CORPUS1_WORDS)
    result = compute_substring_frequencies_corpus(
        pretokenizer_no_multi_word, corpus, max_token_length=10, strategy="fallback"
    )
    encoded_expected = _encode_expected(CORPUS1_EXPECTED_FALLBACK_NO_SPACES)
    _assert_is_equal_to_expected(corpus, result, encoded_expected, pretokenizer_no_multi_word)


def test_simple_vs_corpus_consistency():
    """Verify corpus algorithm results are consistent with simple enumeration.

    The corpus (LCP-based) algorithm only returns repeated patterns (freq > 1).
    The simple algorithm returns ALL substrings. So we verify that:
    - All multi-token patterns from corpus appear in simple with same frequency
    - Single tokens have correct corpus-wide counts in both
    """
    corpus = _make_corpus(CORPUS1_WORDS)
    result_simple = compute_substring_frequencies_simple(pretokenizer, corpus, max_token_length=10)
    result_corpus = compute_substring_frequencies_corpus(
        pretokenizer, corpus, max_token_length=10, strategy="intermediate"
    )

    # All patterns in corpus result should appear in simple with same or higher frequency
    for pattern, freq in result_corpus.items():
        assert pattern in result_simple, f"Pattern {pattern} in corpus result but not in simple"
        assert result_simple[pattern] == freq, (
            f"Pattern {pattern} has freq {freq} in corpus but {result_simple[pattern]} in simple"
        )


