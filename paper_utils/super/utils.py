"""Shared utilities for supertoken experiments.

Includes:
- Supertoken filter functions
- Results directory and caching
- Line-regex pretokenizer for supertoken training
"""

import json
from collections.abc import Callable
from pathlib import Path

import regex as re

from script_bpe.pretokenize import PRETOKENIZER_REGISTRY, Pretokenizer
from script_bpe.pretokenize.pretokenizer import ScriptPretokenizerConfig
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV1

# ========== RESULTS CONFIG ==========

PRETOKENIZER_NAME = "scriptenc_cb"
LINE_PRETOKENIZER_NAME = "scriptenc_line"
RESULTS_DIR = Path("results/supertoken_experiments")

# ========== LINE-REGEX PRETOKENIZER ==========
# This pretokenizer uses line-based chunking instead of word-level,
# allowing supertokens to span across word boundaries.

LINE_REGEX = r"[\r\n]*[^\r\n]*"

PRETOKENIZER_REGISTRY[LINE_PRETOKENIZER_NAME] = ScriptPretokenizerConfig(
    regex_pattern=LINE_REGEX,
    script_split=False,
    script_config=ScriptEncodingV1,
)

# ========== SUPERTOKEN FILTERS ==========
#
# Patterns observed in surviving supertokens:
# 1. Space-prefixed phrases: " of the", " in the", " one of the most"
# 2. Punctuation transitions: ", and", ". The", ". I"
# 3. Contractions: "'s", "it's", "I'm", "don't"
# 4. Suffix patterns: "s,", "s and", "s.", "s are"
# 5. Abbreviations: "U.S.", "e.g."
# 6. Hyphenated: "-year-old", "well-known"
# 7. Domains: ".com", "https://"

WORD = r"[\p{L}]+"
WORD_APOS = r"[\p{L}]+(?:['']\p{L}+)?"  # word with optional apostrophe


# ========== Pattern matchers ==========

def _is_space_phrase(s: str) -> bool:
    """Multi-word starting with space: ' word word...'"""
    return re.fullmatch(rf" {WORD}(?: {WORD})+", s) is not None


def _is_punct_transition(s: str) -> bool:
    """Punctuation + word: ', and', '. The', '; but'"""
    return re.fullmatch(rf"[.,;:!?] {WORD}", s) is not None


def _is_contraction(s: str) -> bool:
    """Words with apostrophe: "'s", "it's", "don't", "I'm" """
    # Standalone 's or 't
    if re.fullmatch(r"[''](s|t|m|d|ll|ve|re)", s) is not None:
        return True
    # Word with apostrophe: it's, don't, I'm
    if re.fullmatch(rf" ?{WORD}[''](s|t|m|d|ll|ve|re)", s) is not None:
        return True
    return False


def _is_suffix_punct(s: str) -> bool:
    """Suffix + punctuation/word: 's,', 's.', 's and'"""
    # Letter(s) + punctuation
    if re.fullmatch(r"[\p{L}]{1,3}[.,;:!?]", s) is not None:
        return True
    # Letter(s) + space + word
    if re.fullmatch(r"[\p{L}]{1,3} " + WORD, s) is not None:
        return True
    return False


def _is_abbreviation(s: str) -> bool:
    """Abbreviations: 'U.S.', 'e.g.', 'i.e.'"""
    return re.fullmatch(r" ?(\p{L}\.){2,6}", s) is not None


def _is_hyphenated(s: str) -> bool:
    """Hyphenated compounds: '-year-old', 'well-known'"""
    return re.fullmatch(rf" ?{WORD}(?:-{WORD})+|-{WORD}(?:-{WORD})*", s) is not None


def _is_domain(s: str) -> bool:
    """Domain patterns: '.com', 'https://', 'www.'"""
    if re.fullmatch(r"\.[a-z]{2,4}", s) is not None:
        return True
    if re.fullmatch(r"[a-z]{3,8}://(?:www\.)?", s) is not None:
        return True
    if re.fullmatch(r"www\.", s) is not None:
        return True
    return False


# ========== Filter functions ==========

def supertoken_filter_all(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Accept all supertokens."""
    return True


def supertoken_filter_space_phrase(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Multi-word phrases starting with space."""
    return _is_space_phrase(pretokenizer.decode(seq))


def supertoken_filter_punct_trans(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Punctuation + word transitions."""
    return _is_punct_transition(pretokenizer.decode(seq))


def supertoken_filter_contraction(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Contractions with apostrophe."""
    return _is_contraction(pretokenizer.decode(seq))


def supertoken_filter_suffix(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Suffix + punctuation/word patterns."""
    return _is_suffix_punct(pretokenizer.decode(seq))


def supertoken_filter_abbrev(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Abbreviations like U.S., e.g."""
    return _is_abbreviation(pretokenizer.decode(seq))


def supertoken_filter_hyphen(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Hyphenated compounds."""
    return _is_hyphenated(pretokenizer.decode(seq))


def supertoken_filter_domain(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Domain and URL patterns."""
    return _is_domain(pretokenizer.decode(seq))


def supertoken_filter_semantic(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """All semantic patterns: space_phrase + contraction + abbrev + hyphen."""
    s = pretokenizer.decode(seq)
    return _is_space_phrase(s) or _is_contraction(s) or _is_abbreviation(s) or _is_hyphenated(s)


def supertoken_filter_all_patterns(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """All recognized patterns (semantic + punct_trans + suffix + domain)."""
    s = pretokenizer.decode(seq)
    return (
        _is_space_phrase(s) or _is_punct_transition(s) or _is_contraction(s) or
        _is_suffix_punct(s) or _is_abbreviation(s) or _is_hyphenated(s) or _is_domain(s)
    )


def supertoken_filter_len(pretokenizer: Pretokenizer, seq: tuple[int, ...], max_len: int) -> bool:
    """Accept supertokens with at most max_len atomic tokens."""
    return len(seq) <= max_len


def supertoken_filter_len_8c(pretokenizer: Pretokenizer, seq: tuple[int, ...]) -> bool:
    """Accept supertokens with at most 16 atomic tokens (~8 script-encoded chars)."""
    return supertoken_filter_len(pretokenizer, seq, 16)


# Filter registry: name -> filter function
SUPERTOKEN_FILTERS: dict[str, Callable[[Pretokenizer, tuple[int, ...]], bool]] = {
    # Accept all
    "all": supertoken_filter_all,
    # Individual patterns
    "space_phrase": supertoken_filter_space_phrase,  # " of the", " in the"
    "punct_trans": supertoken_filter_punct_trans,    # ", and", ". The"
    "contraction": supertoken_filter_contraction,    # "'s", "it's", "don't"
    "suffix": supertoken_filter_suffix,              # "s,", "s.", "s and"
    "abbrev": supertoken_filter_abbrev,              # "U.S.", "e.g."
    "hyphen": supertoken_filter_hyphen,              # "-year-old"
    "domain": supertoken_filter_domain,              # ".com", "https://"
    # Combinations
    "semantic": supertoken_filter_semantic,          # space_phrase + contraction + abbrev + hyphen
    "all_patterns": supertoken_filter_all_patterns,  # All recognized patterns
    # Length limits
    "len_8c": supertoken_filter_len_8c,              # Max 8 chars
}


# ========== JSON CACHE ==========


class JsonCache:
    """Simple JSON file cache for evaluation results."""

    def __init__(self, name: str):
        self.path = RESULTS_DIR / f"cache_{name}.json"
        self._cache: dict | None = None

    def _load(self) -> dict:
        if self._cache is None:
            if self.path.exists():
                with open(self.path) as f:
                    self._cache = json.load(f)
            else:
                self._cache = {}
        return self._cache

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, key: str) -> dict | None:
        return self._load().get(key)

    def set(self, key: str, value: dict):
        self._load()[key] = value
        self._save()


# Cache instances
_eval_cache = JsonCache("eval")

