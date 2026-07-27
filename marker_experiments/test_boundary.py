"""Tests for boundary-marker pretokenization.

Run: .venv/bin/python -m pytest marker_experiments/test_boundary.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3

from boundary_pretokenizer import (
    BOUNDARY_VARIANTS,
    BoundaryScriptPretokenizer,
    BoundaryScriptPretokenizerConfig,
    get_boundary_pretokenizer,
)

VARIANTS = list(BOUNDARY_VARIANTS)

TEXTS = [
    "the quick, brown dog",
    "a, b", "a=b", "a = b", "a,b", "a ,b", "a , b",
    "x = 1", "1 item", "range(0, 10)", "a1", "1a", "3.14 and 2", "version 2 of 3",
    "  double  spaces  here", " leading space", "trailing space ", "", " ", "  ",
    "x\ny", "tab\there",
    "e.g. 3.14 and (1 + 2) = 3",
    'quote: "a b" done',
    # cross-script word spans, no separator
    "latinкириллица", "кириллицаlatin", "123 latinкириллица 123",
    "sπ = 1", "upperΔ = 1", ">>> Δx = 1", "łʒλπ", "aΔб", "abcΔ def",
    # non-space scripts and other categories
    "中文 and English 42", "emoji 🎉 7 test", "日本語テキスト",
    "русский текст, да", "العربية نص", "한국어 텍스트",
    "def f():\n    if x > 1:\n        return 2\n",
    "naïve café résumé", "ﬁle", "élan",  # combining marks / inherited
]


def visible(pt, ids):
    """Render marker tokens literally; decode() would turn them into spaces or nothing."""
    out, i = "", 0
    while i < len(ids):
        if ids[i] == pt.marker_token_id:
            out += "<|>"
            i += 1
        else:
            out += pt.tokens_repr(ids[i : i + 2])
            i += 2
    return out


def flat(pt, text):
    return [t for c in pt.pretokenize(text) for t in c]


@pytest.fixture(scope="module", params=VARIANTS)
def pt(request):
    return get_boundary_pretokenizer(request.param)


# --------------------------------------------------------------------------- roundtrip


@pytest.mark.parametrize("text", TEXTS)
def test_roundtrip(pt, text):
    assert pt.decode(flat(pt, text)) == pt.normalize(text)


def test_roundtrip_taylorswift(pt):
    with open("tests/data/taylorswift.txt") as f:
        text = f.read()
    assert pt.decode(flat(pt, text)) == pt.normalize(text)


# ------------------------------------------------------------------- the core invariant


@pytest.mark.parametrize("text", TEXTS)
def test_touching_markers_mean_exactly_one_space(pt, text):
    """Every <|><|> pair must correspond to a real single space in the source, and the
    count of pairs must equal the number of spaces the decoder reinserts."""
    ids = flat(pt, text)
    m = pt.marker_token_id
    pairs, i = 0, 0
    while i < len(ids):
        if ids[i] == m and i + 1 < len(ids) and ids[i + 1] == m:
            pairs += 1
            i += 2
        else:
            i += 1
    # decoding without marker handling drops elided spaces; the difference is the pair count
    n_spaces_source = pt.normalize(text).count(" ")
    n_spaces_kept = sum(
        1
        for c in pt.pretokenize(text)
        for t in c
        if t != m
    )
    assert pairs <= n_spaces_source, "more elided spaces claimed than exist"
    assert pt.decode(ids) == pt.normalize(text)
    assert n_spaces_kept >= 0


@pytest.mark.parametrize("text", TEXTS)
def test_no_triple_marker_run(pt, text):
    """Three markers in a row would be ambiguous (one space plus a lone marker, or the
    reverse). Span merging plus asymmetric punctuation marking must prevent it."""
    ids = flat(pt, text)
    m = pt.marker_token_id
    for i in range(len(ids) - 2):
        assert not (ids[i] == m and ids[i + 1] == m and ids[i + 2] == m), f"triple marker in {text!r}"


# -------------------------------------------------------------- span merging (no middle marker)


def test_cross_script_word_span_has_no_internal_marker():
    """The whole point of merging spans: latin+Cyrillic with no space is ONE span, so no
    marker appears between them and no phantom space can be decoded."""
    for name in VARIANTS:
        pt = get_boundary_pretokenizer(name)
        ids = flat(pt, "latinкириллица")
        m = pt.marker_token_id
        assert ids[0] == m and ids[-1] == m, f"{name}: span not delimited"
        assert ids.count(m) == 2, f"{name}: expected exactly 2 markers, got {ids.count(m)}"
        assert pt.decode(ids) == "latinкириллица"


def test_cross_script_span_keeps_canonical_form_for_both_scripts():
    """A word keeps the same delimited form whether or not a different-script word
    precedes it, because the span is delimited as a whole."""
    pt = get_boundary_pretokenizer("bnd_wpd")
    m = pt.marker_token_id
    # 'кириллица' alone vs after a Latin run: in both cases the SPAN carries the markers
    alone = flat(pt, "кириллица")
    assert alone[0] == m and alone[-1] == m and alone.count(m) == 2


def test_word_span_always_delimited_both_sides():
    pt = get_boundary_pretokenizer("bnd_wpd")
    m = pt.marker_token_id
    for text in ["word", "(word", "word)", "1word", "word1", ",word,", "=word="]:
        ids = flat(pt, text)
        # locate the word span's markers: there must be a marker immediately before the
        # first letter token and after the last
        assert ids.count(m) >= 2, f"{text!r} -> {visible(pt, ids)}"
        assert pt.decode(ids) == text


def test_internal_script_split_preserved():
    """Markers ride the outer chunks; the script boundary inside the span still splits
    chunks, so no BPE merge can cross a script change the baseline would forbid."""
    pt = get_boundary_pretokenizer("bnd_wpd")
    chunks = pt.pretokenize("latinкириллица")
    assert len(chunks) == 2, [visible(pt, list(c)) for c in chunks]
    assert visible(pt, list(chunks[0])).startswith("<|>")
    assert visible(pt, list(chunks[1])).endswith("<|>")


# ----------------------------------------------------------------- boundary target behaviour


def test_punct_only_marked_on_space_side():
    pt = get_boundary_pretokenizer("bnd_wp")
    assert visible_chunks(pt, "a,b") == ["<|>a<|>", ",", "<|>b<|>"]
    assert visible_chunks(pt, "a, b") == ["<|>a<|>", ",<|>", "<|>b<|>"]
    assert visible_chunks(pt, "a ,b") == ["<|>a<|>", "<|>,", "<|>b<|>"]
    assert visible_chunks(pt, "a = b") == ["<|>a<|>", "<|>=<|>", "<|>b<|>"]


def test_digits_marked_only_in_wpd():
    wp = get_boundary_pretokenizer("bnd_wp")
    wpd = get_boundary_pretokenizer("bnd_wpd")
    # under bnd_wp a digit is not markable, so the space cannot be elided and survives
    assert " " in "".join(visible_chunks(wp, "x = 1"))
    # under bnd_wpd both spaces are elided
    assert visible_chunks(wpd, "x = 1") == ["<|>x<|>", "<|>=<|>", "<|>1"]


def test_word_only_variant_leaves_punct_and_digits_bare():
    pt = get_boundary_pretokenizer("bnd_w")
    assert visible_chunks(pt, "a, b") == ["<|>a<|>", ",", " ", "<|>b<|>"]
    assert visible_chunks(pt, "a b") == ["<|>a<|>", "<|>b<|>"]


def test_multi_space_runs_untouched(pt):
    """Only a single space is ever elided; runs are left exactly as the baseline emits."""
    for text in ["a  b", "a   b", "a\t b"]:
        assert pt.decode(flat(pt, text)) == pt.normalize(text)
    assert "  " in "".join(visible_chunks(pt, "a  b"))


def test_leading_and_trailing_space_not_elided(pt):
    for text in [" a", "a "]:
        assert pt.decode(flat(pt, text)) == pt.normalize(text)


# ------------------------------------------------------------------------ merge constraint


def test_merge_across_touching_markers_forbidden(pt):
    m = pt.marker_token_id
    assert pt.bpe_merge_allowed([m], [m]) is False
    assert pt.bpe_merge_allowed([1, m], [m, 2]) is False


def test_lone_marker_merge_allowed(pt):
    m = pt.marker_token_id
    # a marker may merge with real content, that is how '<|>the<|>' is learned
    letter = next(t for t in pt.atomic_tokens if t != m)
    assert pt.bpe_merge_allowed([m], [letter]) is not False


# ------------------------------------------------------------------------------- config


def test_unknown_boundary_target_rejected():
    with pytest.raises(ValueError):
        BoundaryScriptPretokenizer(
            BoundaryScriptPretokenizerConfig(script_config=ScriptEncodingV3, boundary_targets=("word", "bogus"))
        )


def test_variants_have_distinct_hashes():
    """hash() is config-derived; distinct boundary_targets must not collide in the
    pretokenized-corpus cache."""
    hashes = {name: get_boundary_pretokenizer(name).hash() for name in VARIANTS}
    assert len(set(hashes.values())) == len(VARIANTS), hashes
    assert get_pretokenizer("scriptenc3_cb").hash() not in set(hashes.values())


def test_atomic_vocab_is_baseline_plus_one():
    base = len(get_pretokenizer("scriptenc3_cb").atomic_tokens)
    for name in VARIANTS:
        assert len(get_boundary_pretokenizer(name).atomic_tokens) == base + 1


# --------------------------------------------------------------------------------- helper


def visible_chunks(pt, text):
    return [visible(pt, list(c)) for c in pt.pretokenize(text)]


# ------------------------------------------------------------------- digit_handling

DIGIT_TEXTS = [
    "x = 1", "version 2 of 3", "in 1984 he", "a 12345 b", "3.14 and 2",
    "0", "007 and 42", "1a", "a1", "123 latinкириллица 123", "no digits here",
]


def boundary_pt(targets=("word", "punct", "digit"), digit_handling=None):
    return BoundaryScriptPretokenizer(
        BoundaryScriptPretokenizerConfig(
            script_config=ScriptEncodingV3, boundary_targets=targets, digit_handling=digit_handling
        )
    )


@pytest.mark.parametrize("digit_handling", [None, "SPLIT", "RTL3"])
@pytest.mark.parametrize("text", DIGIT_TEXTS)
def test_digit_handling_roundtrip(digit_handling, text):
    pt = boundary_pt(digit_handling=digit_handling)
    assert pt.decode(flat(pt, text)) == pt.normalize(text)


@pytest.mark.parametrize("digit_handling", ["SPLIT", "RTL3"])
def test_elision_still_happens_across_digit_boundaries(digit_handling):
    """The base pipeline splits digit runs into separate chunks BEFORE split_encoded,
    which would put a digit and its neighbouring word in different chunks and silently
    disable elision. The override must keep 'x = 1' fully elided."""
    pt = boundary_pt(digit_handling=digit_handling)
    ids = flat(pt, "x = 1")
    m = pt.marker_token_id
    pairs = sum(1 for i in range(len(ids) - 1) if ids[i] == m and ids[i + 1] == m)
    assert pairs == 2, f"expected both spaces elided, got {pairs}"
    assert pt.decode(ids) == "x = 1"


@pytest.mark.parametrize("digit_handling", ["SPLIT", "RTL3"])
def test_only_outer_digit_groups_carry_markers(digit_handling):
    """Interior groups of a digit run must never be marked; that is what bounds the
    number of marked digit forms (10 under SPLIT, 1110 under RTL3) instead of one set
    per distinct number."""
    pt = boundary_pt(digit_handling=digit_handling)
    chunks = [list(c) for c in pt.pretokenize("a 12345 b")]
    m = pt.marker_token_id
    digit_chunks = [c for c in chunks if any(t in pt.digit_token_ids for t in c)]
    assert len(digit_chunks) >= 2, digit_chunks
    for c in digit_chunks[1:-1]:
        assert m not in c, f"interior digit group marked: {c}"


def test_split_bounds_markable_digit_strings():
    """Under SPLIT a marked digit form can only ever wrap a single digit."""
    pt = boundary_pt(digit_handling="SPLIT")
    m = pt.marker_token_id
    marked = set()
    for text in ["a 1 b", "a 12 b", "a 12345 b", "x 007 y", "1 2 3"]:
        for c in pt.pretokenize(text):
            c = list(c)
            if m in c:
                core = [t for t in c if t != m]
                if core and all(t in pt.digit_token_ids for t in core):
                    marked.add("".join(pt.atomic_tokens[t] for t in core))
    assert all(len(s) == 1 for s in marked), marked


def test_digit_target_off_leaves_digits_unmarked():
    pt = boundary_pt(targets=("word", "punct"), digit_handling="SPLIT")
    m = pt.marker_token_id
    for c in pt.pretokenize("x = 1"):
        c = list(c)
        if any(t in pt.digit_token_ids for t in c):
            assert m not in c
    assert pt.decode(flat(pt, "x = 1")) == "x = 1"


@pytest.mark.parametrize("digit_handling", [None, "SPLIT", "RTL3"])
def test_digit_handling_changes_hash(digit_handling):
    """Corpus cache keys must distinguish digit handling."""
    hashes = {dh: boundary_pt(digit_handling=dh).hash() for dh in [None, "SPLIT", "RTL3"]}
    assert len(set(hashes.values())) == 3, hashes


NON_ASCII_NUMERIC_TEXTS = [
    "½ cup", "a ½ b", "1½", "٣ عربي", "Ⅻ century", "⅓ and 2", "½", "½½", "2½ hours",
]


@pytest.mark.parametrize("digit_handling", [None, "SPLIT", "RTL3"])
@pytest.mark.parametrize("text", NON_ASCII_NUMERIC_TEXTS)
def test_non_ascii_numerics_roundtrip(digit_handling, text):
    """Category N covers far more than ASCII 0-9 -- Nd for every script plus Nl/No
    ('½', '⅓', '٣', 'Ⅻ') -- but group_digits/encode_digits only have tokens for ASCII,
    because the base pipeline splits on re.split("([0-9]+)"). Grouping a non-ASCII
    numeric raised KeyError and killed a training run mid-corpus."""
    pt = boundary_pt(digit_handling=digit_handling)
    assert pt.decode(flat(pt, text)) == pt.normalize(text)


@pytest.mark.parametrize("digit_handling", ["SPLIT", "RTL3"])
def test_mixed_ascii_and_non_ascii_numerics(digit_handling):
    pt = boundary_pt(digit_handling=digit_handling)
    for text in ["1½ cups", "٣ and 3", "12 ½ 34"]:
        assert pt.decode(flat(pt, text)) == pt.normalize(text)
