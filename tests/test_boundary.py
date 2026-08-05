"""Tests for boundary-marker pretokenization.

Run: uv run pytest tests/test_boundary.py -q
"""

import functools

import pytest

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3

from paper_utils.boundary.boundary_pretokenizer import (
    BOUNDARY_VARIANTS,
    BoundaryScriptPretokenizer,
    BoundaryScriptPretokenizerConfig,
    get_boundary_pretokenizer,
)
from paper_utils.boundary.vocab_duplicates import decase, despace, duplicate_slots, special_texts, surface

VARIANTS = list(BOUNDARY_VARIANTS)
# The three boundary scopes without case codes. Several invariants are about the markers
# alone and are cheaper to state over these than over all six.
SCOPES = ["bnd_w", "bnd_wp", "bnd_wpd"]

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
    "naïve café résumé", "ﬁle", "élan",  # combining marks / inherited
]

CASE_TEXTS = [
    "The cat", "NASA rocket", "GaN WiFi", "the cat", "A", "I am", "McDonald",
    "Hello, World!", "ALL CAPS HERE", "Title Case Words", "iPhone", "eBay", "MiXeD",
    "Привет Мир", "ПРИВЕТ", "Ελλάδα", "ΕΛΛΑΔΑ", "ΟΔΟΣ", "Οδος", "ές ΟΔΟΣ",
    "Ünicode Ötzi", "ﬁle", "İstanbul", "STRASSE", "Straße", "ẞ", "ǅungla", "Ǆ",
    "ARMÉE", "Ångström", "x = 1", "A1", "",
    "The NASA report, 2024.", "MiXeD GaN WiFi stays literal", "\nThe\n", "one\ttwo",
    "кириллица Кириллица КИРИЛЛИЦА", 'The 1980 Fagan Park, "Ida", to A.B.',
    "A B C D", "THE THE the The",
]

DIGIT_TEXTS = [
    "x = 1", "version 2 of 3", "in 1984 he", "a 12345 b", "3.14 and 2",
    "0", "007 and 42", "1a", "a1", "123 latinкириллица 123", "no digits here",
]

NON_ASCII_NUMERIC_TEXTS = [
    "½ cup", "a ½ b", "1½", "٣ عربي", "Ⅻ century", "⅓ and 2", "½", "½½", "2½ hours",
]


@functools.cache
def _build(targets, options):
    return BoundaryScriptPretokenizer(
        BoundaryScriptPretokenizerConfig(
            script_config=ScriptEncodingV3, boundary_targets=targets, **dict(options))
    )


def boundary_pt(targets=("punct", "digit"), **kw):
    """A pretokenizer straight from the config, for options the variant names do not cover.

    Cached: a construction costs about 0.13s and builds the whole atomic vocabulary, and
    the tests below ask for the same dozen configurations once per parametrized case.
    Pretokenizers are not mutated by anything here, so sharing one is safe.
    """
    return _build(tuple(targets), tuple(sorted(kw.items())))


def flat(pt, text):
    return [t for c in pt.pretokenize(text) for t in c]


def visible(pt, ids):
    """Render marker and case-code tokens literally; decode() turns them into spaces,
    capitals or nothing. The paper's duplicate count is built on the same repr, and
    depends on it: decoded, a delimited word and a word-internal fragment are one string."""
    return surface(pt, ids, special_texts(pt))


def visible_chunks(pt, text):
    return [visible(pt, list(c)) for c in pt.pretokenize(text)]


def code_count(pt, text):
    return sum(1 for i in flat(pt, text) if i in pt.caps_code_ids)


@pytest.fixture(scope="module", params=VARIANTS)
def pt(request):
    return get_boundary_pretokenizer(request.param)


@pytest.fixture(scope="module")
def caps():
    """The paper's case-coded arm."""
    return get_boundary_pretokenizer("bnd_wpd_caps")


# ------------------------------------------------------------------ the span invariants


def assert_span_invariants(pt, text):
    """Everything that must hold of any scheme's output for any text.

    One function rather than three tests over the same parametrization: each of these
    needs the same pretokenization, and asserting them separately meant encoding every
    text three times per scheme. The messages still say which invariant broke.
    """
    ids = flat(pt, text)
    m = pt.marker_token_id

    assert pt.decode(ids) == pt.normalize(text), f"round trip failed for {text!r}"

    # Every <|><|> pair must correspond to a real single space in the source.
    pairs, i = 0, 0
    while i < len(ids):
        if ids[i] == m and i + 1 < len(ids) and ids[i + 1] == m:
            pairs += 1
            i += 2
        else:
            i += 1
    assert pairs <= pt.normalize(text).count(" "), f"more elided spaces than exist in {text!r}"

    # Three markers in a row would be ambiguous (one space plus a lone marker, or the
    # reverse). Span merging plus asymmetric punctuation marking must prevent it.
    for i in range(len(ids) - 2):
        assert not (ids[i] == m and ids[i + 1] == m and ids[i + 2] == m), f"triple marker in {text!r}"


@pytest.mark.parametrize("text", TEXTS + CASE_TEXTS)
def test_span_invariants(pt, text):
    assert_span_invariants(pt, text)


def test_span_invariants_on_a_real_document(pt):
    with open("tests/data/taylorswift.txt", encoding="utf-8") as f:
        assert_span_invariants(pt, f.read())


def test_multi_space_runs_untouched(pt):
    """Only a single space is ever elided; runs are left exactly as the baseline emits."""
    for text in ["a  b", "a   b", "a\t b"]:
        assert pt.decode(flat(pt, text)) == pt.normalize(text)
    assert "  " in "".join(visible_chunks(pt, "a  b"))


def test_leading_and_trailing_space_not_elided(pt):
    for text in [" a", "a "]:
        assert pt.decode(flat(pt, text)) == pt.normalize(text)


# -------------------------------------------------- span merging (no middle marker)


@pytest.mark.parametrize("name", VARIANTS)
def test_cross_script_word_span_has_no_internal_marker(name):
    """The whole point of merging spans: latin+Cyrillic with no space is ONE span, so no
    marker appears between them and no phantom space can be decoded."""
    pt = get_boundary_pretokenizer(name)
    ids = flat(pt, "latinкириллица")
    m = pt.marker_token_id
    assert ids[0] == m and ids[-1] == m, f"{name}: span not delimited"
    assert ids.count(m) == 2, f"{name}: expected exactly 2 markers, got {ids.count(m)}"
    assert pt.decode(ids) == "latinкириллица"


def test_word_span_always_delimited_both_sides():
    pt = get_boundary_pretokenizer("bnd_wpd")
    m = pt.marker_token_id
    for text in ["word", "(word", "word)", "1word", "word1", ",word,", "=word="]:
        ids = flat(pt, text)
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


# ----------------------------------------------------- boundary_targets behaviour


def test_word_spans_are_delimited_under_every_scope():
    """`word` is not a member of boundary_targets: it is always on."""
    for targets in [(), ("punct",), ("digit",), ("punct", "digit")]:
        pt = boundary_pt(targets=targets)
        assert visible_chunks(pt, "a b") == ["<|>a<|>", "<|>b<|>"], targets


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
    assert visible_chunks(wpd, "x = 1") == ["<|>x<|>", "<|>=<|>", "<|>1"]


def test_word_only_variant_leaves_punct_and_digits_bare():
    pt = get_boundary_pretokenizer("bnd_w")
    assert visible_chunks(pt, "a, b") == ["<|>a<|>", ",", " ", "<|>b<|>"]
    assert visible_chunks(pt, "a b") == ["<|>a<|>", "<|>b<|>"]


def test_unknown_boundary_target_rejected():
    with pytest.raises(ValueError):
        BoundaryScriptPretokenizerConfig(script_config=ScriptEncodingV3, boundary_targets=("bogus",))


def test_word_is_not_a_boundary_target():
    """Word spans are delimited unconditionally, so "word" is not a selectable target.
    Passing it must fail loudly rather than be silently ignored."""
    with pytest.raises(ValueError):
        BoundaryScriptPretokenizerConfig(script_config=ScriptEncodingV3, boundary_targets=("word", "punct"))


# ------------------------------------------------------------------------- config hashes


def assert_distinct_hashes(by_label):
    """No two configurations that chunk differently may share a hash.

    hash() is config-derived and keys the pretokenized-corpus cache, so a collision does
    not fail anywhere -- it silently trains one scheme on another's corpus.
    """
    hashes = {label: pt.hash() for label, pt in by_label.items()}
    assert len(set(hashes.values())) == len(hashes), hashes
    assert get_pretokenizer("scriptenc3_cb").hash() not in set(hashes.values())


def test_variants_have_distinct_hashes():
    assert_distinct_hashes({n: get_boundary_pretokenizer(n) for n in VARIANTS})


def test_digit_handling_has_distinct_hashes():
    assert_distinct_hashes({dh: boundary_pt(digit_handling=dh) for dh in [None, "SPLIT", "RTL3"]})


def test_case_options_have_distinct_hashes():
    # Every configuration in the table, keyed by the same ids the other two tests use.
    assert_distinct_hashes({i: boundary_pt(**o) for i, (o, _) in zip(CASE_IDS, CASE_OPTIONS)})


@pytest.mark.parametrize("scope", SCOPES)
def test_atomic_vocab_is_baseline_plus_marker_and_enabled_codes(scope):
    """The marker always costs one slot; each enabled case code costs one more, and a
    disabled one costs nothing."""
    base = len(get_pretokenizer("scriptenc3_cb").atomic_tokens)
    assert len(get_boundary_pretokenizer(scope).atomic_tokens) == base + 1
    assert len(get_boundary_pretokenizer(f"{scope}_caps").atomic_tokens) == base + 3
    assert len(boundary_pt(shift_code=True).atomic_tokens) == base + 2
    assert len(boundary_pt(caps_code=True).atomic_tokens) == base + 2
    off = boundary_pt()
    assert off.shift_token_id is None and off.caps_token_id is None and off.caps_code_ids == set()


# ------------------------------------------------------------------- digit_handling


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
    pt = boundary_pt(targets=("punct",), digit_handling="SPLIT")
    m = pt.marker_token_id
    for c in pt.pretokenize("x = 1"):
        c = list(c)
        if any(t in pt.digit_token_ids for t in c):
            assert m not in c
    assert pt.decode(flat(pt, "x = 1")) == "x = 1"


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


# ----------------------------------------------------------------------- case codes


def test_code_sits_outside_the_markers(caps):
    """Same pre-token as the word, but outside its markers, so the pre-token count is
    unchanged and no code is stranded as a token of its own."""
    assert visible_chunks(caps, "The cat") == ["<^><|>the<|>", "<|>cat<|>"]
    assert visible_chunks(caps, "THE cat") == ["<^^><|>the<|>", "<|>cat<|>"]
    assert len(visible_chunks(caps, "The cat")) == len(visible_chunks(caps, "the cat"))


def test_lowercase_span_is_available_as_a_piece(caps):
    """The property the design exists for: the atomic sequence of the lowercase chunk
    occurs inside the cased one, so the trainer MAY cover it with the same piece. It is
    not obliged to -- merging the code in is the right call when frequency justifies it."""
    lower, title, upper = flat(caps, "the"), flat(caps, "The"), flat(caps, "THE")
    assert title[1:] == lower, "title-case span must end with the lowercase sequence"
    assert upper[1:] == lower
    assert title[0] == caps.shift_token_id and upper[0] == caps.caps_token_id


def test_space_still_elides_across_a_code(caps):
    """Two delimited spans separated only by a code had a space between them."""
    assert visible_chunks(caps, "a The b") == ["<|>a<|>", "<^><|>the<|>", "<|>b<|>"]
    assert caps.decode(flat(caps, "a The b")) == "a The b"


def test_case_codes_applied_where_invertible(caps):
    assert flat(caps, "The")[0] == caps.shift_token_id
    assert flat(caps, "NASA")[0] == caps.caps_token_id
    # mixed case is left literal, as in the tokenizer this mirrors
    for text in ["GaN", "WiFi", "the", "iPhone"]:
        assert code_count(caps, text) == 0, text
    assert visible_chunks(caps, "GaN") == ["<|>GaN<|>"]


def test_case_codes_skipped_when_not_invertible(caps):
    """Unicode case mapping is not a bijection. U+0130 lowercases to two characters and
    U+1E9E uppercases to 'SS', so neither may take a case code."""
    for text in ["İstanbul", "İ", "ẞ", "ǅungla"]:
        assert code_count(caps, text) == 0, text
        assert caps.decode(flat(caps, text)) == caps.normalize(text)


@pytest.mark.parametrize("text", ["العربية نص", "한국어 텍스트", "עברית טקסט", "हिन्दी पाठ"])
def test_uncased_scripts_are_left_alone(caps, text):
    """`str.islower()` is False for text with no cased character, so a test for "not
    lowercase" reads every Arabic, Hangul, Hebrew or Devanagari span as title case and
    codes it. `istitle() or isupper()` is what those spans have to fail."""
    assert code_count(caps, text) == 0
    assert caps.decode(flat(caps, text)) == text


@pytest.mark.parametrize("digit_handling", ["SPLIT", "RTL3"])
def test_case_roundtrip_with_digit_splitting(digit_handling):
    pt = boundary_pt(shift_code=True, caps_code=True, digit_handling=digit_handling)
    for text in CASE_TEXTS + DIGIT_TEXTS:
        assert pt.decode(flat(pt, text)) == pt.normalize(text)


# ------------------------------------------------------- the individual case options

# "The" is title case, "A" is a single-character title case, "NASA" and "ABC" are all
# caps of length 4 and 3. One string exercises every case option.
CASE_PROBE = "The NASA A ABC ok"

# One table, three properties: how many spans each configuration codes, that every
# configuration round-trips, and that no two of them share a hash. Stated once because a
# second copy drifts the first time an option is added -- which is how the list here came
# to differ from the list two tests below it.
CASE_OPTIONS = [
    (dict(), 0),                                                          # both codes off
    (dict(shift_code=True), 2),                                           # The, A
    (dict(caps_code=True), 2),                                            # NASA, ABC
    (dict(shift_code=True, caps_code=True), 4),
    (dict(shift_code=True, caps_code=True, single_char_shift=False), 3),  # drops A
    (dict(shift_code=True, caps_code=True, min_caps_length=4), 3),        # drops ABC
    (dict(shift_code=True, caps_code=True, min_caps_length=5), 2),        # and NASA
    (dict(shift_code=True, single_char_shift=False), 1),                  # The only
]
CASE_IDS = [",".join(f"{k}={v}" for k, v in o.items()) or "none" for o, _ in CASE_OPTIONS]


@pytest.mark.parametrize("options,expected", CASE_OPTIONS, ids=CASE_IDS)
def test_each_case_option_gates_its_own_spans(options, expected):
    pt = boundary_pt(**options)
    assert code_count(pt, CASE_PROBE) == expected
    assert pt.decode(flat(pt, CASE_PROBE)) == CASE_PROBE


@pytest.mark.parametrize("options", [o for o, _ in CASE_OPTIONS], ids=CASE_IDS)
def test_every_case_option_combination_roundtrips(options):
    pt = boundary_pt(**options)
    for text in CASE_TEXTS + TEXTS:
        assert pt.decode(flat(pt, text)) == pt.normalize(text), (options, text)


def test_a_disabled_code_leaves_its_spans_literal():
    """The span is not coded and not lowercased -- it comes through exactly as written."""
    shift_only = boundary_pt(shift_code=True)
    caps_only = boundary_pt(caps_code=True)
    assert visible_chunks(shift_only, "NASA") == ["<|>NASA<|>"]
    assert visible_chunks(shift_only, "The") == ["<^><|>the<|>"]
    assert visible_chunks(caps_only, "The") == ["<|>The<|>"]
    assert visible_chunks(caps_only, "NASA") == ["<^^><|>nasa<|>"]


def test_single_char_shift_off_leaves_one_letter_spans_literal():
    off = boundary_pt(shift_code=True, caps_code=True, single_char_shift=False)
    on = boundary_pt(shift_code=True, caps_code=True, single_char_shift=True)
    assert visible_chunks(off, "A b") == ["<|>A<|>", "<|>b<|>"]
    assert visible_chunks(on, "A b") == ["<^><|>a<|>", "<|>b<|>"]
    # a multi-character title-case span is unaffected either way
    assert visible_chunks(off, "The") == visible_chunks(on, "The") == ["<^><|>the<|>"]


def test_min_caps_length_leaves_short_acronyms_literal():
    pt = boundary_pt(shift_code=True, caps_code=True, min_caps_length=4)
    assert visible_chunks(pt, "ABC") == ["<|>ABC<|>"]
    assert visible_chunks(pt, "NASA") == ["<^^><|>nasa<|>"]


# ------------------------------------------------------- duplicate vocabulary entries


def test_duplicate_count_reads_markers_that_decoding_would_erase(caps):
    """The repr the paper's duplicate count is built on, and why it is not `decode`."""
    whole = flat(caps, "the")
    fragment = whole[1:-1]  # the same letters without the markers: a word-internal piece
    assert visible(caps, whole) == "<|>the<|>"
    assert visible(caps, fragment) == "the"
    # Decoded, the two are one string, so a count taken over decoded forms reports two
    # distinct entries as a duplicate pair.
    assert caps.decode(whole) == caps.decode(fragment) == "the"


def test_duplicate_keys_group_only_the_forms_each_scheme_removes():
    forms = ["<|>the<|>", "the", " the", "The", "<^><|>the<|>", "<^^><|>the<|>"]
    # A leading space is the baseline's word-initial form: `the` and ` the`, two slots.
    assert duplicate_slots(forms, despace) == 2
    # Case ignores the space and strips a code, so `The` joins `the`, and both coded spans
    # join the span they code: 2 + 3.
    assert duplicate_slots(forms, decase) == 5
    # Neither counts a form the vocabulary holds once.
    assert duplicate_slots(["<|>the<|>", "theory", " and"], despace) == 0
    assert duplicate_slots(["<|>the<|>", "theory", " and"], decase) == 0
