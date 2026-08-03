"""Caps codes only share a vocabulary entry if they sit outside the delimited span.

The shipped layout puts the code inside: `<|><^>the<|>`, a different chunk from
`<|>the<|>`, so the trainer learns a separate token and the case duplication survives.
The extcaps layout puts the code in its own chunk, so the word token is byte-identical
between cases and `The` is `the` plus one code.
"""

import pytest

from marker_experiments.boundary_pretokenizer import get_boundary_pretokenizer


def _chunks(pt, text):
    mk, sh, cp = pt.marker_token_id, pt.shift_token_id, pt.caps_token_id
    lab = {mk: "<|>", sh: "<^>", cp: "<^^>"}
    out = []
    for chunk in pt.pretokenize(text):
        s, run = "", []
        for i in chunk:
            if i in lab:
                if run:
                    s += pt.try_decode_strict(run) or "?"
                    run = []
                s += lab[i]
            else:
                run.append(i)
        if run:
            s += pt.try_decode_strict(run) or "?"
        out.append(s)
    return out


def _roundtrip(pt, text):
    return pt.decode([t for chunk in pt.pretokenize(text) for t in chunk])


@pytest.fixture
def ext():
    return get_boundary_pretokenizer("bnd_wpd_extcaps")


def test_code_sits_outside_the_markers(ext):
    """Same pre-token as the word, but outside its markers, so the pre-token count is
    unchanged and no code is stranded as a token of its own."""
    assert _chunks(ext, "The cat") == ["<^><|>the<|>", "<|>cat<|>"]
    assert _chunks(ext, "THE cat") == ["<^^><|>the<|>", "<|>cat<|>"]
    assert len(_chunks(ext, "The cat")) == len(_chunks(ext, "the cat"))


def test_lowercase_span_is_available_as_a_piece(ext):
    """The property the design exists for: the atomic sequence of the lowercase chunk
    occurs inside the cased one, so the trainer MAY cover it with the same piece. It is
    not obliged to -- merging the code in is the right call when frequency justifies it."""
    lower = [t for c in ext.pretokenize("the") for t in c]
    title = [t for c in ext.pretokenize("The") for t in c]
    upper = [t for c in ext.pretokenize("THE") for t in c]
    assert title[1:] == lower, "title-case span must end with the lowercase sequence"
    assert upper[1:] == lower
    assert title[0] == ext.shift_token_id and upper[0] == ext.caps_token_id


def test_inside_layout_does_not_share(ext):
    """Contrast, so a regression that silently reverts the layout is caught."""
    inside = get_boundary_pretokenizer("bnd_wpd_caps")
    assert _chunks(inside, "The cat")[0] == "<|><^>the<|>"
    assert _chunks(inside, "the cat")[0] == "<|>the<|>"


def test_space_still_elides_across_a_code(ext):
    """Two delimited spans separated only by a code had a space between them."""
    assert _chunks(ext, "a The b") == ["<|>a<|>", "<^><|>the<|>", "<|>b<|>"]
    assert _roundtrip(ext, "a The b") == "a The b"


@pytest.mark.parametrize("text", [
    "", "the cat", "The cat", "THE cat", "a The b", "The", "A", "I am",
    "The NASA report, 2024.", "MiXeD GaN WiFi stays literal",
    "Hello, World! The End.", "x  y", "\nThe\n", "one\ttwo",
    "Ünicode Ärger ÅNGSTRÖM", "кириллица Кириллица КИРИЛЛИЦА",
    "The 1980 Fagan Park, \"Ida\", to A.B.", "A B C D", "THE THE the The",
])
def test_roundtrip(ext, text):
    assert _roundtrip(ext, text) == text


def test_roundtrip_on_a_real_document(ext):
    text = open("tests/data/taylorswift.txt", encoding="utf-8").read()
    assert _roundtrip(ext, text) == text


def test_mixed_case_stays_literal(ext):
    """Only whole-span title or upper case is coded; anything else must not be touched."""
    assert _chunks(ext, "GaN") == ["<|>GaN<|>"]
    assert _roundtrip(ext, "GaN WiFi") == "GaN WiFi"


def test_distinct_hash_from_the_inside_layout(ext):
    """Different chunking, so it must never share a corpus cache with the other layout."""
    inside = get_boundary_pretokenizer("bnd_wpd_caps")
    plain = get_boundary_pretokenizer("bnd_wpd")
    assert len({ext.hash(), inside.hash(), plain.hash()}) == 3


def test_existing_variant_hashes_are_unchanged():
    """Adding extcaps must not invalidate corpora already built elsewhere."""
    assert get_boundary_pretokenizer("bnd_wpd").hash() == "PT-0cddd747"
    assert get_boundary_pretokenizer("bnd_wpd_caps").hash() == "PT-b75c6555"
