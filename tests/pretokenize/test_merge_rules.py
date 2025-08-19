import pytest

from script_bpe.pretokenize import get_pretokenizer


@pytest.fixture
def regex_utf8b():
    return get_pretokenizer("bytes_gpt4o_cb")


@pytest.fixture
def scriptenc_cb():
    return get_pretokenizer("scriptenc_cb")


@pytest.fixture
def scriptenc():
    return get_pretokenizer("scriptenc")


UTF8_TEST_CASES = [
    # Complete + Complete (Allowed)
    pytest.param("a", "b", True, id="A+B"),
    pytest.param("ü", "a", True, id="Uml+A"),
    pytest.param("a", "ü", True, id="A+Uml"),
    pytest.param("ü", "ü", True, id="Uml+Uml"),
    pytest.param("한", "한", True, id="Han+Han"),
    pytest.param("ab", "ü", True, id="AB+Uml"),
    # Partial + Partial (Allowed - forms char)
    pytest.param(("ü", slice(0, 1)), ("ü", slice(1, 2)), True, id="Uml_P1+P2"),
    pytest.param(("한", slice(0, 2)), ("한", slice(2, 3)), True, id="Han_P12+P3"),
    # Complete + Partial (Disallowed)
    pytest.param("a", ("ü", slice(0, 1)), False, id="A+Uml_P1"),
   # pytest.param("ü", ("한", slice(2, 3)), False, id="Uml+Han_P3_Cont"), # cant happen
    # Partial + Complete (Disallowed)
    pytest.param(("ü", slice(0, 1)), "a", False, id="Uml_P1+A"),
    pytest.param(("ü", slice(1, 2)), "a", False, id="Uml_P2_Cont+A"),
    # Partial + Partial (Disallowed - invalid combo)
    pytest.param(("ü", slice(1, 2)), ("ü", slice(1, 2)), False, id="Cont+Cont"),
    pytest.param(("ü", slice(0, 1)), ("ü", slice(0, 1)), False, id="Start+Start"),
    pytest.param(("ü", slice(1, 2)), ("한", slice(0, 1)), False, id="Cont+Start"),
]


@pytest.mark.parametrize("marker1, marker2, expected", UTF8_TEST_CASES)
def test_bpe_merge_allowed_ext(regex_utf8b, marker1, marker2, expected: bool):
    """Tests bpe_merge_allowed using pre-tokenized sequences."""
    regex_utf8b_tokenizer = regex_utf8b
    seqs = []
    for marker in [marker1, marker2]:
        base_str = marker if isinstance(marker, str) else marker[0]
        slice_obj = marker[1] if isinstance(marker, tuple) else slice(None)
        chunks = regex_utf8b_tokenizer.pretokenize(base_str)
        seqs.append(chunks[0][slice_obj])

    result = regex_utf8b_tokenizer.bpe_merge_allowed(seqs[0], seqs[1])
    assert result == expected


SE_TEST_CASES = [pytest.param(a, b, id=f"{a}+{b}") for a in ["a", "ü", "한", "ab"] for b in ["a", "ü", "한", "ab"]]


@pytest.mark.parametrize("a, b", SE_TEST_CASES)
def test_merge_scriptenc_cb(a, b, scriptenc_cb):
    c1 = scriptenc_cb.pretokenize(a)[0]
    c2 = scriptenc_cb.pretokenize(b)[0]
    assert scriptenc_cb.bpe_merge_allowed(c1, c2)
    assert scriptenc_cb.bpe_merge_allowed([c1[0]], [c1[1]])
    assert scriptenc_cb.bpe_merge_allowed([c1[0]], [c1[1]])
    assert not scriptenc_cb.bpe_merge_allowed([c1[1]], [c1[0]])
    assert not scriptenc_cb.bpe_merge_allowed(c1, [c2[0]])
    assert not scriptenc_cb.bpe_merge_allowed([c1[-1]], [c2[0]])
    assert not scriptenc_cb.bpe_merge_allowed([c1[-1]], c2)


@pytest.mark.parametrize("a, b", SE_TEST_CASES)
def test_merge_scriptenc(a, b, scriptenc):  # all allowed
    c1 = scriptenc.pretokenize(a)[0]
    c2 = scriptenc.pretokenize(b)[0]
    assert scriptenc.bpe_merge_allowed(c1, c2)
    assert scriptenc.bpe_merge_allowed([c1[0]], [c1[1]])
    assert scriptenc.bpe_merge_allowed([c1[1]], [c1[0]])
    assert scriptenc.bpe_merge_allowed(c1, [c2[0]])
    assert scriptenc.bpe_merge_allowed([c1[-1]], [c2[0]])
    assert scriptenc.bpe_merge_allowed([c1[-1]], c2)
