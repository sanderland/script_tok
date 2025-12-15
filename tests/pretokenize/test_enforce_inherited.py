import pytest

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.utils import token_array


def token_ids_for(pt, s: str):
    chs = pt.encode_text(s)
    return token_array([tid for ch in chs for tid in ch.atomic_token_ids])


@pytest.mark.parametrize(
    "name",
    [
        "scriptenc_cbi",
        "bytes_nosplit_cbi",
        "bytes_gpt4_cbi",
        "bytes_gpt4o_cbi",
        "scriptenc_gpt4o_cbi",
        "scriptenc2_cbi",
    ],
)
def test_inherited_at_start_disallows_multi_char(name):
    pt = get_pretokenizer(name)
    inherited = "\ufe0f"  # VARIATION SELECTOR-16 (Inherited)
    base = "a"

    single = token_ids_for(pt, inherited)
    assert pt.token_allowed(single)

    two = token_ids_for(pt, inherited + base)
    assert not pt.token_allowed(two)


@pytest.mark.parametrize(
    "name",
    [
        "scriptenc_cb",
        "bytes_nosplit_cb",
        "bytes_gpt4_cb",
        "bytes_gpt4o_cb",
        "scriptenc_gpt4o_cb",
    ],
)
def test_no_enforcement_allows_multi_char(name):
    pt = get_pretokenizer(name)
    inherited = "\ufe0f"
    base = "a"
    seq = token_ids_for(pt, inherited + base)
    assert pt.token_allowed(seq)
