import unicodedata

import pytest

from script_bpe.encoding import ScriptEncodingV1
from script_bpe.encoding.script_util import END_CODEPOINT, unicode_script_map
from script_bpe.utils import UNASSIGNED_CATEGORIES


@pytest.fixture
def sc_map():
    return unicode_script_map()


@pytest.fixture
def script_encoding():
    return ScriptEncodingV1()


def test_build_script_encoding(script_encoding):
    blocks = script_encoding.blocks
    config = script_encoding.config
    assert isinstance(config, dict)
    assert isinstance(blocks, list)
    assert "version" in config
    assert "num_index_tokens" in config
    assert "num_blocks" in config
    assert isinstance(config["num_index_tokens"], int)
    assert isinstance(config["num_blocks"], int)
    assert len(blocks) == config["num_blocks"]

    seen_chars = set()
    seen_sids = set()
    seen_sss = set()
    for sid, script, supercat, sub_block_id, cs in blocks:
        assert isinstance(sid, int)
        if script != "Hiragana":
            assert (sid, sub_block_id) not in seen_sids
            seen_sids.add((sid, sub_block_id))

        assert isinstance(script, str)
        assert isinstance(supercat, str)
        assert isinstance(sub_block_id, int)
        assert (script, supercat, sub_block_id) not in seen_sss
        seen_sss.add((script, supercat, sub_block_id))

        assert isinstance(cs, str)
        assert len(cs) > 0
        for c in cs:
            assert c not in seen_chars
            seen_chars.add(c)


def test_unicode_script_map_unknown_script_cat(sc_map):
    for c, entry in sc_map.items():
        assert "script" in entry
        assert "category" in entry

    # Check unassigned codepoints are in UNASSIGNED_CATEGORIES
    for i in range(END_CODEPOINT):
        if chr(i) not in sc_map:
            category = unicodedata.category(chr(i))
            assert category in UNASSIGNED_CATEGORIES, f"Unexpected category {category} for U+{ord(c):X} script None"


