import unicodedata

import pytest

from script_bpe.encoding import END_CODEPOINT, ScriptEncodingV1, ScriptEncodingV2, unicode_script_map


@pytest.fixture
def sc_map():
    return unicode_script_map()


@pytest.mark.parametrize("script_encoding_cls", [ScriptEncodingV1, ScriptEncodingV2])
def test_build_script_encoding(script_encoding_cls):
    script_encoding = script_encoding_cls()
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
    for block in blocks:
        assert isinstance(block["script_id"], int)
        if block["script"] != "Hiragana":
            assert (block["script_id"], block["sub_block_id"]) not in seen_sids
            seen_sids.add((block["script_id"], block["sub_block_id"]))

        assert isinstance(block["script"], str)
        assert isinstance(block["category"], str)
        assert isinstance(block["sub_block_id"], int)
        assert (block["script"], block["category"], block["sub_block_id"]) not in seen_sss
        seen_sss.add((block["script"], block["category"], block["sub_block_id"]))

        assert isinstance(block["chars"], str)
        assert len(block["chars"]) > 0
        for c in block["chars"]:
            assert c not in seen_chars
            seen_chars.add(c)


def test_unicode_script_map_unknown_script_cat(sc_map):
    for c, entry in sc_map.items():
        assert "script" in entry
        assert "category" in entry
    UNASSIGNED_CATEGORIES = {"Cn", "Co", "Cs"}

    # Check unassigned codepoints are in UNASSIGNED_CATEGORIES
    for i in range(END_CODEPOINT):
        if chr(i) not in sc_map:
            category = unicodedata.category(chr(i))
            assert category in UNASSIGNED_CATEGORIES, f"Unexpected category {category} for U+{ord(c):X} script None"
