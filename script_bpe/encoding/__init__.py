from types import SimpleNamespace

from script_bpe.pretokenize.scriptencoding import (
    END_CODEPOINT,
    ScriptEncodingV1 as _SCRIPTENC_V1_CFG,
    ScriptEncodingV2 as _SCRIPTENC_V2_CFG,
    unicode_script_map,
    ScriptConfig,
    ScriptBlock,
)


def _to_legacy(config: ScriptConfig) -> SimpleNamespace:
    blocks = [
        {
            "script_id": b.script_id,
            "script": b.script,
            "category": b.category,
            "sub_block_id": b.sub_block_id,
            "chars": b.chars,
        }
        for b in config.blocks
    ]
    num_index_tokens = max(len(b["chars"]) for b in blocks) if blocks else 0
    version = 1 if config.supercategory_type == "V1" else 2
    cfg_dict = {
        "version": version,
        "num_index_tokens": num_index_tokens,
        "num_blocks": len(blocks),
    }
    return SimpleNamespace(blocks=blocks, config=cfg_dict)


def ScriptEncodingV1() -> SimpleNamespace:
    return _to_legacy(_SCRIPTENC_V1_CFG)


def ScriptEncodingV2() -> SimpleNamespace:
    return _to_legacy(_SCRIPTENC_V2_CFG)


__all__ = [
    "END_CODEPOINT",
    "ScriptEncodingV1",
    "ScriptEncodingV2",
    "unicode_script_map",
    "ScriptConfig",
    "ScriptBlock",
]


