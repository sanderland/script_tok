from .pretokenizer import (
    Pretokenizer,
    PretokenizerConfig,
    ScriptPretokenizerConfig,
    UTF8PretokenizerConfig,
)
from .scriptencoding import ScriptEncodingV1 as ScriptEncodingV1, ScriptEncodingV2 as ScriptEncodingV2

GPT2_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
GPT4_REGEX = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
GPT4O_REGEX = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)


# Registry of available pretokenizers
PRETOKENIZER_REGISTRY: dict[str, PretokenizerConfig] = {
    "bytes_gpt4": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX, enforce_char_boundaries=False),
    "bytes_gpt4_cb": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX),
    "bytes_gpt4o": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, enforce_char_boundaries=False),
    "bytes_gpt4o_cb": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX),
    "bytes_nosplit_cb": UTF8PretokenizerConfig(regex_pattern=None),
    "scriptenc": ScriptPretokenizerConfig(enforce_char_boundaries=False),
    "scriptenc_cb": ScriptPretokenizerConfig(),
    "scriptenc_gpt4o": ScriptPretokenizerConfig(
        regex_pattern=GPT4O_REGEX, script_split=False, enforce_char_boundaries=False
    ),
    "scriptenc_gpt4o_cb": ScriptPretokenizerConfig(regex_pattern=GPT4O_REGEX, script_split=False),
    "scriptenc_nosplit_cb": ScriptPretokenizerConfig(regex_pattern=None, script_split=False),
    "scriptenc2_cb": ScriptPretokenizerConfig(script_config=ScriptEncodingV2, enforce_char_boundaries=True),
    "scriptenc2_gpt4o_cb": ScriptPretokenizerConfig(
        script_config=ScriptEncodingV2, regex_pattern=GPT4O_REGEX, script_split=False
    ),
}


def get_pretokenizer(name: str) -> Pretokenizer:
    if name not in PRETOKENIZER_REGISTRY:
        raise ValueError(f"Pretokenizer '{name}' is not registered. Available: {list(PRETOKENIZER_REGISTRY.keys())}")
    config = PRETOKENIZER_REGISTRY[name]
    for config_type, pretokenizer_cls in Pretokenizer.REGISTRY.values():
        if config_type is config.__class__:
            return pretokenizer_cls(config)
    raise ValueError(f"config.__class__: {config.__class__} not found in any pretokenizer")


def load_pretokenizer(serialized: dict) -> Pretokenizer:
    config_type, pretokenizer_cls = Pretokenizer.REGISTRY[serialized["config_class"]]
    config = config_type.model_validate(serialized["config"])
    return pretokenizer_cls(config)


def export_pretokenizer(pretokenizer: Pretokenizer) -> dict:
    return dict(config_class=pretokenizer.__class__.__name__, config=pretokenizer.config.model_dump())
