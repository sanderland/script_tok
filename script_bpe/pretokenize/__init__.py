from typing import Callable

from .config import (
    PretokenizerConfig,
    Pretokenizer,
    UTF8Pretokenizer,
    ScriptPretokenizer,
    UTF8PretokenizerConfig,
    ScriptPretokenizerConfig,
)
from .scriptenc import (
    ScriptEncodingV1,
    ScriptEncodingV2,
)


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
PRETOKENIZER_REGISTRY: dict[str, Callable[[], Pretokenizer]] = {
    "bytes_gpt4": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX, enforce_char_boundaries=False),
    "bytes_gpt4_cb": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX),
    "bytes_gpt4o": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, enforce_char_boundaries=False),
    "bytes_gpt4o_cb": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX),
    "bytes_nosplit_cb": UTF8PretokenizerConfig(regex_pattern=None),
    "scriptenc": ScriptPretokenizerConfig(enforce_char_boundaries=False),
    "scriptenc_cb": ScriptPretokenizerConfig(),
    "scriptenc_gpt4o_cb": ScriptPretokenizerConfig(regex_pattern=GPT4O_REGEX, script_split=False),
    "scriptenc_nosplit_cb": ScriptPretokenizerConfig(regex_pattern=None, script_split=False),
    "scriptenc2_cb": ScriptPretokenizerConfig(script_config=ScriptEncodingV2, enforce_char_boundaries=True),
    "scriptenc2_gpt4o_cb": ScriptPretokenizerConfig(script_config=ScriptEncodingV2, regex=GPT4O_REGEX, script_split=False),
}


def get_pretokenizer(name: str) -> Pretokenizer:
    """
    Get or initialize a pretokenizer by name.
    :param name: The name of the pretokenizer (must be in the registry).
    :return: An instance of the requested pretokenizer.
    """
    if name not in PRETOKENIZER_REGISTRY:
        raise ValueError(f"Pretokenizer '{name}' is not registered. Available: {list(PRETOKENIZER_REGISTRY.keys())}")
    return config_to_pretokenizer(PRETOKENIZER_REGISTRY[name])

def config_to_pretokenizer(config: PretokenizerConfig) -> Pretokenizer:
    if isinstance(config, UTF8PretokenizerConfig):
        return UTF8Pretokenizer(config)
    elif isinstance(config, ScriptPretokenizerConfig):
        return ScriptPretokenizer(config)
    else:
        raise ValueError(f"Unknown pretokenizer class: {config.__class__.__name__}")

def make_pretokenizer(config) -> Pretokenizer:
    cls = globals()[config["class"]]
    config = cls.model_validate(config["config"])
    return config_to_pretokenizer(config)


def export_pretokenizer(config: PretokenizerConfig) -> dict:
    return {
        "class": config.__class__.__name__,
        "config": config.model_dump(),
    }
