from typing import Callable

from .config import PretokenizerConfig, Pretokenizer, UTF8Pretokenizer, ScriptPretokenizer, UTF8PretokenizerConfig, ScriptPretokenizerConfig
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
    "bytes_gpt4_cb": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX, enforce_char_boundaries=True),
    "bytes_gpt4o": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, enforce_char_boundaries=False),
    "bytes_gpt4o_cb": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, enforce_char_boundaries=True),
    "bytes_nosplit_cb": UTF8PretokenizerConfig(regex_pattern=None, enforce_char_boundaries=True),
    "scriptenc": ScriptPretokenizerConfig(),
    "scriptenc_cb": ScriptPretokenizerConfig(enforce_char_boundaries=True),
    "scriptenc_gpt4o_cb": ScriptPretokenizerConfig(regex_pattern=GPT4O_REGEX, script_split=False, enforce_char_boundaries=True),
    "scriptenc_nosplit_cb": ScriptPretokenizerConfig(regex_pattern=None, enforce_char_boundaries=True),
    "scriptenc2_cb": ScriptPretokenizerConfig(script_config=ScriptEncodingV2, enforce_char_boundaries=True),
    "scriptenc2_gpt4o_cb": ScriptPretokenizerConfig(script_config=ScriptEncodingV2, regex=GPT4O_REGEX, script_split=False, enforce_char_boundaries=True),
}


def get_pretokenizer(name: str) -> Pretokenizer:
    """
    Get or initialize a pretokenizer by name.
    :param name: The name of the pretokenizer (must be in the registry).
    :return: An instance of the requested pretokenizer.
    """
    if name not in PRETOKENIZER_REGISTRY:
        raise ValueError(f"Pretokenizer '{name}' is not registered. Available: {list(PRETOKENIZER_REGISTRY.keys())}")
    return PRETOKENIZER_REGISTRY[name]()


def make_pretokenizer(config) -> PretokenizerConfig:
    cls = globals()[config["class"]]
    return cls.model_validate(config["config"])


def export_pretokenizer(config: PretokenizerConfig) -> dict:
    return {
        "class": config.__class__.__name__,
        "config": config.model_dump(),
    }
