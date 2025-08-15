from typing import Any, Callable, Type

from script_bpe.encoding import ScriptEncodingV1

from .base import BasePretokenizer
from .regex import RegexBytePretokenizer
from .scriptencoding import (
    ScriptEncodingPretokenizer,
    ScriptEncodingPretokenizerRegexSplitting,
)

DefaultScriptEncodingV1 = ScriptEncodingV1().export_config()


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
NO_SPLIT_REGEX = r"\A.*\Z"


# Registry of available pretokenizers
PRETOKENIZER_REGISTRY: dict[str, Callable[[], BasePretokenizer]] = {
    "bytes_gpt4": lambda: RegexBytePretokenizer({"regex": GPT4_REGEX, "enforce_char_boundaries": False}),
    "bytes_gpt4_cb": lambda: RegexBytePretokenizer({"regex": GPT4_REGEX, "enforce_char_boundaries": True}),
    "bytes_gpt4o": lambda: RegexBytePretokenizer({"regex": GPT4O_REGEX, "enforce_char_boundaries": False}),
    "bytes_gpt4o_cb": lambda: RegexBytePretokenizer({"regex": GPT4O_REGEX, "enforce_char_boundaries": True}),
    "bytes_nosplit_cb": lambda: RegexBytePretokenizer({"regex": NO_SPLIT_REGEX, "enforce_char_boundaries": True}),
    "scriptenc": lambda: ScriptEncodingPretokenizer({**DefaultScriptEncodingV1, "enforce_char_boundaries": False}),
    "scriptenc_cb": lambda: ScriptEncodingPretokenizer({**DefaultScriptEncodingV1, "enforce_char_boundaries": True}),
    "scriptenc_gpt4o_cb": lambda: ScriptEncodingPretokenizerRegexSplitting(
        {**DefaultScriptEncodingV1, "regex": GPT4O_REGEX, "enforce_char_boundaries": True}
    ),
    "scriptenc_gpt4o": lambda: ScriptEncodingPretokenizerRegexSplitting(
        {**DefaultScriptEncodingV1, "regex": GPT4O_REGEX, "enforce_char_boundaries": False}
    ),
    "scriptenc_nosplit_cb": lambda: ScriptEncodingPretokenizerRegexSplitting(
        {**DefaultScriptEncodingV1, "regex": NO_SPLIT_REGEX, "enforce_char_boundaries": True}
    ),
}


def get_pretokenizer(name: str) -> BasePretokenizer:
    """
    Get or initialize a pretokenizer by name.
    :param name: The name of the pretokenizer (must be in the registry).
    :return: An instance of the requested pretokenizer.
    """
    if name not in PRETOKENIZER_REGISTRY:
        raise ValueError(f"Pretokenizer '{name}' is not registered. Available: {list(PRETOKENIZER_REGISTRY.keys())}")
    return PRETOKENIZER_REGISTRY[name]()


def make_pretokenizer(data: dict):
    cls = globals()[data["class"]]
    config = data["config"]
    return cls(config)


def export_pretokenizer(pretok: BasePretokenizer) -> dict:
    return {
        "class": pretok.__class__.__name__,
        "config": pretok.config,
    }
