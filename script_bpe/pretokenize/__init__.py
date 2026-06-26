from script_bpe.pretokenize.pretokenizer import (
    Pretokenizer,
    PretokenizerConfig,
    ScriptPretokenizerConfig,
    UTF8PretokenizerConfig,
)
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV1, ScriptEncodingV2, ScriptEncodingV3

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

# Length-limited GPT-4o regex used by the ConvexTok comparison setup:
# caps word match at 32 and other runs at 16 to bound split-tree/pretoken size.
# Copied verbatim from Craig's BPE tokenizer.json pre_tokenizer (the paper's exact regex).
GPT4O_REGEX_LL = (
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]{0,32}[\p{Ll}\p{Lm}\p{Lo}\p{M}]{1,32}(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]{1,32}[\p{Ll}\p{Lm}\p{Lo}\p{M}]{0,32}(?i:'s|'t|'re|'ve|'m|'ll|'d)?"""
    r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]{1,16}[\r\n/]{0,16}|\s{0,15}[\r\n]{1,16}|\s{1,16}(?!\S)|\s{1,16}"""
)


# Registry of available pretokenizers
PRETOKENIZER_REGISTRY: dict[str, PretokenizerConfig] = {
    "bytes_gpt4": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX, enforce_char_boundaries=False),
    "bytes_gpt4_cb": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX),
    "bytes_gpt4o": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, enforce_char_boundaries=False),
    "bytes_gpt4o_cb": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX),
    "bytes_gpt4o_ll_cb": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX_LL),  # length-limited (paper's exact)
    "bytes_gpt4o_cbi": UTF8PretokenizerConfig(regex_pattern=GPT4O_REGEX, enforce_inherited=True),
    "bytes_nosplit_cb": UTF8PretokenizerConfig(regex_pattern=None),
    # Backward-compatible alias names expected by tests
    "bytes_nosplit_cbi": UTF8PretokenizerConfig(regex_pattern=None, enforce_inherited=True),
    "bytes_gpt4_cbi": UTF8PretokenizerConfig(regex_pattern=GPT4_REGEX, enforce_inherited=True),
    "scriptenc": ScriptPretokenizerConfig(enforce_char_boundaries=False),
    "scriptenc_cb": ScriptPretokenizerConfig(),
    "scriptenc_cbi": ScriptPretokenizerConfig(enforce_inherited=True),
    "scriptenc_gpt4o": ScriptPretokenizerConfig(
        regex_pattern=GPT4O_REGEX, script_split=False, enforce_char_boundaries=False
    ),
    "scriptenc_gpt4o_cb": ScriptPretokenizerConfig(regex_pattern=GPT4O_REGEX, script_split=False),
    "scriptenc_gpt4o_cbi": ScriptPretokenizerConfig(
        regex_pattern=GPT4O_REGEX, script_split=False, enforce_inherited=True
    ),
    "scriptenc_nosplit_cb": ScriptPretokenizerConfig(regex_pattern=None, script_split=False),
    "scriptenc2_cb": ScriptPretokenizerConfig(script_config=ScriptEncodingV2, enforce_char_boundaries=True),
    "scriptenc2_cbi": ScriptPretokenizerConfig(script_config=ScriptEncodingV2, enforce_inherited=True),
    "scriptenc3_cb": ScriptPretokenizerConfig(script_config=ScriptEncodingV3, enforce_char_boundaries=True),
    "scriptenc3_cbi": ScriptPretokenizerConfig(script_config=ScriptEncodingV3, enforce_inherited=True),
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
