import functools
import os
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from frozendict import frozendict

SCRIPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unicode_scripts.txt")
END_CODEPOINT = 0xE0FFF  # full cover of non private use code points, excluding surrogates


def char_name(c: str):
    return unicodedata.name(c, "<UNKNOWN>")


@functools.cache
def unicode_script_map(filename=SCRIPTS_PATH) -> dict[str, dict[str, str]]:
    """
    Load Unicode script and category data from a file.

    Returns:
        A list of dictionaries with 'cp', 'char', 'script' and 'category' keys
    """
    char_infos = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse 0000..001F    ; Common # Cc  [32] <control-0000>..<control-001F>
            range_str, semicol, script, hash, category, *_ = line.split()
            assert semicol == ";" and hash == "#", f"Unexpected format in line: {line}"
            # Handle single codepoint or range
            if ".." in range_str:
                start_str, end_str = range_str.split("..")
                start, end = int(start_str, 16), int(end_str, 16)
            else:
                start = end = int(range_str, 16)
            # Add each codepoint in the range to the result dictionary
            for cp in range(start, end + 1):
                char_infos[chr(cp)] = frozendict(
                    cp=cp,
                    char=chr(cp),
                    script=script,
                    category=category,  # file is typically newer than python's unicodedata
                )
    return char_infos


class ScriptEncodingBase:
    FIRST_TOKEN_ID = 1  # leave 0 for padding
    ALL = "✭"  # symbol used to represent "any" script or category

    SCRIPT_LM_WITH_SPACES = {
        "Latin",  # Near space 18.0% of time, overall count 296,227,775,159
        "Arabic",  # Near space 18.6% of time, overall count 116,931,857,999
        "Devanagari",  # Near space 23.0% of time, overall count 3,391,578,438
        "Hangul",  # Near space 28.8% of time, overall count 2,456,037,316
        "Ethiopic",  # Near space 22.3% of time, overall count 493,107,008
        "Cyrillic",  # Near space 13.6% of time, overall count 119,738,736
        "Greek",  # Near space 17.2% of time, overall count 13,040,710
        "Hebrew",  # Near space 17.0% of time, overall count 6,081,243
        "Bengali",  # Near space 16.4% of time, overall count 3,630,267
        "Syriac",  # Near space 12.6% of time, overall count 2,669,016
        "Oriya",  # Near space 16.6% of time, overall count 1,147,764
        "Tamil",  # Near space 11.9% of time, overall count 1,084,075
        "Telugu",  # Near space 13.6% of time, overall count 694,309
        "Gurmukhi",  # Near space 22.8% of time, overall count 394,438
        "Gujarati",  # Near space 18.3% of time, overall count 388,983
        "Sinhala",  # Near space 17.5% of time, overall count 369,417
        "Malayalam",  # Near space 10.7% of time, overall count 339,796
        "Armenian",  # Near space 14.3% of time, overall count 338,586
        "Kannada",  # Near space 13.3% of time, overall count 326,104
        "Georgian",  # Near space 14.1% of time, overall count 277,463}
    }
    SCRIPT_HIGH_RESOURCE_NO_SPACES = {
        "Common",
        "Han",
        "Hiragana",
        "Katakana",
        "Thai",
        "Myanmar",
        "Khmer",
        "Lao",
    }
    # Pretokenize allows a single leading space for these. PS is only for V1 and ASCII/PSF is only for V2
    SCRIPT_CAT_WITH_SPACE = [(s, "LM") for s in SCRIPT_LM_WITH_SPACES]
    # all blocks larger than this will be split
    LARGEST_BLOCK_SCRIPT_CAT = ("Latin", "LM")
    MERGE_HIRAGANA_WITH_HAN = True

    def __init__(self):
        self.blocks, self.config = self.create_blocks()
        self.config["version"] = self.__class__.__name__
        self.config["num_blocks"] = len(self.blocks)
        largest_block = max(self.blocks, key=lambda b: len(b["chars"]))
        self.config["num_index_tokens"] = len(largest_block["chars"])

    @classmethod
    def script_category(cls, char_info) -> tuple[str, str]:
        raise NotImplementedError

    @classmethod
    @functools.cache
    def _is_higher_resource_script(cls, script: str) -> bool:
        return script in cls.SCRIPT_HIGH_RESOURCE_NO_SPACES or script in cls.SCRIPT_LM_WITH_SPACES

    def _merge_hiragana_with_han(self, blocks: list[dict]) -> list[dict]:
        # recode Hiragana to Han for pretokenizing
        han_lm_script_id = next(b["script_id"] for b in blocks if b["script"] == "Han" and b["category"] == "LM")
        for b in blocks:
            if b["script"] == "Hiragana" and b["category"] == "LM":
                b["script_id"] = han_lm_script_id
        return blocks

    def create_blocks(self) -> tuple[list[dict], dict]:
        chars_by_sc = defaultdict(list)
        num_chars_by_script = Counter()
        for char_info in unicode_script_map().values():
            chars_by_sc[self.script_category(char_info)].append(char_info["char"])
            num_chars_by_script[char_info["script"]] += 1

        assert self.LARGEST_BLOCK_SCRIPT_CAT in chars_by_sc, (
            f"{self.LARGEST_BLOCK_SCRIPT_CAT} not found. Blocks are: {chars_by_sc.keys()}"
        )
        num_index_tokens = len(chars_by_sc[self.LARGEST_BLOCK_SCRIPT_CAT])
        blocks = []
        for sc, cps in sorted(chars_by_sc.items(), key=lambda kv: (num_chars_by_script[kv[0][0]], kv[1]), reverse=True):
            sid = len(blocks) + num_index_tokens
            script, supercat = sc
            combines_with_spaces = (script, supercat) in self.SCRIPT_CAT_WITH_SPACE
            for sub_block, start in enumerate(range(0, len(cps), num_index_tokens)):
                blocks.append(
                    dict(
                        script_id=sid,
                        script=script,
                        category=supercat,
                        sub_block_id=sub_block,
                        combines_with_spaces=combines_with_spaces,
                        chars="".join(cps[start : start + num_index_tokens]),
                    )
                )

        if self.MERGE_HIRAGANA_WITH_HAN:
            blocks = self._merge_hiragana_with_han(blocks)
        return blocks, {}

    def export_config(self) -> dict[str, Any]:
        return dict(
            **self.config,
            blocks=self.blocks,
        )


class ScriptEncodingV1(ScriptEncodingBase):
    SCRIPT_CAT_WITH_SPACE = ScriptEncodingBase.SCRIPT_CAT_WITH_SPACE + [("Common", "PS")]
    SCRIPT_CAT_OVERRIDE = {
        "\n": ("Common", "Z"),  # Newline – whitespace
        "\t": ("Common", "Z"),  # Tab – whitespace
        "\u30fc": ("Inherited", "LM"),  # カー (カ + ー) Katakana-Hiragana Prolonged Sound Mark in Japanese
        "\uff70": ("Inherited", "LM"),  # ﾊﾟｰﾃｨｰ (halfwidth)
        "\u0640": ("Arabic", "LM"),  # ـــمــر (used in Arabic script shaping)
    }

    @classmethod
    @functools.cache
    def script_category(cls, char_info) -> tuple[str, str]:
        if char_info["char"] in cls.SCRIPT_CAT_OVERRIDE:
            return cls.SCRIPT_CAT_OVERRIDE[char_info["char"]]

        category, script = char_info["category"], char_info["script"]
        supercat = category[0]
        if supercat in {"P", "S"}:
            supercat = "PS"  # Punctuation/Symbol
        if supercat in {"L", "M"}:
            supercat = "LM"  # Letter/Non-spacing Mark (like accept modifiers)
        return script, supercat


class ScriptEncodingV2(ScriptEncodingBase):
    ASCII_SCRIPT = "ASCII"
    SCRIPT_CAT_WITH_SPACE = ScriptEncodingBase.SCRIPT_CAT_WITH_SPACE + [(ASCII_SCRIPT, "PSF")]
    SCRIPT_CAT_OVERRIDE = {
        "\u30fc": ("Inherited", "LM"),  # カー (カ + ー) Katakana-Hiragana Prolonged Sound Mark in Japanese
        "\uff70": ("Inherited", "LM"),  # ﾊﾟｰﾃｨｰ (halfwidth)
        "\u0640": ("Arabic", "LM"),  # ـــمــر (used in Arabic script shaping)
    }

    @classmethod
    @functools.cache
    def script_category(cls, char_info) -> tuple[str, str]:
        if char_info["char"] in cls.SCRIPT_CAT_OVERRIDE:
            return cls.SCRIPT_CAT_OVERRIDE[char_info["char"]]

        category, script = char_info["category"], char_info["script"]
        supercat = category[0]

        # low resource scripts are (script, *) with no blocks for category
        if not cls._is_higher_resource_script(script):
            supercat = cls.ALL  # for low resource script, ignore category
            return script, supercat

        # merge categories into supercategories
        if supercat in {"L", "M"}:
            supercat = "LM"  # Letter/Mark
        elif supercat == "Z" or category == "Cc":
            supercat = "ZC"  # whitespace/control, which includes \n,\t, etc
        elif supercat in {"P", "S"} or category == "Cf":
            supercat = "PSF"  # Punctuation/Symbol/Formatting
            if ord(char_info["char"]) < 128:  # ascii punctuation is its own script, which allows for leading spaces
                script = cls.ASCII_SCRIPT

        # for all non-letters we ignore the script
        if supercat != "LM" and script != cls.ASCII_SCRIPT:
            script = cls.ALL  # for non-letters, ignore script

        return script, supercat
