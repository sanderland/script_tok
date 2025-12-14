import functools
import os
import unicodedata
from collections import Counter, defaultdict
from typing import ClassVar, Literal

from frozendict import frozendict
from pydantic import BaseModel

SCRIPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unicode_scripts.txt")
END_CODEPOINT = 0xE0FFF  # full cover of non private use code points, excluding surrogates


def char_name(c: str):
    return unicodedata.name(c, "<UNKNOWN>")


@functools.cache
def unicode_script_map(filename=SCRIPTS_PATH) -> dict[str, frozendict[str, object]]:
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


# Use deterministic lists for config-level data to avoid nondeterministic set ordering
ALL = "✭"
DEFAULT_SCRIPTS_LM_WITH_SPACES = [
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
    "Georgian",  # Near space 14.1% of time, overall count 277,463
]
DEFAULT_SCRIPT_WITH_SPACES = [(s, "LM") for s in DEFAULT_SCRIPTS_LM_WITH_SPACES]
DEFAULT_SCRIPT_HIGH_RESOURCE_NO_SPACES = [
    "Common",
    "Han",
    "Hiragana",
    "Katakana",
    "Thai",
    "Myanmar",
    "Khmer",
    "Lao",
]
ARABIC_SCRIPT_CAT_OVERRIDE = {
    "\u0640": ("Arabic", "LM"),  # ـــمــر (used in Arabic script shaping)
}
V1_SCRIPT_CAT_OVERRIDE = ARABIC_SCRIPT_CAT_OVERRIDE | {
    "\n": ("Common", "Z"),  # Newline – whitespace
    "\t": ("Common", "Z"),  # Tab – whitespace
    "\u30fc": ("Inherited", "LM"),  # カー (カ + ー) Katakana-Hiragana Prolonged Sound Mark in Japanese
    "\uff70": ("Inherited", "LM"),  # ﾊﾟｰﾃｨｰ (halfwidth)
}
V2_SCRIPT_CAT_OVERRIDE = ARABIC_SCRIPT_CAT_OVERRIDE | {
    "\u30fc": ("Inherited", ALL),  # カー (カ + ー) Katakana-Hiragana Prolonged Sound Mark in Japanese
    "\uff70": ("Inherited", ALL),  # ﾊﾟｰﾃｨｰ (halfwidth)
}


class ScriptBlock(BaseModel):
    script_id: int
    script: str
    category: str
    sub_block_id: int
    combines_with_spaces: bool
    chars: str


class ScriptConfig(BaseModel):
    ALL: ClassVar[str] = ALL
    ASCII_SCRIPT: ClassVar[str] = "ASCII"
    blocks: list[ScriptBlock] = []
    supercategory_type: Literal["V1", "V2", "V3"]
    largest_block: tuple[str, str] = ("Latin", "LM")
    merge_hiragana_with_han: bool = True
    script_cat_with_spaces: list[tuple[str, str]] = DEFAULT_SCRIPT_WITH_SPACES + [("Common", "PS")]
    script_cat_override: dict[str, tuple[str, str]] = ARABIC_SCRIPT_CAT_OVERRIDE
    # v2 only: avoid sets in config; use list for determinism and convert to set internally
    higher_resource_scripts: list[str] = DEFAULT_SCRIPTS_LM_WITH_SPACES + DEFAULT_SCRIPT_HIGH_RESOURCE_NO_SPACES

    def model_post_init(self, __context):
        """builds blocks"""
        # for fast membership checks, while keeping config deterministic
        self._higher_resource_scripts_set = set(self.higher_resource_scripts)
        self._script_cat_with_spaces_set = set(self.script_cat_with_spaces)
        if self.blocks:
            return  # we trust blocks when they are provided
        chars_by_sc = defaultdict(list)
        num_chars_by_script = Counter()
        for char_info in unicode_script_map().values():
            if char_info["char"] in self.script_cat_override:
                script_cat = self.script_cat_override[char_info["char"]]
            elif self.supercategory_type == "V1":
                script_cat = self.script_category_v1(char_info)
            elif self.supercategory_type == "V2":
                script_cat = self.script_category_v2(char_info)
            elif self.supercategory_type == "V3":
                script_cat = self.script_category_v3(char_info)
            chars_by_sc[script_cat].append(char_info["char"])
            num_chars_by_script[char_info["script"]] += 1

        assert self.largest_block in chars_by_sc, f"{self.largest_block} not found. Blocks are: {chars_by_sc.keys()}"
        num_index_tokens = len(chars_by_sc[self.largest_block])
        self.blocks = []
        for sc, cps in sorted(chars_by_sc.items(), key=lambda kv: (num_chars_by_script[kv[0][0]], kv[1]), reverse=True):
            sid = len(self.blocks) + 1
            script, supercat = sc
            combines_with_spaces = (script, supercat) in self._script_cat_with_spaces_set
            for sub_block, start in enumerate(range(0, len(cps), num_index_tokens)):
                self.blocks.append(
                    ScriptBlock(
                        script_id=sid,
                        script=script,
                        category=supercat,
                        sub_block_id=sub_block,
                        combines_with_spaces=combines_with_spaces,
                        chars="".join(cps[start : start + num_index_tokens]),
                    )
                )

        if self.merge_hiragana_with_han:
            self._merge_hiragana_with_han()

    def script_category_v1(self, char_info) -> tuple[str, str]:
        category, script = char_info["category"], char_info["script"]
        supercat = category[0]
        if supercat in {"P", "S"}:
            supercat = "PS"  # Punctuation/Symbol
        if supercat in {"L", "M"}:
            supercat = "LM"  # Letter/Non-spacing Mark (like accept modifiers)
        return script, supercat

    def script_category_v2(self, char_info) -> tuple[str, str]:
        category, script = char_info["category"], char_info["script"]
        supercat = category[0]

        # low resource scripts are (script, *) with no blocks for category
        if script not in self._higher_resource_scripts_set:
            supercat = self.ALL
            return script, supercat

        # merge categories into supercategories
        if supercat in {"L", "M"}:
            supercat = "LM"  # Letter/Mark
        elif supercat == "Z" or category == "Cc":
            supercat = "ZC"  # whitespace/control, which includes \n,\t, etc
        elif supercat in {"P", "S"} or category == "Cf":
            supercat = "PSF"  # Punctuation/Symbol/Formatting
            if ord(char_info["char"]) < 128:  # ascii punctuation is its own script, which allows for leading spaces
                script = self.ASCII_SCRIPT

        # for all non-letters we ignore the script
        if supercat != "LM" and script != self.ASCII_SCRIPT:
            script = self.ALL  # for non-letters, ignore script

        return script, supercat

    def script_category_v3(self, char_info) -> tuple[str, str]:
        category, script = char_info["category"], char_info["script"]
        supercat = category[0]

        # low resource scripts are (script, *) with no blocks for category
        if script not in self._higher_resource_scripts_set:
            supercat = self.ALL
            return script, supercat

        # merge categories into supercategories
        if supercat in {"L", "M"}:
            supercat = "LM"  # Letter/Mark
        elif supercat == "Z" or category == "Cc":
            supercat = "ZC"  # whitespace/control, which includes \n,\t, etc
        elif category == "So":  # emoji etc
            supercat = "So"
        elif supercat in {"P", "S"} or category == "Cf":
            supercat = "PSF"  # Punctuation/Symbol/Formatting

        # for all non-letters we ignore the script
        if supercat != "LM":
            script = self.ALL  # for non-letters, ignore script

        return script, supercat

    def _merge_hiragana_with_han(self) -> list[ScriptBlock]:  # recode Hiragana to Han for pretokenizing
        han_lm_script_id = next(b.script_id for b in self.blocks if b.script == "Han" and b.category == "LM")
        for b in self.blocks:
            if b.script == "Hiragana" and b.category == "LM":
                b.script_id = han_lm_script_id
        return self.blocks


ScriptEncodingV1 = ScriptConfig(supercategory_type="V1", script_cat_override=V1_SCRIPT_CAT_OVERRIDE)
ScriptEncodingV2 = ScriptConfig(
    supercategory_type="V2",
    script_cat_override=V2_SCRIPT_CAT_OVERRIDE,
    script_cat_with_spaces=DEFAULT_SCRIPT_WITH_SPACES + [(ScriptConfig.ASCII_SCRIPT, "PSF")],
)
ScriptEncodingV3 = ScriptConfig(
    supercategory_type="V3",
    script_cat_override=V2_SCRIPT_CAT_OVERRIDE,
    script_cat_with_spaces=DEFAULT_SCRIPT_WITH_SPACES + [(ALL, "PSF")],
)
