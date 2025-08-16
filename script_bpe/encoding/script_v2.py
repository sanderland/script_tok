import functools
from script_bpe.encoding.script_util import ScriptEncodingBase

ALL = "⭐" # symbol used to represent "any" script or category

class ScriptEncodingV2(ScriptEncodingBase):
    LARGEST_BLOCK_SCRIPT_CAT = ("Latin", "LM")
    SCRIPT_CAT_OVERRIDE = {
        "\u30fc": ("Inherited", ALL),  # カー (カ + ー) Katakana-Hiragana Prolonged Sound Mark in Japanese
        "\uff70": ("Inherited", ALL),  # ﾊﾟｰﾃｨｰ (halfwidth)
        "\u0640": ("Arabic", "LM"),  # ـــمــر (used in Arabic script shaping)
    }


    @classmethod
    @functools.cache
    def script_category(cls, char_info) -> tuple[str, str]:
        if char_info['char'] in cls.SCRIPT_CAT_OVERRIDE:
            return cls.SCRIPT_CAT_OVERRIDE[char_info['char']]

        category, script = char_info["category"], char_info["script"]
        supercat = category[0]
        if supercat in {"L", "M"}:
            supercat  = "LM"  # Letter/Non-spacing Mark (like accept modifiers)
        else:
            script = ALL  
        if supercat in {"P", "S"} or category=="Cf":
            supercat = "PSF"  # Punctuation/Symbol
        if supercat == "Z" or category == "Cc":
            supercat = "ZC" # whitespace/control
        # TODO: "other" script?
        return script, supercat
