from script_bpe.encoding.script_util import ScriptEncodingBase
import functools


class ScriptEncodingV1(ScriptEncodingBase):
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
        if char_info['char'] in cls.SCRIPT_CAT_OVERRIDE:
            return cls.SCRIPT_CAT_OVERRIDE[char_info['char']]

        category, script = char_info["category"], char_info["script"]
        supercat = category[0]
        if supercat in {"P", "S"}:
            supercat = "PS"  # Punctuation/Symbol
        if supercat in {"L", "M"}:
            supercat = "LM"  # Letter/Non-spacing Mark (like accept modifiers)
        return script, supercat
