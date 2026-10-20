"""Regex pretokenizer equivalent of SentencePiece's trainer-side piece constraints.

SentencePiece enforces most of its splitting in TrainerInterface::IsValidSentencePiece, which rejects
candidate pieces rather than cutting words. A pretokenizer chunk is the maximal span over which those
checks allow a piece, so BPE on these chunks sees the same candidate space as SentencePiece BPE.
"""

import functools

from script_bpe.pretokenize.scriptencoding import SCRIPTS_PATH

# SentencePiece merges Hiragana, Katakana and the prolonged sound mark (Common) into Han.
_HAN = r"\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}\u30FC"
# is_unicode_decimal_number in trainer_interface.cc: ASCII and fullwidth digits only.
_DIGITS = r"0-9\uFF10-\uFF19"
_INHERITED = r"\p{sc=Inherited}"


@functools.cache
def _script_classes() -> list[str]:
    """Class contents per SentencePiece script group, excluding Common and Inherited.

    Codepoints missing from SentencePiece's script table map to Common, so Common is the
    complement of these groups (and of U+0020) and needs no class of its own.
    """
    names = set()
    with open(SCRIPTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                names.add(line.split(";")[1].split("#")[0].strip())
    names -= {"Common", "Inherited", "Han", "Hiragana", "Katakana"}
    return [_HAN] + [f"\\p{{sc={name}}}" for name in sorted(names)]


def sentencepiece_regex(
    split_by_whitespace: bool = True,
    split_by_unicode_script: bool = True,
    split_by_number: bool = True,
    split_digits: bool = False,
    allow_whitespace_only_pieces: bool = False,
) -> str:
    """Regex whose findall chunks are the spans SentencePiece's trainer lets a piece cover.

    Defaults match spm_train. The input is unnormalized text, so U+0020 plays the role of U+2581.
    treat_whitespace_as_suffix is not supported.

    Approximations: with split_by_number=False a digit run joins the script run around it, whereas
    SentencePiece also lets one digit bridge two different scripts ("a1α"). Inherited-script
    characters (combining marks) take the script of what precedes them, as in SentencePiece.
    max_sentencepiece_length caps candidate pieces and has no chunk equivalent.
    """
    excluded = " " + (_DIGITS if split_digits else "")
    if split_by_unicode_script:
        # Digits are Common, so the negated Common class holds them; with split_by_number=False
        # they are wildcards and join every script.
        wild = _DIGITS if not split_by_number and not split_digits else ""
        scripts = _script_classes()
        classes = [f"[{c}{_INHERITED}{wild}]" for c in scripts]
        classes.append(f"[^{''.join(scripts)}{excluded}]")
    else:
        classes = [f"[^{excluded}]"]

    lead = f"[{_INHERITED}]*"
    if split_by_whitespace:
        # A piece is an optional single leading space plus one script run.
        runs = [f" ?{lead}{c}+" for c in classes]
    else:
        # Spaces may sit anywhere except at the end; the script check skips them.
        runs = [f" *{lead}{c}+(?: +{c}+)*" for c in classes]
    alts = list(runs)
    if split_digits:
        alts.append(f"[{_DIGITS}]")
    alts.append(f"[{_INHERITED}]+")
    if allow_whitespace_only_pieces:
        if split_by_whitespace:
            # Leave the last space of a run for the word after it, unless that word can't take one.
            prefixable = f"[^ {_DIGITS}]" if split_digits else "[^ ]"
            alts.append(f" +(?= {prefixable})")
        alts.append(" +")
    else:
        alts.append(" ")
    return "|".join(alts)
