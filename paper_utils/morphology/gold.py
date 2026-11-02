"""Surface morph boundaries from existing gold, as {word: sorted internal boundary offsets}.

SIGMORPHON 2022 word-level gold is canonical (`un @@hap @@y @@ness` for `unhappiness`), so
its morphs do not always spell the word. Boundaries are carried over by aligning the
concatenated canonical morphs to the surface form; characters the surface inserts go to the
morph on their left, which is where English consonant doubling and Russian stem extensions
belong. A row whose morphs concatenate to the word needs no alignment and is marked exact.
"""

import difflib
import os

SIGMORPHON = os.path.expanduser("~/Desktop/mingram-hack/morphdata/sig22/data")
CELEX_EN = os.path.join(os.path.dirname(__file__), "../../eval/morph-tok-eval/data/morpho/eng-unimorph2uniseg_CELEX.tsv")


def offsets(morphs: list[str]) -> list[int]:
    out, at = [], 0
    for m in morphs[:-1]:
        at += len(m)
        out.append(at)
    return out


def project(canonical: str, surface: str, bounds: list[int]) -> list[int]:
    """Map boundary offsets in `canonical` onto `surface` through a character alignment."""
    ops = difflib.SequenceMatcher(None, canonical, surface, autojunk=False).get_opcodes()
    out = set()
    for b in bounds:
        pos = None
        for tag, i1, i2, j1, j2 in ops:
            if tag == "insert":
                if i1 == b and pos is not None:
                    pos = j2              # surface-only characters at the boundary join the left morph
                continue
            if i1 < b <= i2:
                pos = j1 + (b - i1) if tag == "equal" else j1 + round((b - i1) * (j2 - j1) / (i2 - i1))
            elif i1 >= b and pos is not None:
                break
        if pos is not None and 0 < pos < len(surface):
            out.add(pos)
    return sorted(out)


def sigmorphon(lang: str) -> dict[str, tuple[list[int], bool]]:
    """{lowercased word: (boundaries, exact)} over train, dev and test gold."""
    gold = {}
    for split in ("train", "dev", "test.gold"):
        with open(os.path.join(SIGMORPHON, f"{lang}.word.{split}.tsv"), encoding="utf-8") as f:
            for line in f:
                word, seg, *_ = line.rstrip("\n").split("\t")
                morphs = seg.replace("@@", "").split(" ")
                bounds = offsets(morphs)
                joined = "".join(morphs)
                exact = joined == word
                gold[word.lower()] = (bounds if exact else project(joined.lower(), word.lower(), bounds), exact)
    return gold


def celex_en() -> dict[str, list[int]]:
    gold = {}
    with open(CELEX_EN, encoding="utf-8") as f:
        for line in f:
            word, _tag, seg = line.rstrip("\n").split("\t")
            morphs = seg.split("|")
            if "".join(morphs) == word:
                gold[word.lower()] = offsets(morphs)
    return gold


def show(word: str, bounds: list[int]) -> str:
    cuts = [0, *bounds, len(word)]
    return "|".join(word[a:b] for a, b in zip(cuts, cuts[1:]))
