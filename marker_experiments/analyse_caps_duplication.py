#!/usr/bin/env python3
r"""Count the case-variant pairs a vocabulary spends slots on, per scheme.

The caps schemes exist to stop `the`/`The`/`THE` from taking three vocabulary entries.
They differ only in where the case code sits relative to the boundary markers:

    bnd_wpd_caps      <|><^>the<|>     code inside  -- the span including the code is a
                                       distinct atomic sequence, so a learned token
                                       covering it cannot be the one covering <|>the<|>
    bnd_wpd_extcaps   <^><|>the<|>     code outside -- <|>the<|> occurs identically in
                                       both, so one entry can serve the cased and
                                       uncased forms

Whether that difference shows up in a trained vocabulary is an empirical question, and
this answers it by reading the vocabularies rather than the design. A token's surface is
what it actually contributes to output: its core decoded through the pretokenizer, with
the case code applied (shift -> capitalize, caps -> upper). Two entries whose surfaces
differ only by case are the duplication being counted.

Leading spaces are stripped before comparing, so the `plain` baseline is counted on case
alone -- its ' the'/'the' duplication is a different axis and is not what this measures.
Surfaces shorter than `min_length` characters are ignored: two-letter case pairs are
dominated by abbreviations and inflections where a shared entry would not have helped.

    uv run python marker_experiments/analyse_caps_duplication.py
    uv run python marker_experiments/analyse_caps_duplication.py --lang en --trainer mingram
"""

import json
import os

import cyclopts

from script_bpe.tokenizers.load import load_tokenizer

# Registers the boundary pretokenizers, without which the tokenizers do not load.
import marker_experiments.downstream.boundary_tokenizer  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZERS = os.path.join(HERE, "downstream", "tokenizers")
GENERATED = os.path.join(HERE, "paper", "generated")

ARMS = ["plain", "bnd_wpd", "bnd_wpd_caps", "bnd_wpd_extcaps"]

app = cyclopts.App()


def surfaces(tokenizer):
    """Every learned token's output surface, and whether it carried a case code.

    Atomic tokens are excluded: they are the alphabet every arm starts from, not a choice
    the trainer made. A core that does not decode on its own -- a fragment that only means
    something in context -- is skipped rather than guessed at.
    """
    pt = tokenizer.pretokenizer
    marker = getattr(pt, "marker_token_id", None)
    shift, caps = getattr(pt, "shift_token_id", None), getattr(pt, "caps_token_id", None)

    out = []
    for token in tokenizer.tokens.values():
        ids = list(token.atomic_tokens)
        if len(ids) == 1:
            continue
        code = "shift" if shift in ids else ("caps" if caps in ids else None)
        core = [i for i in ids if i not in (marker, shift, caps)]
        if not core:
            continue
        text = pt.try_decode_strict(core)
        if text is None:
            continue
        text = text[1:] if text.startswith(" ") else text   # case axis only
        if code == "shift":
            text = text[:1].upper() + text[1:]
        elif code == "caps":
            text = text.upper()
        out.append((text, code))
    return out


def count_pairs(tokenizer, min_length):
    """Case-variant pairs among the learned surfaces.

    A pair is counted from the lowercase side, so `the` with both `The` and `THE` present
    contributes to both counts and is one duplicated word, not two.
    """
    seen = {text for text, _ in surfaces(tokenizer)}
    lower = {t for t in seen if t.islower() and t.isalpha() and len(t) >= min_length}
    title = sorted(t for t in lower if t[:1].upper() + t[1:] in seen)
    allcaps = sorted(t for t in lower if t.upper() in seen)
    coded = sum(1 for _, code in surfaces(tokenizer) if code)
    return {
        "learned_surfaces": len(seen),
        "lowercase_words": len(lower),
        "titlecase_duplicates": len(title),
        "allcaps_duplicates": len(allcaps),
        "coded_tokens": coded,
        "examples": title[:5],
    }


@app.default
def main(
    lang: str = "de",
    trainer: str = "bpe",
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    arms: str = ",".join(ARMS),
    min_length: int = 3,
    out: str | None = os.path.join(GENERATED, "caps_duplication.json"),
) -> None:
    """Report case duplication for each arm of one language.

    Args:
        lang: Language code.
        trainer: `bpe` or `mingram`.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        arms: Comma-separated arms to compare.
        min_length: Shortest surface to count, in characters.
        out: JSON to write, or empty to only print.
    """
    rows = {}
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        path = os.path.join(TOKENIZERS, f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}.json.gz")
        if not os.path.exists(path):
            print(f"[skip] {arm}: not trained yet")
            continue
        rows[arm] = count_pairs(load_tokenizer(path), min_length)

    if not rows:
        raise SystemExit("no arms available")
    print(f"\n{lang}/{trainer}, surfaces of >= {min_length} characters\n")
    print(f"{'arm':<18}{'learned':>9}{'lowercase':>11}{'Titlecase':>11}{'ALLCAPS':>9}{'coded':>8}")
    for arm, r in rows.items():
        print(f"{arm:<18}{r['learned_surfaces']:>9,}{r['lowercase_words']:>11,}"
              f"{r['titlecase_duplicates']:>11,}{r['allcaps_duplicates']:>9,}{r['coded_tokens']:>8,}")
    for arm, r in rows.items():
        if r["examples"]:
            print(f"\n{arm} duplicated e.g.: " + ", ".join(f"{t}/{t.capitalize()}" for t in r["examples"]))

    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"lang": lang, "trainer": trainer, "min_length": min_length, "arms": rows}, f,
                      indent=2, sort_keys=True)
        print(f"\n[json] {out}")


if __name__ == "__main__":
    app()
