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


def word_cost(tokenizer, word, carrier=("Das ", " ist gut.")):
    """Tokens the word costs inside a carrier phrase, space included.

    Measured as a difference rather than by encoding the word alone, because the boundary
    schemes elide the space between adjacent spans: a bare word is not the input any arm
    is built for, and the leading-space entry `plain` would use never gets hit.
    """
    with_word = len(tokenizer.encode(carrier[0] + word + carrier[1]))
    without = len(tokenizer.encode(carrier[0].rstrip() + carrier[1]))
    return with_word - without


def allcaps_pairs(tokenizer, min_length):
    seen = {text for text, _ in surfaces(tokenizer)}
    return {t for t in seen
            if t.islower() and t.isalpha() and len(t) >= min_length and t.upper() in seen}


@app.command
def collapsed(
    lang: str = "de",
    trainer: str = "bpe",
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    min_length: int = 3,
    baseline: str = "plain",
    variants: str = "bnd_wpd_caps,bnd_wpd_extcaps",
    reference: str = "bnd_wpd",
    examples: int = 8,
) -> None:
    """What the case codes collapsed, and what those words cost once collapsed.

    The baseline spends two entries on a word it also has in ALLCAPS. Where a caps variant
    no longer does, that slot came back -- but the ALLCAPS form still has to be spelled,
    and this reports what it now costs in tokens. Also lists the longest surviving ALLCAPS
    surface per arm, which is where a scheme is still paying for a whole word.

    Args:
        lang: Language code.
        trainer: `bpe` or `mingram`.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        min_length: Shortest surface to count, in characters.
        baseline: Arm whose ALLCAPS pairs are the starting set.
        variants: Comma-separated arms that may have collapsed them.
        reference: Arm costed alongside but excluded from the collapse set -- the same
            boundary scheme without caps codes, so the codes' share of any cost change is
            separable from the markers'.
        examples: How many longest-ALLCAPS surfaces to show per arm.
    """
    variant_arms = [v.strip() for v in variants.split(",") if v.strip()]
    arms = [baseline, *([reference] if reference else []), *variant_arms]
    loaded = {}
    for arm in arms:
        path = os.path.join(TOKENIZERS, f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}.json.gz")
        if not os.path.exists(path):
            print(f"[skip] {arm}: not trained yet")
            continue
        loaded[arm] = load_tokenizer(path)
    if baseline not in loaded:
        raise SystemExit(f"{baseline} not available")

    pairs = {arm: allcaps_pairs(tok, min_length) for arm, tok in loaded.items()}
    collapsing = [a for a in variant_arms if a in pairs]
    survivors = set().union(*(pairs[a] for a in collapsing)) if collapsing else set()
    gone = sorted(pairs[baseline] - survivors)
    print(f"\n{lang}/{trainer}: {len(pairs[baseline])} ALLCAPS pairs in {baseline}, "
          f"{len(gone)} collapsed by every variant, {len(pairs[baseline]) - len(gone)} surviving somewhere\n")

    print(f"{'arm':<18}{'ALLCAPS cost':>14}{'lowercase cost':>16}{'pairs left':>12}")
    for arm, tok in loaded.items():
        up = [word_cost(tok, w.upper()) for w in gone]
        low = [word_cost(tok, w) for w in gone]
        print(f"{arm:<18}{sum(up) / len(up):>14.2f}{sum(low) / len(low):>16.2f}{len(pairs[arm]):>12,}")

    for arm, tok in loaded.items():
        longest = sorted({t for t, _ in surfaces(tok) if t.isupper() and t.isalpha() and len(t) > 1},
                         key=len, reverse=True)[:examples]
        print(f"\n{arm} longest ALLCAPS: " + ", ".join(f"{t}({len(t)})" for t in longest))

    sample = gone[:examples]
    if sample:
        print(f"\ncost of the ALLCAPS form, {examples} collapsed words:")
        print(f"{'word':<16}" + "".join(f"{a:>18}" for a in loaded))
        for w in sample:
            print(f"{w.upper():<16}" + "".join(f"{word_cost(t, w.upper()):>18}" for t in loaded.values()))


def span_kinds(tokenizer):
    """Split learned tokens into whole spans and fragments, with their surfaces.

    The markers make this exact rather than heuristic: a token that covers a complete
    boundary span opens and closes with one, so after the case codes are set aside the
    first and last atomic tokens are both the marker. Anything else is a piece that only
    means something with more text either side. `plain` has no markers and cannot be
    classified this way, which is why it is not compared here.
    """
    pt = tokenizer.pretokenizer
    marker = getattr(pt, "marker_token_id", None)
    shift, caps = getattr(pt, "shift_token_id", None), getattr(pt, "caps_token_id", None)
    if marker is None:
        return None

    whole, fragment = [], []
    for token in tokenizer.tokens.values():
        ids = [i for i in token.atomic_tokens if i not in (shift, caps)]
        if len(list(token.atomic_tokens)) == 1 or not ids:
            continue
        code = "shift" if shift in token.atomic_tokens else ("caps" if caps in token.atomic_tokens else None)
        core = [i for i in ids if i != marker]
        text = pt.try_decode_strict(core) if core else None
        if text is None:
            continue
        text = text[1:] if text.startswith(" ") else text
        if code == "shift":
            text = text[:1].upper() + text[1:]
        elif code == "caps":
            text = text.upper()
        (whole if len(ids) > 1 and ids[0] == marker and ids[-1] == marker else fragment).append(text)
    return whole, fragment


@app.command
def fragments(
    lang: str = "de",
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    arms: str = "bnd_wpd,bnd_wpd_caps,bnd_wpd_extcaps",
    trainers: str = "bpe,mingram",
    examples: int = 6,
) -> None:
    """Whole spans versus fragments per arm and trainer, and the ALLCAPS split within them.

    BPE builds a vocabulary by merging outward from characters, so a frequent long word
    leaves its prefixes behind as entries in their own right. MinGram selects a vocabulary
    against an objective instead, so a piece has to earn its slot. This checks whether that
    difference is what puts ALLCAPS fragments like `OSTENLOS` in the vocabulary.

    Args:
        lang: Language code.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        arms: Comma-separated boundary arms. `plain` cannot be classified and is rejected.
        trainers: Comma-separated trainers to compare.
        examples: How many longest ALLCAPS fragments to show.
    """
    print(f"\n{lang}, learned tokens by span coverage\n")
    print(f"{'arm':<18}{'trainer':<10}{'whole':>9}{'fragment':>10}{'frag %':>9}"
          f"{'CAPS whole':>12}{'CAPS frag':>11}")
    shown = {}
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        for trainer in [t.strip() for t in trainers.split(",") if t.strip()]:
            path = os.path.join(TOKENIZERS, f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}.json.gz")
            if not os.path.exists(path):
                print(f"{arm:<18}{trainer:<10}  not trained yet")
                continue
            kinds = span_kinds(load_tokenizer(path))
            if kinds is None:
                raise SystemExit(f"{arm} has no markers; only boundary arms can be classified")
            whole, frag = kinds
            caps_w = [t for t in whole if t.isupper() and t.isalpha() and len(t) > 1]
            caps_f = [t for t in frag if t.isupper() and t.isalpha() and len(t) > 1]
            shown[(arm, trainer)] = caps_f
            print(f"{arm:<18}{trainer:<10}{len(whole):>9,}{len(frag):>10,}"
                  f"{100 * len(frag) / (len(whole) + len(frag)):>8.1f}%{len(caps_w):>12,}{len(caps_f):>11,}")

    for (arm, trainer), caps_f in shown.items():
        longest = sorted(set(caps_f), key=len, reverse=True)[:examples]
        print(f"\n{arm}/{trainer} longest ALLCAPS fragments: "
              + (", ".join(f"{t}({len(t)})" for t in longest) or "none"))


def render(pt, ids):
    """One token as readable text, with its markers and codes visible.

    `|` is a boundary marker, `^` a title-case code, `^^` a caps-lock code. Without them
    the interesting part is invisible: two schemes can spell a word with the same letters
    and differ entirely in where the codes and markers fall.
    """
    marker = getattr(pt, "marker_token_id", None)
    shift, caps = getattr(pt, "shift_token_id", None), getattr(pt, "caps_token_id", None)
    out, run = "", []
    for i in ids:
        if i in (marker, shift, caps):
            if run:
                out += pt.try_decode_strict(run) or "?"
                run = []
            out += {marker: "|", shift: "^", caps: "^^"}[i]
        else:
            run.append(i)
    if run:
        out += pt.try_decode_strict(run) or "?"
    return out


def spelling(tokenizer, word):
    pt = tokenizer.pretokenizer
    ids = tokenizer.encode(word)
    pieces = [render(pt, list(tokenizer.tokens[i].atomic_tokens)) for i in ids]
    return len(ids), " · ".join(pieces)


@app.command
def spelled(
    lang: str = "de",
    corpus: str = "fineweb_{lang}_5gb_quick",
    vocab: int = 34_685,
    baseline: str = "plain",
    variants: str = "bnd_wpd_caps,bnd_wpd_extcaps",
    trainers: str = "bpe,mingram",
    examples: int = 6,
) -> None:
    """The longest ALLCAPS entries the baseline has and no caps variant does, and how the
    variants spell them instead.

    These are the words the codes bought a slot back on. What that slot cost is visible
    only in the spelling: whether the variant still has one entry for the word and adds a
    code, or has to assemble it from pieces.

    Args:
        lang: Language code.
        corpus: Corpus name pattern, `{lang}` substituted.
        vocab: Matched total vocabulary in the tokenizer filenames.
        baseline: Arm whose ALLCAPS entries are the starting set.
        variants: Comma-separated arms that may have dropped them.
        trainers: Comma-separated trainers, each compared separately.
        examples: How many words to show per trainer.
    """
    def load(arm, trainer):
        path = os.path.join(TOKENIZERS, f"{corpus.format(lang=lang)}_{arm}_{trainer}_v{vocab}.json.gz")
        return load_tokenizer(path) if os.path.exists(path) else None

    def allcaps(tok):
        return {t for t, _ in surfaces(tok) if t.isupper() and t.isalpha() and len(t) > 1}

    variant_arms = [v.strip() for v in variants.split(",") if v.strip()]
    for trainer in [t.strip() for t in trainers.split(",") if t.strip()]:
        base = load(baseline, trainer)
        loaded = {arm: load(arm, trainer) for arm in variant_arms}
        if base is None or any(t is None for t in loaded.values()):
            print(f"\n[skip] {trainer}: not every arm is trained yet")
            continue
        only = allcaps(base) - set().union(*(allcaps(t) for t in loaded.values()))
        print(f"\n{lang}/{trainer}: {len(only)} ALLCAPS entries in {baseline} that no variant has\n")
        for word in sorted(only, key=lambda w: (-len(w), w))[:examples]:
            n, spelled_base = spelling(base, word)
            print(f"  {word} ({len(word)} chars)")
            print(f"    {baseline:<18} {n}  {spelled_base}")
            for arm, tok in loaded.items():
                n, text = spelling(tok, word)
                print(f"    {arm:<18} {n}  {text}")


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
