"""Surface morph segmentation by an LLM, in batches, through nit (neko-algo `libs/nit`).

Models are nit registry ids (claude-haiku-4-5, claude-sonnet-5, gpt-6-astra, ...). nit is not
a dependency of this repo, so add it to the run:

    uv run --group morphology --with-editable ~/Desktop/neko/neko-algo/libs/nit \\
        python -m paper_utils.morphology.llm_segment --lang eng --model claude-haiku-4-5 --n 300
"""

import asyncio
import json
import logging
import os
import random

import cyclopts
from cur8.nit import get_model_provider

from paper_utils.morphology.evaluate import report
from paper_utils.morphology.gold import offsets, sigmorphon

OUT = os.path.join(os.path.dirname(__file__), "../../results/morphology")
GOLDFISH = {"eng": "eng_latn", "deu": "deu_latn", "fin": "fin_latn", "rus": "rus_cyrl"}
LANG_NAME = {"eng": "English", "deu": "German", "fin": "Finnish", "rus": "Russian"}

PROMPT = """Segment each {language} word into its surface morphs: contiguous substrings that \
concatenate exactly to the word as written. Split inflectional endings, derivational affixes \
and compound members. Do not split a root that has no further morphological structure, and \
do not split loanwords or names unless they contain a productive affix. When spelling changes \
at a boundary (doubled consonant, dropped e, y->i), keep the extra or changed letters with the \
stem: run|n|ing is wrong, runn|ing is right; hop|ing, happi|ness.

Output exactly one line per input word, in the input order, as: word<TAB>morph|morph|...
No other text.

{words}"""


MINIMAL_PROMPT = """Split each {language} word into its morphemes, writing the pieces so they \
concatenate back to the word exactly.

Output one line per word: word<TAB>piece|piece|...
No other text.

{words}"""

TYPED_PROMPT = """Segment each {language} word into surface morphs and label every boundary.

The morphs must concatenate exactly to the word as written. Mark each boundary with one symbol:
  +  inflection (plural, case, tense, person, comparison, participle endings)
  -  derivation (an affix that forms a new word, e.g. -ness, -al, -ize, -ly, un-, re-)
  =  compound (between two members that are words in their own right)

Rules:
- Split a derivational affix only if the remaining base is (a spelling variant of) a real word
  of the language that the word is derived from: conclus-ion (from conclude), but pretend,
  federal, surgery stay whole.
- Split all the way down: person-al-ize+d, not personalize+d.
- Affixes keep their dictionary spelling; spelling changes belong to the stem: ignor+ed,
  patch+es, runn+ing, happi-ness, therapi+es.
- Names, loanwords and unanalysable roots stay whole.

Output exactly one line per input word, in the input order, as: word<TAB>segmentation
No other text.

{words}"""
BOUNDARY = {"+": "infl", "-": "deriv", "=": "comp"}
log = logging.getLogger(__name__)


def parse_typed(word: str, seg: str) -> list[tuple[int, str]] | None:
    out, at, buf = [], 0, ""
    for ch in seg.strip():
        if ch in BOUNDARY:
            at += len(buf)
            out.append((at, BOUNDARY[ch]))
            buf = ""
        else:
            buf += ch
    return out if seg.strip().translate({ord(c): None for c in BOUNDARY}) == word else None


async def ask(model: str, prompt: str) -> tuple[str, float]:
    """One completion through nit; returns the reply and its cost in USD (0 when litellm has no price)."""
    try:
        r = await get_model_provider(model).complete([{"role": "user", "content": prompt}], log_name="morph segment")
    except ValueError as e:
        # Foundry intermittently answers a plain word list with no content (content_filter, or
        # stop with nothing); the same batch passes on rerun. Treat it as a dropped batch: the
        # caller retries missing words and reports what stays missing.
        if "no content from" not in str(e):
            raise
        log.warning("%s batch dropped: %s", model, e)
        return "", 0.0
    if r.finish_reason != "stop":
        log.warning("%s reply ended with %s; words past the cut are retried", model, r.finish_reason)
    return r.text, r.cost or 0.0


def parse_lines(out: str, words: list[str], parse) -> dict:
    got, wanted = {}, set(words)
    for line in out.splitlines():
        if "\t" not in line:
            continue
        word, seg = line.split("\t", 1)
        if word in wanted and (parsed := parse(word, seg)) is not None:
            got[word] = parsed
    return got


def parse_plain(word: str, seg: str) -> list[int] | None:
    morphs = seg.strip().split("|")
    return offsets(morphs) if "".join(morphs) == word else None


async def segment_batch(words: list[str], language: str, model: str, typed: bool = False,
                        minimal: bool = False) -> tuple[dict, float]:
    template = TYPED_PROMPT if typed else MINIMAL_PROMPT if minimal else PROMPT
    out, cost = await ask(model, template.format(language=language, words="\n".join(words)))
    return parse_lines(out, words, parse_typed if typed else parse_plain), cost


async def segment(words: list[str], language: str, model: str, typed: bool = False, batch: int = 100,
                  minimal: bool = False, on_batch=None) -> dict:
    """All batches at once; nit's per-model `max_concurrency` bounds what is in flight.

    `on_batch(got, cost)` runs as each batch returns, so a caller can cache partial progress.
    """
    chunks = [words[i:i + batch] for i in range(0, len(words), batch)]
    got = {}

    async def one(chunk: list[str]) -> None:
        part, cost = await segment_batch(chunk, language, model, typed, minimal)
        got.update(part)
        if on_batch:
            on_batch(part, cost)

    await asyncio.gather(*(one(c) for c in chunks))
    return got


app = cyclopts.App()


@app.default
def main(lang: str = "eng", model: str = "claude-haiku-4-5", n: int = 300, top: int = 10_000, seed: int = 0,
         typed: bool = False, minimal: bool = False, tag: str = "") -> None:
    ref = {w: b for w, (b, _) in sigmorphon(lang).items()}
    with open(os.path.join(OUT, "counts", f"{GOLDFISH[lang]}.json")) as f:
        ranked = list(json.load(f)["counts"])[:top]
    sample = random.Random(seed).sample([w for w in ranked if w in ref], n)
    pred = asyncio.run(segment(sample, LANG_NAME[lang], model, typed, minimal=minimal))
    print(f"{len(pred)}/{len(sample)} parsed")
    flat = {w: [b for b, *_ in v] if typed else v for w, v in pred.items()}
    # one band: the sample is drawn from the whole top-k
    kind = "_typed" if typed else "_minimal" if minimal else ""
    report(f"LLM {model}{kind}", flat, ref, sample)
    name = f"llm_{lang}_{model}{kind}{tag}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(pred, f, ensure_ascii=False)


if __name__ == "__main__":
    app()
