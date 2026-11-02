"""Surface morph segmentation by an LLM through the Claude Code CLI, in batches.

    uv run python -m paper_utils.morphology.llm_segment --lang eng --model haiku --n 300
"""

import concurrent.futures
import json
import os
import random
import subprocess

import cyclopts

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


def ask(model: str, prompt: str) -> str:
    """Claude models through the Claude Code CLI, anything else through the Copilot CLI."""
    if model in ("haiku", "sonnet", "opus", "fable"):
        cmd = ["claude", "-p", "--model", model, prompt]
    else:
        cmd = ["copilot", "-p", prompt, "--model", model, "-s"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def segment_typed_batch(words: list[str], language: str, model: str) -> dict[str, list[tuple[int, str]]]:
    out = ask(model, TYPED_PROMPT.format(language=language, words="\n".join(words)))
    got = {}
    for line in out.splitlines():
        if "\t" not in line:
            continue
        word, seg = line.split("\t", 1)
        if word in words and (parsed := parse_typed(word, seg)) is not None:
            got[word] = parsed
    return got


def segment_batch(words: list[str], language: str, model: str) -> dict[str, list[int]]:
    out = ask(model, PROMPT.format(language=language, words="\n".join(words)))
    got = {}
    for line in out.splitlines():
        if "\t" not in line:
            continue
        word, seg = line.split("\t", 1)
        morphs = seg.strip().split("|")
        if word in words and "".join(morphs) == word:
            got[word] = offsets(morphs)
    return got


def segment(words: list[str], language: str, model: str, typed: bool = False, batch: int = 100,
            workers: int = 4) -> dict:
    chunks = [words[i:i + batch] for i in range(0, len(words), batch)]
    fn = segment_typed_batch if typed else segment_batch
    got = {}
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for part in pool.map(lambda c: fn(c, language, model), chunks):
            got.update(part)
    return got


app = cyclopts.App()


@app.default
def main(lang: str = "eng", model: str = "haiku", n: int = 300, top: int = 10_000, seed: int = 0,
         typed: bool = False, tag: str = "") -> None:
    ref = {w: b for w, (b, _) in sigmorphon(lang).items()}
    with open(os.path.join(OUT, "counts", f"{GOLDFISH[lang]}.json")) as f:
        ranked = list(json.load(f)["counts"])[:top]
    sample = random.Random(seed).sample([w for w in ranked if w in ref], n)
    pred = segment(sample, LANG_NAME[lang], model, typed)
    print(f"{len(pred)}/{len(sample)} parsed")
    flat = {w: [b for b, *_ in v] if typed else v for w, v in pred.items()}
    # one band: the sample is drawn from the whole top-k
    report(f"LLM {model}{' typed' if typed else ''}", flat, ref, sample)
    name = f"llm_{lang}_{model}{'_typed' if typed else ''}{tag}.json"
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(pred, f, ensure_ascii=False)


if __name__ == "__main__":
    app()
