"""Silver surface segmentation for the most frequent words: several LLMs, per-boundary vote.

Each model annotates every word with typed boundaries (`llm_segment.TYPED_PROMPT`). A boundary
is kept when a strict majority of the models that returned a parse for the word place it; its
type is the most common label among them. `agreement` is the share of models whose full
segmentation equals the voted one, so a consumer can filter to unanimous words.

Output TSV, one word per line, frequency order:
    rank  word  count  segmentation  agreement  n_models
with the segmentation written as morphs joined by `+` (inflection), `-` (derivation) and `=`
(compound).

    uv run --group morphology --with "cur8-nit @ git+https://github.com/Neko-Health/neko-algo.git@main#subdirectory=libs/nit" \\
        python -m paper_utils.morphology.build_silver --lang eng --top 500
"""

import asyncio
import collections
import json
import os

import cyclopts
import regex

from paper_utils.morphology.llm_segment import GOLDFISH, LANG_NAME, segment

OUT = os.path.join(os.path.dirname(__file__), "../../results/morphology/silver")
COUNTS = os.path.join(os.path.dirname(__file__), "../../results/morphology/counts")
SYMBOL = {"infl": "+", "deriv": "-", "comp": "="}

app = cyclopts.App()


async def annotate(words: list[str], lang: str, model: str, cache: str) -> dict:
    """Annotate, caching per model after every batch so an interrupted or extended run only asks for what is missing."""
    got = json.load(open(cache)) if os.path.exists(cache) else {}
    spent = 0.0

    def save(part: dict, cost: float) -> None:
        nonlocal spent
        got.update(part)
        spent += cost
        with open(cache, "w") as f:
            json.dump(got, f, ensure_ascii=False)

    for _attempt in range(2):            # one retry for words a batch dropped or mangled
        todo = [w for w in words if w not in got]
        if not todo:
            break
        await segment(todo, LANG_NAME[lang], model, typed=True, on_batch=save)
    print(f"{model}: {sum(w in got for w in words):,}/{len(words):,} parsed, ${spent:.3f} this run")
    return got


def vote(parses: list[list[tuple[int, str]]]) -> tuple[list[tuple[int, str]], float]:
    need = len(parses) // 2 + 1
    where = collections.Counter(b for p in parses for b, _ in p)
    kinds = collections.defaultdict(collections.Counter)
    for p in parses:
        for b, t in p:
            kinds[b][t] += 1
    voted = sorted((b, kinds[b].most_common(1)[0][0]) for b, c in where.items() if c >= need)
    offsets = [b for b, _ in voted]
    agree = sum([b for b, _ in p] == offsets for p in parses) / len(parses)
    return voted, agree


def render(word: str, bounds: list[tuple[int, str]]) -> str:
    out, at = "", 0
    for b, t in bounds:
        out += word[at:b] + SYMBOL[t]
        at = b
    return out + word[at:]


@app.default
def main(lang: str = "eng", top: int = 500, models: str = "gpt-6.1-sol,claude-sonnet-5-5,gpt-6-astra") -> None:
    with open(os.path.join(COUNTS, f"{GOLDFISH[lang]}.json")) as f:
        counts = json.load(f)["counts"]
    # single letters carry no boundary; everything else in the frequency list is annotated
    rank = {w: i for i, w in enumerate(counts, 1)}
    words = [w for w in list(counts)[:top] if len(w) > 1 and regex.fullmatch(r"\p{L}+", w)]
    os.makedirs(OUT, exist_ok=True)
    names = models.split(",")

    async def run_all() -> list[dict]:
        return await asyncio.gather(*(annotate(words, lang, m, os.path.join(OUT, f"{lang}_{m}.json")) for m in names))

    per_model = dict(zip(names, asyncio.run(run_all())))
    path = os.path.join(OUT, f"{lang}_top{top}.tsv")
    missing, unanimous = 0, 0
    with open(path, "w", encoding="utf-8") as f:
        print("rank\tword\tcount\tsegmentation\tagreement\tn_models", file=f)
        for w in words:
            parses = [[tuple(x) for x in per_model[m][w]] for m in per_model if w in per_model[m]]
            if len(parses) < 2:
                missing += 1
                continue
            voted, agree = vote(parses)
            unanimous += agree == 1.0
            print(f"{rank[w]}\t{w}\t{counts[w]}\t{render(w, voted)}\t{agree:.2f}\t{len(parses)}", file=f)
    print(f"{path}: {len(words) - missing:,} words, {unanimous:,} unanimous, {missing} without two parses")


if __name__ == "__main__":
    app()
