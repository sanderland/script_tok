"""Good-piece gold: per word, the segmentations into meaningful pieces, and a token-level score.

Notation, one line per word:

    un<happi+ness            prefix `<`, suffix `+`, compound `=`
    happi+ness | happ?+iness?  alternatives after `|`
    to?=day?                 `?` marks a piece worth half credit (opaque, historical)
    women                    nothing splits off

A tokenizer token scores the credit of the span it covers: a single piece, or a run of
pieces forming a constituent (a stem with any of its prefixes and suffixes: personal,
unhappi, happiness), with the lowest credit in the run. A run of affixes only (ized) gets
half of that, anything else 0. The whole word is always a constituent, so it scores 1. A
word's score is the mean over its tokens; a tokenizer is scored on the words it splits.

    uv run --group morphology --with "cur8-nit @ git+https://github.com/Neko-Health/neko-algo.git@main#subdirectory=libs/nit" \\
        python -m paper_utils.morphology.pieces --lang eng
"""

import asyncio
import json
import os
import random

import cyclopts
import regex

from paper_utils.morphology.llm_segment import GOLDFISH, LANG_NAME, ask, parse_lines

OUT = os.path.join(os.path.dirname(__file__), "../../results/morphology/pieces")
COUNTS = os.path.join(os.path.dirname(__file__), "../../results/morphology/counts")

PROMPT = """For each {language} word, write how it splits into meaningful pieces.

- Pieces are contiguous substrings that concatenate exactly to the word as written.
- A good piece is a morph that recurs with the same meaning in related words: a stem \
(happi in happier, happily), an inflectional ending, a derivational affix, or a compound member.
- Mark the boundary after a prefix with <, before a suffix (inflectional or derivational) with +, \
and between compound members with =.
- The stem keeps its dictionary form when the spelling allows: use+d, run+ning, stop+ped. When it \
cannot, use the form the stem takes across related words: happi+ness, mak+ing.
- Split all the way down: un<happi+ness, person+al+ize+d.
- Mark a piece with ? when it is only half good: a historical or opaque part (to?=day?), or a \
bound root with no clear meaning of its own (univers?+ity).
- Irregular forms whose grammar is not a separable piece stay whole: women, went, has.
- Leave the word whole when nothing splits off.
- Give an alternative after | only when two segmentations are equally defensible.

Output exactly one line per input word, in the input order, as: word<TAB>segmentation
No other text.

{words}"""

SEP = {"<": "prefix", "+": "suffix", "=": "compound"}
BANDS = [(1, 1_000), (1_001, 10_000), (10_001, 100_000)]

Piece = tuple[int, int, str, float]  # start, end, kind (prefix/stem/suffix), credit


def parse_analysis(word: str, text: str) -> list[Piece] | None:
    tokens = regex.split(r"([<+=])", text.strip())
    pieces, at = [], 0
    for i in range(0, len(tokens), 2):
        raw = tokens[i].strip()
        half = raw.endswith("?")
        surface = raw.rstrip("?")
        before = tokens[i - 1] if i else ""
        after = tokens[i + 1] if i + 1 < len(tokens) else ""
        kind = "prefix" if after == "<" else "suffix" if before == "+" else "stem"
        pieces.append((at, at + len(surface), kind, 0.5 if half else 1.0))
        at += len(surface)
    surfaces = "".join(t.strip().rstrip("?") for t in tokens[::2])
    return pieces if surfaces == word else None


def parse(word: str, text: str) -> list[list[Piece]] | None:
    analyses = [parse_analysis(word, alt) for alt in text.split("|")]
    return None if any(a is None for a in analyses) else analyses


def spans(analyses: list[list[Piece]]) -> dict[tuple[int, int], float]:
    """Credit for every span a token could cover, maximised over analyses."""
    out: dict[tuple[int, int], float] = {}
    for pieces in analyses:
        for i in range(len(pieces)):
            for j in range(i, len(pieces)):
                run = pieces[i:j + 1]
                credit = min(p[3] for p in run)
                if j > i:
                    if run[0][2] == "suffix" and any(p[2] == "stem" for p in run):
                        credit = 0.0      # a suffix glued to a following stem (ness=web)
                    elif run[-1][2] == "prefix" and any(p[2] == "stem" for p in run):
                        credit = 0.0
                    elif not any(p[2] == "stem" for p in run):
                        credit *= 0.5     # affixes only: ized, alize
                key = (run[0][0], run[-1][1])
                out[key] = max(out.get(key, 0.0), credit)
    return out


def word_score(gold: dict[tuple[int, int], float], cuts: list[int], length: int) -> list[float]:
    edges = [0, *cuts, length]
    return [gold.get((a, b), 0.0) for a, b in zip(edges, edges[1:])]


async def annotate(words: list[str], lang: str, model: str, batch: int = 50) -> dict[str, str]:
    """Raw annotation per word, cached after every batch; one retry pass for dropped words."""
    path = os.path.join(OUT, f"{lang}_{model}.json")
    got = json.load(open(path)) if os.path.exists(path) else {}
    spent = 0.0

    async def one(chunk: list[str]) -> None:
        nonlocal spent
        out, cost = await ask(model, PROMPT.format(language=LANG_NAME[lang], words="\n".join(chunk)))
        spent += cost
        got.update(parse_lines(out, chunk, lambda w, s: s.strip() if parse(w, s) else None))
        with open(path, "w") as f:
            json.dump(got, f, ensure_ascii=False, indent=0)

    for _attempt in range(2):
        todo = [w for w in words if w not in got]
        await asyncio.gather(*(one(todo[i:i + batch]) for i in range(0, len(todo), batch)))
    print(f"{model}: {sum(w in got for w in words)}/{len(words)} parsed, ${spent:.3f} this run")
    return got


def sample_words(lang: str, per_band: tuple[int, ...], seed: int) -> dict[str, int]:
    with open(os.path.join(COUNTS, f"{GOLDFISH[lang]}.json")) as f:
        ranked = list(json.load(f)["counts"])[:BANDS[-1][1]]
    rng = random.Random(seed)
    out = {}
    for (lo, hi), n in zip(BANDS, per_band):
        pool = [w for w in ranked[lo - 1:hi] if len(w) > 1 and regex.fullmatch(r"\p{L}+", w)]
        out.update({w: ranked.index(w, lo - 1) + 1 for w in rng.sample(pool, n)})
    return out


def band(rank: int) -> str:
    return next(f"{lo}-{hi}" for lo, hi in BANDS if lo <= rank <= hi)


def score_tokenizers(golds: dict[str, dict[str, str]], ranks: dict[str, int], lang: str) -> dict:
    # Imported here: loading tokenizers registers pretokenizers and is slow.
    import paper_utils.boundary.downstream.boundary_tokenizer  # noqa: F401
    from paper_utils.boundary.morphscore_boundary import segment_in_context
    from paper_utils.morphology.gold import offsets
    from paper_utils.morphology.tokenizer_eval import TOKENIZERS
    from script_bpe.tokenizers.load import load_tokenizer

    names = sorted(f[:-len(".json.gz")] for f in os.listdir(TOKENIZERS)
                   if f.startswith(f"fineweb_{lang[:2]}_") and "caps" not in f)
    results = {}
    for name in names:
        tok = load_tokenizer(os.path.join(TOKENIZERS, f"{name}.json.gz"))
        cuts = {w: offsets(p) for w in ranks if (p := segment_in_context(tok, w)) is not None}
        res = {}
        for model, gold in golds.items():
            # Words with no good split (petunia, osram) score 0 whenever split, which says how
            # often the vocabulary forces a split, not how well it splits; kept apart.
            groups: dict[str, list[list[float]]] = {}
            for w, c in cuts.items():
                if w not in gold:
                    continue
                analyses = parse(w, gold[w])
                kind = "complex" if any(len(a) > 1 for a in analyses) else "simplex"
                groups.setdefault(f"{kind}_all", []).append(word_score(spans(analyses), c, len(w)) if c else [])
                groups.setdefault(f"{kind}_{band(ranks[w])}", []).append(groups[f"{kind}_all"][-1])
            res[model] = {}
            for g, v in sorted(groups.items()):
                split = [s for s in v if len(s) > 1]
                res[model][g] = {"n": len(v), "split": len(split) / len(v),
                                 "score": sum(sum(s) / len(s) for s in split) / len(split) if split else None,
                                 "all_good": sum(min(s) == 1.0 for s in split) / len(split) if split else None}
        results[name] = res
        short = name.replace(f"fineweb_{lang[:2]}_5gb_quick_", "").split("_v")[0]
        cells = []
        for model, r in res.items():
            c, s = r["complex_all"], r["simplex_all"]
            cells.append(f"{model}: complex n={c['n']} split {c['split']:.2f} score {c['score']:.3f} "
                         f"all-good {c['all_good']:.2f} | simplex n={s['n']} split {s['split']:.2f}")
        print(f"{short:<18}" + "   ".join(cells), flush=True)
    return results


app = cyclopts.App()


@app.default
def main(lang: str = "eng", models: str = "gpt-6.1-sol,claude-sonnet-5-5", per_band: str = "100,200,200",
         seed: int = 0) -> None:
    os.makedirs(OUT, exist_ok=True)
    ranks = sample_words(lang, tuple(int(n) for n in per_band.split(",")), seed)
    names = models.split(",")

    async def run_all() -> list[dict]:
        return await asyncio.gather(*(annotate(list(ranks), lang, m) for m in names))

    golds = dict(zip(names, asyncio.run(run_all())))
    words = [w for w in ranks if all(w in g for g in golds.values())]
    with open(os.path.join(OUT, f"{lang}_sample.tsv"), "w", encoding="utf-8") as f:
        print("rank\tword\t" + "\t".join(names), file=f)
        for w in sorted(words, key=ranks.get):
            print(f"{ranks[w]}\t{w}\t" + "\t".join(golds[m][w] for m in names), file=f)
    same = sum(len({golds[m][w] for m in names}) == 1 for w in words)
    print(f"{len(words)} words annotated by all, {same} identical")
    results = score_tokenizers(golds, ranks, lang)
    with open(os.path.join(OUT, f"{lang}_tokenizers.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    app()
