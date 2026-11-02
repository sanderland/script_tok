"""Word frequency lists from the head of a Goldfish corpus.

A word is a maximal run of letters (combining marks included), lowercased. The head of the
file is enough: a top-100k list stabilises long before a gigabyte.

    uv run python -m paper_utils.morphology.word_counts --lang eng_latn --gigabytes 1
"""

import collections
import glob
import json
import os

import cyclopts
import regex

GOLDFISH = os.path.expanduser("~/.cache/huggingface/hub/datasets--goldfish-models--fish-food/snapshots/*/{}.txt")
OUT = os.path.join(os.path.dirname(__file__), "../../results/morphology/counts")
WORD = regex.compile(r"\p{L}[\p{L}\p{M}]*")

app = cyclopts.App()


@app.default
def main(lang: str, gigabytes: float = 1.0, top: int = 200_000) -> None:
    (path,) = glob.glob(GOLDFISH.format(lang))
    budget = int(gigabytes * 1e9)
    counts: collections.Counter[str] = collections.Counter()
    read = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            counts.update(m.lower() for m in WORD.findall(line))
            read += len(line)
            if read >= budget:
                break
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{lang}.json")
    with open(out, "w") as f:
        json.dump({"chars_read": read, "total_words": sum(counts.values()),
                   "counts": dict(counts.most_common(top))}, f, ensure_ascii=False)
    print(f"{lang}: {read / 1e9:.2f}G chars, {sum(counts.values()):,} words, {len(counts):,} types -> {out}")


if __name__ == "__main__":
    app()
