"""Score tokenizers against silver gold: micro boundary P/R/F1 over word-internal boundaries.

Each word is tokenized inside a carrier phrase (`morphscore_boundary.segment_in_context`), so
a leading-space vocabulary is measured on the split it makes in running text. A word the
tokenizer keeps whole adds its gold boundaries to the recall denominator and nothing to
precision. `--kinds` restricts the gold to some boundary types; predicted boundaries that fall
on a gold boundary of an excluded type are ignored rather than counted as false positives, so
an inflection-only score does not punish a tokenizer for also splitting derivation.

    uv run python -m paper_utils.morphology.tokenizer_eval --silver results/morphology/silver/eng_top10000.tsv
"""

import json
import os

import cyclopts

# Registers the boundary pretokenizers, without which the tokenizers do not load.
import paper_utils.boundary.downstream.boundary_tokenizer  # noqa: F401
from paper_utils.boundary.morphscore_boundary import segment_in_context
from paper_utils.morphology.gold import offsets
from script_bpe.tokenizers.load import load_tokenizer

TOKENIZERS = os.path.join(os.path.dirname(__file__), "../boundary/downstream/tokenizers")
OUT = os.path.join(os.path.dirname(__file__), "../../results/morphology")
KIND = {"+": "infl", "-": "deriv", "=": "comp"}
BANDS = [(1, 1_000), (1_001, 10_000), (10_001, 100_000)]

app = cyclopts.App()


def read_silver(path: str, min_agreement: float) -> list[tuple[int, str, list[tuple[int, str]]]]:
    out = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            rank, word, _count, seg, agree, _n = line.rstrip("\n").split("\t")
            if float(agree) < min_agreement:
                continue
            bounds, at = [], 0
            for ch in seg:
                if ch in KIND:
                    bounds.append((at, KIND[ch]))
                else:
                    at += 1
            out.append((int(rank), word, bounds))
    return out


def score(pred: dict[str, list[int]], silver, kinds: set[str]) -> dict:
    res = {}
    for lo, hi in BANDS:
        tp = fp = fn = whole = n = 0
        for rank, word, bounds in silver:
            if not lo <= rank <= hi or word not in pred:
                continue
            gold = {b for b, k in bounds if k in kinds}
            ignore = {b for b, k in bounds if k not in kinds}
            p = set(pred[word]) - ignore
            tp += len(p & gold)
            fp += len(p - gold)
            fn += len(gold - p)
            whole += not pred[word]
            n += 1
        if n:
            P = tp / (tp + fp) if tp + fp else 1.0
            R = tp / (tp + fn) if tp + fn else 1.0
            res[f"{lo}-{hi}"] = {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0,
                                 "whole": whole / n, "n": n}
    return res


@app.default
def main(silver: str, names: str = "", min_agreement: float = 0.0, lang: str = "en",
         vocab: int = 34_685) -> None:
    rows = read_silver(silver, min_agreement)
    words = [w for _, w, _ in rows]
    names = names.split(",") if names else sorted(
        f[:-len(".json.gz")] for f in os.listdir(TOKENIZERS)
        if f.startswith(f"fineweb_{lang}_") and f.endswith(f"_v{vocab}.json.gz"))
    results = {}
    for name in names:
        tok = load_tokenizer(os.path.join(TOKENIZERS, f"{name}.json.gz"))
        pred = {}
        for w in words:
            pieces = segment_in_context(tok, w)
            if pieces is not None:
                pred[w] = offsets(pieces)
        results[name] = {"inflection": score(pred, rows, {"infl"}),
                         "all": score(pred, rows, {"infl", "deriv", "comp"}),
                         "segmented": len(pred), "words": len(words)}
        short = name.replace(f"fineweb_{lang}_5gb_quick_", "").replace(f"_v{vocab}", "")
        cells = "  ".join(f"{band}: infl F1 {results[name]['inflection'][band]['F1']:.3f} "
                          f"all F1 {v['F1']:.3f} (P {v['P']:.2f} R {v['R']:.2f}) whole {v['whole']:.2f}"
                          for band, v in results[name]["all"].items())
        print(f"{short:<22}{cells}", flush=True)
    base = os.path.splitext(os.path.basename(silver))[0]
    with open(os.path.join(OUT, f"tokenizer_eval_{base}.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    app()
