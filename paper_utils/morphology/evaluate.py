"""Micro boundary precision, recall and F1 of a segmenter against a reference, by frequency band."""

import json
import os

COUNTS = os.path.join(os.path.dirname(__file__), "../../results/morphology/counts")
BANDS = [(0, 1_000), (1_000, 10_000), (10_000, 50_000)]


def top_words(lang: str, n: int) -> list[str]:
    with open(os.path.join(COUNTS, f"{lang}.json")) as f:
        return list(json.load(f)["counts"])[:n]


def prf(pred: dict[str, list[int]], ref: dict[str, list[int]], words: list[str]) -> dict:
    tp = fp = fn = exact = n = 0
    for w in words:
        if w not in ref or w not in pred:
            continue
        p, g = set(pred[w]), set(ref[w])
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
        exact += p == g
        n += 1
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": n, "P": precision, "R": recall, "F1": f1, "word_acc": exact / n if n else 0.0}


def report(name: str, pred: dict, ref: dict, ranked: list[str]) -> dict:
    rows = {f"{a // 1000}k-{b // 1000}k": prf(pred, ref, ranked[a:b]) for a, b in BANDS}
    cells = "  ".join(f"{k}: F1 {v['F1']:.3f} (P {v['P']:.2f} R {v['R']:.2f}, acc {v['word_acc']:.2f}, n {v['n']})"
                      for k, v in rows.items() if v["n"])
    print(f"{name:<28}{cells}", flush=True)
    return rows
