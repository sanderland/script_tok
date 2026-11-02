"""Segmenters: Morfessor (unsupervised and semi-supervised) and a CRF boundary tagger."""

import math

import morfessor
import sklearn_crfsuite

from paper_utils.morphology.gold import offsets


def morfessor_model(counts: dict[str, int], annotations: dict[str, list[int]] | None = None,
                    corpusweight: float = 1.0, annotationweight: float | None = None):
    model = morfessor.BaselineModel(corpusweight=corpusweight)
    # log-dampened counts: raw token frequency makes Morfessor keep frequent words whole
    model.load_data([(1 + int(math.log(c)), w) for w, c in counts.items()])
    if annotations:
        model.set_annotations({w: [tuple(split(w, b))] for w, b in annotations.items()},
                              annotatedcorpusweight=annotationweight)
    model.train_batch()
    return model


def morfessor_segment(model, words: list[str]) -> dict[str, list[int]]:
    return {w: offsets(model.viterbi_segment(w)[0]) for w in words}


def split(word: str, bounds: list[int]) -> list[str]:
    cuts = [0, *bounds, len(word)]
    return [word[a:b] for a, b in zip(cuts, cuts[1:])]


def char_features(word: str, extra: list[set[int]] | None = None) -> list[dict]:
    """Ruokolainen-style left/right substring features at each position."""
    w = f"<{word}>"
    feats = []
    for i in range(1, len(w) - 1):
        f = {"bias": 1.0}
        for k in range(1, 6):
            if i - k >= 0:
                f[f"L{k}"] = w[i - k:i]
            if i + k <= len(w):
                f[f"R{k}"] = w[i:i + k]
        for j, bset in enumerate(extra or []):
            f[f"ext{j}"] = (i - 1) in bset
        feats.append(f)
    return feats


def labels(word: str, bounds: list[int]) -> list[str]:
    b = set(bounds)
    return ["B" if i in b else "M" for i in range(len(word))]


def crf_train(gold: dict[str, list[int]], extra=None, c1: float = 0.1, c2: float = 0.01):
    words = list(gold)
    X = [char_features(w, extra(w) if extra else None) for w in words]
    y = [labels(w, gold[w]) for w in words]
    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=200)
    crf.fit(X, y)
    return crf


def crf_segment(crf, words: list[str], extra=None) -> dict[str, list[int]]:
    tags = crf.predict([char_features(w, extra(w) if extra else None) for w in words])
    return {w: [i for i, t in enumerate(ts) if t == "B" and i > 0] for w, ts in zip(words, tags)}
