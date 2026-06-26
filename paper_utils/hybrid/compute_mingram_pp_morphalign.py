#!/usr/bin/env python3
"""Compute MorphAlign for the MinGram-PP MinGram sweep (eng/deu/fin) and the matched
stock (usage-count) p=0.9 models, folding both into cache_morphalign_scatter.json.

Tests whether the MinGram-PP prune criterion changes morphological alignment, holding the
iterative schedule (p=0.9) fixed. Run after the MinGram-PP sweep has built these langs.
"""


from paper_utils.hybrid.generate_morphalign_scatter import (
    LANGUAGE_CONFIGS,
    MORPHALIGN_CACHE_PATH,
    morphalign_score,
)
from paper_utils.hybrid.utils import load_cache
from paper_utils.hybrid.train_mingram import get_model_path
from script_bpe.tokenizers.mingram.model import MinGramModel

FACTORS = [1.0, 1.05, 1.1, 1.15, 1.25, 1.5, 2.0, 3.0, 5.0]
EM, P = 2, 0.9


def _score(train, gold, f, criterion, cache):
    mp = get_model_path(train, f, EM, P, prune_criterion=criterion)
    if not mp.exists():
        return None
    key = f"{_lang(train)}/mingram/{mp.name}"
    model = MinGramModel.load(str(mp))
    return morphalign_score(model, gold, cache, key)


_LANG_BY_TRAIN = {c["train_corpus"]: c["lang"] for c in LANGUAGE_CONFIGS}


def _lang(train):
    return _LANG_BY_TRAIN[train]


def main() -> None:
    cache = load_cache(MORPHALIGN_CACHE_PATH)
    print(f"{'lang':5} {'f':>5} {'stock_p09':>10} {'mingram_pp':>11} {'delta':>8}")
    for cfg in LANGUAGE_CONFIGS:
        train, gold = cfg["train_corpus"], cfg["gold_file"]
        for f in FACTORS:
            stock = _score(train, gold, f, "usage_count", cache)
            careful = _score(train, gold, f, "mi", cache)
            if stock is None and careful is None:
                continue
            d = (careful - stock) if (stock and careful) else float("nan")
            print(f"{cfg['lang']:5} {f:>5} "
                  f"{(stock if stock else float('nan')):>10.4f} "
                  f"{(careful if careful else float('nan')):>11.4f} {d:>+8.4f}")
    print("done; cache updated at", MORPHALIGN_CACHE_PATH)


if __name__ == "__main__":
    main()
