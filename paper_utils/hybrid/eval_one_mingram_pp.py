#!/usr/bin/env python3
"""Compute held-out token count for one MinGram-PP MinGram model and print a cache line.

Usage: eval_one_mingram_pp.py <train_corpus> <eval_corpus> <factor>
Prints: RESULT tokens/<train>/<eval>/mingram/<model_name> <total_tokens_len>
(matching the cache_train_eval_compression_grid.json key format, so it folds in directly.)
"""

import sys

from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.mingram.model import MinGramModel

from paper_utils.hybrid.train_mingram import get_model_path

MINGRAM_EM, MINGRAM_P = 2, 0.9


def main() -> None:
    train, eval_corpus, factor = sys.argv[1], sys.argv[2], float(sys.argv[3])
    skip = len(sys.argv) > 4 and sys.argv[4].lower() in ("1", "true", "yes")
    mp = get_model_path(train, factor, MINGRAM_EM, MINGRAM_P, prune_criterion="mi", skip_substring_in_batch=skip)
    if not mp.exists():
        print(f"MISSING {mp}", file=sys.stderr)
        sys.exit(2)
    model = MinGramModel.load(str(mp))
    corpus = load_corpus_by_name(eval_corpus, model.pretokenizer)
    tok = int(model.corpus_performance(corpus)["total_tokens_len"])
    print(f"RESULT tokens/{train}/{eval_corpus}/mingram/{mp.name} {tok}")


if __name__ == "__main__":
    main()
