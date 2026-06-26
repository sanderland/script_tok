#!/usr/bin/env python3
"""Held-out token count for one PathPiece model at a given prune_batch_fraction.

Usage: eval_one_pp_pb.py <train_corpus> <eval_corpus> <init_vocab_size> <prune_batch_fraction>
Prints: RESULT tokens/<train>/<eval>/pathpiece_bpe/<model_name> <total_tokens_len>
"""

import sys

from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.pathpiece.model import PathPieceModel

from paper_utils.hybrid.train_pathpiece import get_model_path


def main() -> None:
    train, eval_corpus = sys.argv[1], sys.argv[2]
    iv, pb = int(sys.argv[3]), float(sys.argv[4])
    skip = len(sys.argv) > 5 and sys.argv[5].lower() in ("1", "true", "yes")
    mp = get_model_path(train, init="bpe", init_vocab_size=iv, prune_batch_fraction=pb, skip_substring_in_batch=skip)
    if not mp.exists():
        print(f"MISSING {mp}", file=sys.stderr)
        sys.exit(2)
    model = PathPieceModel.load(str(mp))
    corpus = load_corpus_by_name(eval_corpus, model.pretokenizer)
    tok = int(model.corpus_performance(corpus)["total_tokens_len"])
    print(f"RESULT tokens/{train}/{eval_corpus}/pathpiece_bpe/{mp.name} {tok}")


if __name__ == "__main__":
    main()
