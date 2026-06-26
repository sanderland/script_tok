#!/usr/bin/env python3
"""Test Craig's substring-skip batch rule in MinGram-PP MinGram and PathPiece.

For each method, trains skip=off vs skip=on at a heavy-pruning overshoot (shared BPE init,
so the only difference is the batch rule) and compares held-out compression. Throwaway harness.

  uv run python -m paper_utils.hybrid.proto_substring_skip --corpus eng_latn_300mb --eval flores_plus_eng_latn --factor 2.0
"""

import argparse
import copy
import time

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.pathpiece.trainer import PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import create_logger

PRETOKENIZER_NAME = "scriptenc_cb"
VOCAB = 32768
ATOMS = 1916


def _evaltok(model, eval_corpus):
    return int(model.corpus_performance(eval_corpus)["total_tokens_len"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eng_latn_300mb")
    ap.add_argument("--eval", default="flores_plus_eng_latn")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    logger = create_logger("proto_subskip")
    pretok = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(args.corpus, pretok)
    eval_corpus = load_corpus_by_name(args.eval, pretok)
    iv = round(args.factor * VOCAB) + ATOMS
    rows = []

    # ---- MinGram-PP (p=0.9), shared BPE init ----
    def mg_cfg(skip):
        return MinGramTrainerConfig(additional_vocab_size=VOCAB, overshoot_factor=args.factor,
                                    num_em_iterations=2, pruning_shrinking_factor=0.9,
                                    prune_criterion="mi", skip_substring_in_batch=skip,
                                    num_workers=args.workers, verbose=False)
    mg_init = MinGramTrainer(pretok, corpus, mg_cfg(False))._build_bpe_init_tokens()
    logger.info(f"MinGram shared init |V|={len(mg_init):,}")
    for skip in (False, True):
        tr = MinGramTrainer(pretok, corpus, mg_cfg(skip))
        tr._build_bpe_init_tokens = lambda init=mg_init: copy.deepcopy(init)
        t0 = time.time()
        m = tr.train()
        rows.append(("MinGram-PP", skip, len(m.tokens), _evaltok(m, eval_corpus), time.time() - t0))
        logger.info(f"MinGram-PP skip={skip}: |V|={len(m.tokens):,} tok={rows[-1][3]:,} ({rows[-1][4]:.0f}s)")

    # ---- PathPiece (pb=0.1, init=bpe), shared BPE init ----
    def pp_cfg(skip):
        return PathPieceTrainerConfig(additional_vocab_size=VOCAB, num_workers=args.workers,
                                      init="bpe", init_vocab_size=iv, max_token_width=1024,
                                      prune_batch_fraction=0.1, skip_substring_in_batch=skip,
                                      verbose=False)
    pp_seed = PathPieceTrainer(pretok, corpus, pp_cfg(False))
    pp_init = pp_seed._build_bpe_init_vocab(iv, 1024)
    logger.info(f"PathPiece shared init |V|={len(pp_init):,}")
    for skip in (False, True):
        tr = PathPieceTrainer(pretok, corpus, pp_cfg(skip))
        tr._build_bpe_init_vocab = lambda _iv, _L, init=pp_init: copy.deepcopy(init)
        t0 = time.time()
        m = tr.train()
        rows.append(("PathPiece", skip, len(m.tokens), _evaltok(m, eval_corpus), time.time() - t0))
        logger.info(f"PathPiece skip={skip}: |V|={len(m.tokens):,} tok={rows[-1][3]:,} ({rows[-1][4]:.0f}s)")

    print(f"\n===== Craig substring-skip test  (corpus={args.corpus} eval={args.eval} f={args.factor}) =====")
    print(f"{'method':12}{'skip':>6}{'|V|':>9}{'eval_tokens':>13}{'vs skip=off':>13}")
    for method in ("MinGram-PP", "PathPiece"):
        base = next(r[3] for r in rows if r[0] == method and r[1] is False)
        for r in rows:
            if r[0] != method:
                continue
            d = (r[3] - base) / base * 100
            print(f"{r[0]:12}{str(r[1]):>6}{r[2]:>9,}{r[3]:>13,}{d:>+12.3f}%")


if __name__ == "__main__":
    main()
