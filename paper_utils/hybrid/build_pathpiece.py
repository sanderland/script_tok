"""Train PathPiece (BPE-init, the grid's main config) on an arbitrary corpus, so
the compression grid's FineWeb/h6 panels get PathPiece bars (fills the gaps).
Saves to get_model_path(...) where the grid looks it up. Skips if already trained.

Usage: build_pathpiece.py <corpus_name> [num_workers]
"""
from __future__ import annotations

import sys
import time

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.pathpiece import PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import create_logger

from paper_utils.hybrid.train_pathpiece import (
    MAIN_BPE_INIT_VOCAB,
    MAIN_MAX_TOKEN_WIDTH,
    MAIN_PRUNE_BATCH_FRACTION,
    PRETOKENIZER_NAME,
    get_model_path,
)
from paper_utils.unigram.train_hyperparameters import ADDITIONAL_VOCAB_SIZE


def main() -> None:
    corpus_name = sys.argv[1]
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    logger = create_logger(f"pp[{corpus_name}]", verbose=True)

    out_path = get_model_path(corpus_name, init="bpe")  # iv=262144, L=1024, pb=0.1, n=32768
    if out_path.exists():
        logger.info(f"[skip] cached at {out_path}")
        return

    pretok = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(corpus_name, pretok)
    cfg = PathPieceTrainerConfig(
        additional_vocab_size=ADDITIONAL_VOCAB_SIZE,
        num_workers=num_workers,
        init="bpe",
        init_vocab_size=MAIN_BPE_INIT_VOCAB,
        max_token_width=MAIN_MAX_TOKEN_WIDTH,
        prune_batch_fraction=MAIN_PRUNE_BATCH_FRACTION,
        verbose=True,
    )
    t0 = time.perf_counter()
    model = PathPieceTrainer(pretok, corpus, cfg).train()
    model.metadata["train_corpus"] = corpus_name
    perf = model.corpus_performance(corpus)
    model.metadata["performance_train"] = perf
    model.metadata["performance"] = perf
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    logger.info(f"[done] {(time.perf_counter()-t0)/60:.1f}min, |V|={len(model.tokens):,} -> {out_path}")


if __name__ == "__main__":
    main()
