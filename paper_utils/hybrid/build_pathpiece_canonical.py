#!/usr/bin/env python3
"""Build one canonical PathPiece tokenizer (default iv per init, current MAIN_PRUNE_BATCH_FRACTION).
Usage: build_pathpiece_canonical.py <corpus> <init> [num_workers]
Used to (re)build the canonical pb=0.1 PathPiece set after flipping the default. Forkserver-safe."""
import sys
from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.pathpiece import PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import create_logger
from paper_utils.hybrid.train_pathpiece import (
    ADDITIONAL_VOCAB_SIZE, MAIN_MAX_TOKEN_WIDTH, MAIN_PRUNE_BATCH_FRACTION,
    PRETOKENIZER_NAME, _default_init_vocab, get_model_path,
)

def main():
    corpus_name, init = sys.argv[1], sys.argv[2]
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    iv = _default_init_vocab(init)
    logger = create_logger(f"ppcanon[{corpus_name},{init},pb{MAIN_PRUNE_BATCH_FRACTION}]", verbose=True)
    out = get_model_path(corpus_name, init=init)
    if out.exists():
        logger.info(f"[skip] cached {out}")
        return
    pretok = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(corpus_name, pretok)
    cfg = PathPieceTrainerConfig(additional_vocab_size=ADDITIONAL_VOCAB_SIZE, num_workers=nw,
        init=init, init_vocab_size=iv, max_token_width=MAIN_MAX_TOKEN_WIDTH,
        prune_batch_fraction=MAIN_PRUNE_BATCH_FRACTION, verbose=False)
    model = PathPieceTrainer(pretok, corpus, cfg).train()
    model.metadata["train_corpus"] = corpus_name
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    logger.info(f"Saved: {out}")

if __name__ == "__main__":
    main()
