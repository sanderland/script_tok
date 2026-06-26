"""Train PathPiece at a given BPE init-vocab size (overshoot sweep for Fig 3).

init_vocab_size = f * 32768 sweeps the same overshoot axis as mingram/bpe_init,
so PathPiece can join the anchored f-sensitivity figure. Skips if already trained.

The __main__ guard is REQUIRED: the trainer's mp_ctx is forkserver/spawn, which
re-imports this module in each worker — without the guard that re-runs training.

Usage: build_pathpiece_sweep.py <corpus> <init_vocab_size> [num_workers]
"""
import sys
import time

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.pathpiece import PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import create_logger

from paper_utils.hybrid.train_pathpiece import (
    MAIN_MAX_TOKEN_WIDTH,
    MAIN_PRUNE_BATCH_FRACTION,
    PRETOKENIZER_NAME,
    get_model_path,
)
from paper_utils.unigram.train_hyperparameters import ADDITIONAL_VOCAB_SIZE


def main() -> None:
    corpus_name = sys.argv[1]
    iv = int(sys.argv[2])
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    # optional 4th arg: prune_batch_fraction (default = paper main 10% batch).
    # Encoded in the filename (pbX) so it caches separately.
    pb = float(sys.argv[4]) if len(sys.argv) > 4 else MAIN_PRUNE_BATCH_FRACTION
    # optional 5th arg: skip_substring_in_batch (1/true to enable Craig's rule)
    skip = len(sys.argv) > 5 and sys.argv[5].lower() in ("1", "true", "yes")
    logger = create_logger(f"pp[{corpus_name},iv{iv},pb{pb},ss{int(skip)}]", verbose=True)

    out = get_model_path(corpus_name, init="bpe", init_vocab_size=iv, prune_batch_fraction=pb, skip_substring_in_batch=skip)
    if out.exists():
        logger.info(f"[skip] cached at {out}")
        return

    pretok = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(corpus_name, pretok)
    cfg = PathPieceTrainerConfig(
        additional_vocab_size=ADDITIONAL_VOCAB_SIZE,
        num_workers=nw,
        init="bpe",
        init_vocab_size=iv,
        max_token_width=MAIN_MAX_TOKEN_WIDTH,
        prune_batch_fraction=pb,
        skip_substring_in_batch=skip,
        verbose=False,
    )
    t0 = time.perf_counter()
    model = PathPieceTrainer(pretok, corpus, cfg).train()
    model.metadata["train_corpus"] = corpus_name
    perf = model.corpus_performance(corpus)
    model.metadata["performance_train"] = perf
    model.metadata["performance"] = perf
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    logger.info(f"[done] iv={iv} {(time.perf_counter()-t0)/60:.1f}min |V|={len(model.tokens):,} -> {out}")


if __name__ == "__main__":
    main()
