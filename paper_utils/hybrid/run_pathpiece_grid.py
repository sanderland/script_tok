"""Train PathPiece tokenizers across the main-table language grid.

Two PathPiece variants are trained per language:

  * ``ngram`` initialization (canonical PathPiece);
  * ``bpe``   initialization (PathPiece-V from a BPE seed).

Both share the min-token shortest-path segmentation. They serve as the
direct prior-art baseline for MinGram's min-token objective, as
requested by reviewer comment 6.

Run with::

    uv run python -m paper_utils.hybrid.run_pathpiece_grid \\
        --langs eng deu fin rus arb kor --inits ngram bpe \\
        --num-workers 8

Models are cached at the path returned by ``train_pathpiece.get_model_path``;
re-running skips already-trained models unless ``--retrain`` is passed.
"""

import argparse
import time
from pathlib import Path

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.pathpiece import PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import create_logger

from paper_utils.hybrid.train_pathpiece import (
    MAIN_INIT_VOCAB,
    MAIN_MAX_TOKEN_WIDTH,
    MAIN_PRUNE_BATCH_FRACTION,
    PRETOKENIZER_NAME,
    VARIANTS,
    get_model_path,
)
from paper_utils.unigram.train_hyperparameters import ADDITIONAL_VOCAB_SIZE

LANG_TO_CORPUS = {
    "eng": "fineweb_en_5gb",
    "deu": "fineweb_de_5gb",
    "fin": "fineweb_fi_5gb",
    "rus": "fineweb_ru_5gb",
    "arb": "fineweb_ar_5gb",
    "kor": "fineweb_ko_5gb",
}


def train_one(
    lang: str,
    init: str,
    *,
    init_vocab_size: int,
    max_token_width: int,
    prune_batch_fraction: float,
    additional_vocab_size: int,
    num_workers: int,
    retrain: bool,
) -> Path:
    logger = create_logger(f"pp[{lang}/{init}]", verbose=True)
    corpus_name = LANG_TO_CORPUS[lang]
    out_path = get_model_path(
        corpus_name,
        init=init,
        init_vocab_size=init_vocab_size,
        max_token_width=max_token_width,
        prune_batch_fraction=prune_batch_fraction,
        additional_vocab_size=additional_vocab_size,
    )
    if out_path.exists() and not retrain:
        logger.info(f"Cached at {out_path}")
        return out_path

    logger.info(f"Training PathPiece ({init}) on {corpus_name} -> {out_path}")
    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(corpus_name, pretokenizer)
    cfg = PathPieceTrainerConfig(
        additional_vocab_size=additional_vocab_size,
        num_workers=num_workers,
        init=init,
        init_vocab_size=init_vocab_size,
        max_token_width=max_token_width,
        prune_batch_fraction=prune_batch_fraction,
        verbose=True,
    )
    t0 = time.perf_counter()
    model = PathPieceTrainer(pretokenizer, corpus, cfg).train()
    elapsed = time.perf_counter() - t0
    model.metadata["corpus"] = corpus_name
    model.metadata["train_corpus"] = corpus_name
    model.metadata["lang"] = lang
    model.metadata["time"] = elapsed
    # also record eval-style metrics on the training corpus so downstream
    # cache_train_eval_compression_grid can pick them up without re-encoding.
    train_perf = model.corpus_performance(corpus)
    model.metadata["performance_train"] = train_perf
    model.metadata["performance"] = train_perf
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    logger.info(f"Saved to {out_path} (final |V|={len(model.tokens):,}, {elapsed/60:.1f}min)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PathPiece main-table grid.")
    parser.add_argument("--langs", nargs="+", default=list(LANG_TO_CORPUS), choices=list(LANG_TO_CORPUS))
    parser.add_argument("--inits", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--init-vocab-size", type=int, default=MAIN_INIT_VOCAB)
    parser.add_argument("--max-token-width", type=int, default=MAIN_MAX_TOKEN_WIDTH)
    parser.add_argument("--prune-batch-fraction", type=float, default=MAIN_PRUNE_BATCH_FRACTION)
    parser.add_argument("--additional-vocab-size", type=int, default=ADDITIONAL_VOCAB_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    logger = create_logger("pp_grid", verbose=True)
    for lang in args.langs:
        for init in args.inits:
            try:
                train_one(
                    lang,
                    init,
                    init_vocab_size=args.init_vocab_size,
                    max_token_width=args.max_token_width,
                    prune_batch_fraction=args.prune_batch_fraction,
                    additional_vocab_size=args.additional_vocab_size,
                    num_workers=args.num_workers,
                    retrain=args.retrain,
                )
            except Exception:
                logger.exception(f"Failed: lang={lang} init={init}")


if __name__ == "__main__":
    main()
