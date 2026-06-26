#!/usr/bin/env python3
"""Train one paper tokenizer: one method, one corpus, one configuration.

Methods:
  bpe           - Plain BPE baseline at the experiment vocab size
  default       - Standard Unigram with iterative pruning
  fsp           - Unigram with single flat-score prune
  bpe_init      - Standard Unigram initialized from BPE (requires --overshoot-factor)
  bpe_init_fsp  - FSP Unigram initialized from BPE (requires --overshoot-factor)
  mingram       - MinGram hard-EM on minimum-path lattice (requires --overshoot-factor)

Designed to be called once per (method, corpus, factor) job so all runs can be
dispatched in parallel by run_all_experiments.sh.
"""

import argparse
import os
import time

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from script_bpe.utils import create_logger

from paper_utils.hybrid.train_hybrid import (
    PRETOKENIZER_NAME,
    bpe_to_unigram_tokens,
    get_bpe_model_path,
    get_model_path as get_hybrid_model_path,
    get_or_train_bpe,
)
from paper_utils.hybrid.train_mingram import (
    get_model_path as get_mingram_model_path,
)
from paper_utils.unigram.train_hyperparameters import (
    ADDITIONAL_VOCAB_SIZE,
    DEFAULTS,
    USE_CACHE,
    get_model_path as get_unigram_model_path,
)

_FSP_OVERRIDES: dict = {"flat_score_prune": True, "pre_final_vocab_factor": 1.0}

_UNIGRAM_PARAMS: dict[str, dict] = {
    "default":      DEFAULTS,
    "fsp":          {**DEFAULTS, **_FSP_OVERRIDES},
    "bpe_init":     DEFAULTS,
    "bpe_init_fsp": {**DEFAULTS, **_FSP_OVERRIDES},
}


def get_model_path(
    method: str,
    corpus: str,
    overshoot_factor: float | None,
    num_em: int,
    pruning: float,
    additional_vocab_size: int,
    prune_criterion: str = "usage_count",
    skip_substring: bool = False,
):
    if method == "bpe":
        return get_bpe_model_path(corpus, additional_vocab_size)
    if method == "mingram":
        return get_mingram_model_path(
            corpus, overshoot_factor, num_em, pruning, additional_vocab_size, prune_criterion, skip_substring
        )
    if method in ("bpe_init", "bpe_init_fsp"):
        params = {**_UNIGRAM_PARAMS[method], "overshoot_factor": overshoot_factor}
        if additional_vocab_size != ADDITIONAL_VOCAB_SIZE:
            params["additional_vocab_size"] = additional_vocab_size
        return get_hybrid_model_path(corpus, params)
    params = dict(_UNIGRAM_PARAMS[method])
    if additional_vocab_size != ADDITIONAL_VOCAB_SIZE:
        params["additional_vocab_size"] = additional_vocab_size
    return get_unigram_model_path(corpus, params)


def train(
    method: str,
    corpus: str,
    overshoot_factor: float | None,
    num_em: int,
    pruning: float,
    additional_vocab_size: int,
    num_workers: int,
    prune_criterion: str = "usage_count",
    skip_substring: bool = False,
) -> None:
    logger = create_logger("train_model", verbose=True)
    model_path = get_model_path(
        method, corpus, overshoot_factor, num_em, pruning, additional_vocab_size, prune_criterion, skip_substring
    )

    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    corpus_data = load_corpus_by_name(corpus, pretokenizer)
    t0 = time.perf_counter()

    if method == "bpe":
        existed_before = USE_CACHE and model_path.exists()
        bpe_model = get_or_train_bpe(
            corpus,
            additional_vocab_size,
            pretokenizer,
            corpus_data,
            logger,
            num_workers=num_workers,
        )
        model = bpe_model
        if existed_before:
            model = BPETokenizer.load(str(model_path))
        model.metadata["corpus"] = corpus
        if not existed_before or "performance" not in model.metadata:
            model.metadata["time"] = time.perf_counter() - t0
            model.metadata["performance"] = model.corpus_performance(corpus_data)
            model.save(str(model_path))
            logger.info(f"Saved: {model_path}")
        else:
            logger.info(f"Cached: {model_path}")
        return

    if USE_CACHE and model_path.exists():
        logger.info(f"Cached: {model_path}")
        return

    if method == "mingram":
        cfg = MinGramTrainerConfig(
            additional_vocab_size=additional_vocab_size,
            overshoot_factor=overshoot_factor,
            num_em_iterations=num_em,
            pruning_shrinking_factor=pruning,
            num_workers=num_workers,
            prune_criterion=prune_criterion,
            skip_substring_in_batch=skip_substring,
        )
        model = MinGramTrainer(pretokenizer, corpus_data, cfg).train()
        model.metadata["corpus"] = corpus
        model.metadata["time"] = time.perf_counter() - t0
        model.metadata["performance"] = model.corpus_performance(corpus_data)
    else:
        params = dict(_UNIGRAM_PARAMS[method])
        cfg = UnigramTrainerConfig(additional_vocab_size=additional_vocab_size, num_workers=num_workers, **params)
        if overshoot_factor is not None:
            bpe_vocab_size = int(additional_vocab_size * overshoot_factor)
            bpe_model = get_or_train_bpe(
                corpus,
                bpe_vocab_size,
                pretokenizer,
                corpus_data,
                logger,
                num_workers=num_workers,
            )
            cfg.forced_initial_vocab = bpe_to_unigram_tokens(bpe_model)
        model = UnigramTrainer(pretokenizer, corpus_data, cfg).train()
        model.metadata["corpus"] = corpus
        model.metadata["overshoot_factor"] = overshoot_factor
        model.metadata["time"] = time.perf_counter() - t0
        model.metadata["performance"] = model.corpus_performance(corpus_data)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info(f"Saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train one paper tokenizer model.")
    parser.add_argument("--method", required=True,
                        choices=["bpe", "default", "fsp", "bpe_init", "bpe_init_fsp", "mingram"])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--overshoot-factor", type=float, default=None)
    parser.add_argument("--num-em-iterations", type=int, default=2)
    parser.add_argument("--pruning-shrinking-factor", type=float, default=0.0)
    parser.add_argument("--additional-vocab-size", type=int, default=ADDITIONAL_VOCAB_SIZE)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("SCRIPT_BPE_TRAIN_NUM_WORKERS", "4")))
    parser.add_argument("--prune-criterion", choices=["usage_count", "mi"], default="usage_count",
                        help="mingram only: 'mi' = careful Minimum-Increase prune (vs default usage-count)")
    parser.add_argument("--skip-substring", action="store_true",
                        help="mingram mi only: Craig's rule -- skip dropping a token that is a substring of an already-dropped one in the batch")
    args = parser.parse_args()

    if args.method in ("bpe_init", "bpe_init_fsp", "mingram") and args.overshoot_factor is None:
        parser.error(f"--overshoot-factor is required for --method {args.method}")

    train(
        args.method,
        args.corpus,
        args.overshoot_factor,
        args.num_em_iterations,
        args.pruning_shrinking_factor,
        args.additional_vocab_size,
        args.num_workers,
        args.prune_criterion,
        args.skip_substring,
    )


if __name__ == "__main__":
    main()
