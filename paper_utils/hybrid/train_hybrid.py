"""Hybrid tokenizer training experiments (BPE-init, token bias).

This module contains training logic for hybrid methods that combine
BPE and Unigram approaches.
"""

import math
import time
from pathlib import Path

from tabulate import tabulate

from script_bpe import get_pretokenizer
from script_bpe.analysis import get_config_hash, flatten_model_metadata
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.utils import create_logger
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.unigram import UnigramModel

# Import shared settings from unigram
from paper_utils.unigram.train_hyperparameters import (
    CORPUS_NAMES,
    DEFAULTS,
    PRETOKENIZER_NAME,
    ADDITIONAL_VOCAB_SIZE,
    USE_CACHE,
)

# ========== HYBRID-SPECIFIC SETTINGS ==========

RESULTS_DIR = Path("results/hybrid")

# BPE-init experiment: use BPE vocabulary to initialize Unigram training
BPE_INIT_FACTORS = [1.0, 1.1, 1.25, 1.5, 2.0]

# Token bias sweep values
TOKEN_BIAS_VALUES = [-1, 0.0, 1.0, 2.0, 5.0]


# ========== EXPERIMENT CONFIGURATION ==========


def get_hybrid_experiment_configs(experiment_name: str) -> list[dict]:
    """Get parameter configurations for hybrid experiments.

    Supported experiments:
    - token_bias: Sweep token_bias values
    - bpe_init: BPE-initialized Unigram with different init factors
    - bpe_init_fsp: BPE-init with final style pruning
    - token_bias_fsp: Token bias with FSP
    """
    if experiment_name == "token_bias":
        return [{**DEFAULTS, "token_bias": b} for b in TOKEN_BIAS_VALUES]

    if experiment_name == "bpe_init":
        return [{**DEFAULTS, "bpe_init_factor": f} for f in BPE_INIT_FACTORS]

    if experiment_name == "bpe_init_fsp":
        overrides = dict(final_style_prune=True, pre_final_vocab_factor=1.0)
        return [{**DEFAULTS, **overrides, "bpe_init_factor": f} for f in BPE_INIT_FACTORS]

    if experiment_name == "token_bias_fsp":
        overrides = dict(final_style_prune=True, pre_final_vocab_factor=1.0)
        return [{**DEFAULTS, **overrides, "token_bias": b} for b in TOKEN_BIAS_VALUES]

    raise ValueError(f"Unknown hybrid experiment: {experiment_name}")


def get_model_path(corpus_name: str, params: dict) -> Path:
    """Get the file path for a hybrid model with given corpus and parameters."""
    vocab_size = params.get("additional_vocab_size", ADDITIONAL_VOCAB_SIZE)
    cache_dir = RESULTS_DIR / corpus_name
    config_hash = get_config_hash(
        {
            "corpus": corpus_name,
            "pretokenizer": PRETOKENIZER_NAME,
            "n": vocab_size,
            **params,
        }
    )
    # Use bpe_init prefix for BPE-initialized models
    if "bpe_init_factor" in params:
        prefix = f"bpe_init_{params['bpe_init_factor']}"
    elif "token_bias" in params:
        prefix = f"token_bias_{params['token_bias']}"
    else:
        prefix = params.get('init_vocab_algo', 'corpus_long')
    return cache_dir / f"{prefix}_n{vocab_size}_{config_hash}.model.json.gz"


def load_model_if_cached(corpus_name: str, params: dict) -> tuple[UnigramModel | None, Path]:
    """Load model from cache if it exists, otherwise return None and path."""
    model_path = get_model_path(corpus_name, params)
    if USE_CACHE and model_path.exists():
        model = UnigramModel.load(str(model_path))
        return model, model_path
    return None, model_path


# ========== BPE-INIT UTILITIES ==========


def bpe_to_unigram_tokens(bpe_model: BPETokenizer) -> list[UnigramToken]:
    """Convert BPE tokens to UnigramTokens using BPE's final counts for log_probs."""
    from script_bpe.utils import token_array

    tokens = []
    # Use current_count from BPE (reflects corpus frequency after merges)
    total_count = sum(max(1, t.current_count) for t in bpe_model.tokens.values())
    for i, bpe_token in enumerate(bpe_model.tokens.values()):
        count = max(1, bpe_token.current_count)  # avoid log(0)
        log_prob = math.log(count / total_count)
        tokens.append(UnigramToken(
            id=i,
            atomic_tokens=token_array(bpe_token.atomic_tokens),
            log_prob=log_prob,
            required=len(bpe_token.atomic_tokens) == 1,
        ))
    return tokens


def get_or_train_bpe(corpus_name: str, bpe_vocab_size: int, pretokenizer, corpus, logger) -> BPETokenizer:
    """Get BPE model from cache or train it."""
    bpe_path = RESULTS_DIR / corpus_name / f"bpe_n{bpe_vocab_size}.model.json.gz"

    if USE_CACHE and bpe_path.exists():
        logger.info(f"Loading cached BPE model: {bpe_path}")
        return BPETokenizer.load(str(bpe_path))

    logger.info(f"Training BPE with vocab size {bpe_vocab_size}")
    bpe_cfg = BPETrainerConfig(additional_vocab_size=bpe_vocab_size, num_workers=4)
    bpe_model = BPETrainer(pretokenizer, corpus, bpe_cfg).train()
    bpe_model.metadata["corpus"] = corpus_name
    bpe_path.parent.mkdir(parents=True, exist_ok=True)
    bpe_model.save(str(bpe_path))
    logger.info(f"Saved BPE model to {bpe_path}")
    return bpe_model


# ========== TRAINING ==========


def train_hybrid_model(corpus_name: str, logger=None, **params) -> UnigramModel:
    """Train a hybrid Unigram model with BPE-init or token bias."""
    if logger is None:
        logger = create_logger("train_hybrid", verbose=True)

    logger.info(f"Loading pretokenizer: {PRETOKENIZER_NAME}")
    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)

    logger.info(f"Loading corpus: {corpus_name}")
    corpus = load_corpus_by_name(corpus_name, pretokenizer)

    # Filter out bpe_init_factor from params before passing to UnigramTrainerConfig
    bpe_init_factor = params.pop("bpe_init_factor", None)

    config_params = dict(additional_vocab_size=ADDITIONAL_VOCAB_SIZE)
    config_params.update({k: v for k, v in params.items() if v is not None})
    cfg = UnigramTrainerConfig(**config_params)

    # Handle BPE initialization
    if bpe_init_factor is not None:
        bpe_vocab_size = int(ADDITIONAL_VOCAB_SIZE * bpe_init_factor)
        logger.info(f"BPE-init mode: factor={bpe_init_factor}, BPE vocab size={bpe_vocab_size}")
        bpe_model = get_or_train_bpe(corpus_name, bpe_vocab_size, pretokenizer, corpus, logger)
        unigram_tokens = bpe_to_unigram_tokens(bpe_model)
        cfg.forced_initial_vocab = unigram_tokens
        logger.info(f"Initialized Unigram with {len(unigram_tokens)} BPE tokens")

    init_desc = f"bpe_init_factor={bpe_init_factor}" if bpe_init_factor else f"token_bias={params.get('token_bias', 'default')}"
    logger.info(f"Starting training with {init_desc}")

    t0 = time.perf_counter()
    model = UnigramTrainer(pretokenizer, corpus, cfg).train()

    logger.info("Computing performance metrics")
    model.metadata["corpus"] = corpus_name
    model.metadata["bpe_init_factor"] = bpe_init_factor
    model.metadata["time"] = time.perf_counter() - t0
    model.metadata["performance"] = model.corpus_performance(corpus)

    return model


def run_experiment(experiment_name: str, corpus_names: list[str]) -> list[dict]:
    """Run hybrid training experiments."""
    logger = create_logger("hybrid_experiment", verbose=True)

    configs = get_hybrid_experiment_configs(experiment_name)
    results = []

    for corpus_name in corpus_names:
        cache_dir = RESULTS_DIR / corpus_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        for params in configs:
            model, model_path = load_model_if_cached(corpus_name, params)

            if model is not None:
                logger.info(f"💾 {corpus_name} from {model_path.name}")
                results.append(model.metadata)
                continue

            logger.info(f"▶️  {corpus_name}")
            try:
                model = train_hybrid_model(corpus_name, logger=logger, **params)
                model.save(str(model_path))
                logger.info(f"📄 Saved to {model_path.name}")
                results.append(model.metadata)
            except Exception:
                logger.error(f"❌ {corpus_name}", exc_info=True)
                raise

    return sorted(results, key=lambda r: (r["corpus"], r.get("objective", 0)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train hybrid tokenizers (BPE-init, token bias)")
    parser.add_argument(
        "experiment",
        choices=["bpe_init", "bpe_init_fsp", "token_bias", "token_bias_fsp", "all"],
        help="Experiment name",
    )
    parser.add_argument(
        "--corpus-filter",
        nargs="+",
        help="Filter corpus names by these substrings (e.g., deu hin)",
    )
    args = parser.parse_args()

    def apply_filter(names: list[str]) -> list[str]:
        if not args.corpus_filter:
            return names
        return [name for name in names if any(f in name for f in args.corpus_filter)]

    experiments = (
        ["bpe_init", "bpe_init_fsp", "token_bias", "token_bias_fsp"]
        if args.experiment == "all"
        else [args.experiment]
    )

    for exp in experiments:
        active_corpora = apply_filter(CORPUS_NAMES)
        if active_corpora:
            results = run_experiment(exp, active_corpora)
            print(f"\n=== {exp.upper().replace('_', ' ')} ===")
            print(tabulate(results, headers="keys", tablefmt="grid"))

