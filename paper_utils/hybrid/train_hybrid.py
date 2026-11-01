"""BPE-init utilities and model-path helpers for hybrid tokenizer experiments.

These helpers convert a trained BPE vocabulary into Unigram-style seed tokens
(used to initialize MinGram / BPE-init Unigram training) and resolve the cache
paths for hybrid models. The actual training entrypoints live in train_model.py.
"""

import math
from pathlib import Path

from script_bpe.analysis import get_config_hash
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig

# Import shared settings from unigram
from paper_utils.unigram.train_hyperparameters import (
    PRETOKENIZER_NAME,
    ADDITIONAL_VOCAB_SIZE,
    USE_CACHE,
)

# ========== HYBRID-SPECIFIC SETTINGS ==========

RESULTS_DIR = Path("results/hybrid")

# BPE-init experiment: use BPE vocabulary to initialize Unigram training.
# Reduced set covering key points on the tradeoff curve
OVERSHOOT_FACTORS = [
    1.0,   # Baseline: BPE vocab == final vocab
    1.05,  # Smaller overshoot
    1.1,   # Best compression point
    1.15,
    1.25,  # Slightly less compression, better objective
    1.5,   # Midpoint
    2.0,   # Double-sized BPE vocab
    3.0,   # Large factor
    5.0,   # Very large factor
]


# ========== MODEL PATHS ==========


def get_model_path(corpus_name: str, params: dict) -> Path:
    """Get the file path for a hybrid model with given corpus and parameters."""
    params = dict(params)
    if "overshoot_factor" in params:
        params["bpe_init_factor"] = params.pop("overshoot_factor")
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
    else:
        prefix = params.get('init_vocab_algo', 'corpus_long')
    return cache_dir / f"{prefix}_n{vocab_size}_{config_hash}.model.json.gz"


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


def get_bpe_model_path(corpus_name: str, bpe_vocab_size: int) -> Path:
    return RESULTS_DIR / corpus_name / f"bpe_n{bpe_vocab_size}.model.json.gz"


def get_or_train_bpe(
    corpus_name: str,
    bpe_vocab_size: int,
    pretokenizer,
    corpus,
    logger,
    num_workers: int = 4,
) -> BPETokenizer:
    """Get BPE model from cache or train it."""
    bpe_path = get_bpe_model_path(corpus_name, bpe_vocab_size)

    if USE_CACHE and bpe_path.exists():
        logger.info(f"Loading cached BPE model: {bpe_path}")
        return BPETokenizer.load(str(bpe_path))

    logger.info(f"Training BPE with vocab size {bpe_vocab_size} using {num_workers} workers")
    bpe_cfg = BPETrainerConfig(additional_vocab_size=bpe_vocab_size, num_workers=num_workers)
    bpe_model = BPETrainer(pretokenizer, corpus, bpe_cfg).train()
    bpe_model.metadata["corpus"] = corpus_name
    bpe_path.parent.mkdir(parents=True, exist_ok=True)
    bpe_model.save(str(bpe_path))
    logger.info(f"Saved BPE model to {bpe_path}")
    return bpe_model
