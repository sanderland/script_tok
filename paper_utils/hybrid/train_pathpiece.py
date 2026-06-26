"""PathPiece model path + driver for the hybrid main-table comparison.

PathPiece (Schmidt et al., 2024) is added as a min-token-objective
baseline alongside MinGram. We expose ``get_model_path`` and ``RESULTS_DIR``
so that the existing table/scatter generators can locate PathPiece
artifacts by reading the model directory, mirroring the
``train_mingram`` interface.
"""

from __future__ import annotations

from pathlib import Path

from script_bpe.analysis import get_config_hash

from paper_utils.unigram.train_hyperparameters import ADDITIONAL_VOCAB_SIZE

PRETOKENIZER_NAME = "scriptenc_cb"
RESULTS_DIR = Path("results/pathpiece")

# Token-length cap: dropped. The ablation in results/pathpiece/ablations/
# showed L=32 atomic and effectively-unbounded (L=1024) give identical
# compression for f>=2 (e.g. English f=2: -2.15% vs -2.14%), so the cap is
# an unnecessary hyperparameter. We keep a large finite L=1024 only to bound
# the DP and to drop pathological tokens (e.g. 2000-char dash spam from BPE);
# it never binds for real language tokens.
#
# Seed budget: PathPiece-B saturates at f~=2 (init ~= 2x target); we keep the
# paper's 2^18 budget for headroom.
MAIN_BPE_INIT_VOCAB = 262_144  # 2^18, paper-faithful for BPE-init
MAIN_MAX_TOKEN_WIDTH = 1024  # effectively unbounded (no L cap)
MAIN_PRUNE_BATCH_FRACTION = 0.10  # canonical: matches the trainer/PathPiece default.

PATHPIECE_INIT = "bpe"

# Legacy alias for callers that don't care about per-init defaults.
MAIN_INIT_VOCAB = MAIN_BPE_INIT_VOCAB


def _default_init_vocab(init: str) -> int:
    if init != PATHPIECE_INIT:
        raise ValueError(f"Unknown PathPiece init {init!r}; expected {PATHPIECE_INIT!r}")
    return MAIN_BPE_INIT_VOCAB


def get_model_path(
    corpus_name: str,
    init: str,
    init_vocab_size: int | None = None,
    max_token_width: int = MAIN_MAX_TOKEN_WIDTH,
    prune_batch_fraction: float = MAIN_PRUNE_BATCH_FRACTION,
    additional_vocab_size: int = ADDITIONAL_VOCAB_SIZE,
    skip_substring_in_batch: bool = False,
) -> Path:
    if init_vocab_size is None:
        init_vocab_size = _default_init_vocab(init)
    elif init != PATHPIECE_INIT:
        raise ValueError(f"Unknown PathPiece init {init!r}; expected {PATHPIECE_INIT!r}")
    hash_dict = {
        "model": "pathpiece",
        "corpus": corpus_name,
        "pretokenizer": PRETOKENIZER_NAME,
        "n": additional_vocab_size,
        "init": init,
        "init_vocab_size": init_vocab_size,
        "max_token_width": max_token_width,
        "prune_batch_fraction": prune_batch_fraction,
    }
    prefix = f"pathpiece_{init}_iv{init_vocab_size}_L{max_token_width}_pb{prune_batch_fraction}"
    # default (no skip) keeps the historical filename/hash; substring-skip gets an "_ss" tag.
    if skip_substring_in_batch:
        hash_dict["skip_substring_in_batch"] = True
        prefix += "_ss"
    config_hash = get_config_hash(hash_dict)
    return RESULTS_DIR / corpus_name / f"{prefix}_n{additional_vocab_size}_{config_hash}.model.json.gz"
