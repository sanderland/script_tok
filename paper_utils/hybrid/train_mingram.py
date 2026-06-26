"""MinGram model path and results directory (importable by analysis scripts)."""

from pathlib import Path

from script_bpe.analysis import get_config_hash

from paper_utils.unigram.train_hyperparameters import ADDITIONAL_VOCAB_SIZE

PRETOKENIZER_NAME = "scriptenc_cb"
RESULTS_DIR = Path("results/mingram")


def get_model_path(
    corpus_name: str,
    overshoot_factor: float,
    num_em_iterations: int,
    pruning_shrinking_factor: float,
    additional_vocab_size: int = ADDITIONAL_VOCAB_SIZE,
    prune_criterion: str = "usage_count",
    skip_substring_in_batch: bool = False,
) -> Path:
    hash_dict = {
        "model": "mingram",
        "corpus": corpus_name,
        "pretokenizer": PRETOKENIZER_NAME,
        "n": additional_vocab_size,
        "bpe_init_factor": overshoot_factor,
        "num_em_iterations": num_em_iterations,
        "pruning_shrinking_factor": pruning_shrinking_factor,
    }
    prefix = f"mingram_f{overshoot_factor}_em{num_em_iterations}_p{pruning_shrinking_factor}"
    # Default (usage_count, no skip) keeps the historical hash/filename bit-identical; the
    # MinGram-PP and substring-skip variants get distinct tags so caches don't collide.
    if prune_criterion != "usage_count":
        hash_dict["prune_criterion"] = prune_criterion
        prefix += f"_pc{prune_criterion}"
    if skip_substring_in_batch:
        hash_dict["skip_substring_in_batch"] = True
        prefix += "_ss"
    config_hash = get_config_hash(hash_dict)
    return RESULTS_DIR / corpus_name / f"{prefix}_n{additional_vocab_size}_{config_hash}.model.json.gz"


