"""Unigram hyperparameter training experiments.

This module contains:
- Experiment configuration (CORPUS_NAMES, DEFAULTS, SWEEP_CONFIGS)
- Model path/cache management
- Training runners
"""

import time
from pathlib import Path

from tabulate import tabulate
import pandas as pd
from script_bpe import get_pretokenizer
from script_bpe.analysis import get_config_hash
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.utils import create_logger
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.unigram import UnigramModel

from paper_utils.unigram.utils import (
    PRETOKENIZER_NAME,
    RESULTS_DIR,
    load_experiment_results as _load_experiment_results,
    identify_baseline as _identify_baseline,
    load_baseline_model as _load_baseline_model,
)

# ========== FIXED SETTINGS ==========
CORPUS_NAMES = [
    "eng_latn_300mb",
    "deu_latn_300mb",
    "arb_arab_300mb",
    "hin_deva_300mb",
    "kor_hang_300mb",
    "zho_hans_300mb",
]

FINEWIKI_CORPUS_NAMES = [
    "finewiki_en_1gb",
    "finewiki_de_1gb",
    "finewiki_ar_1gb",
    "finewiki_hi_1gb",
    "finewiki_ko_1gb",
    "finewiki_zh_1gb",
]

ADDITIONAL_VOCAB_SIZE = 32768
USE_CACHE = True

# Hyperparameter defaults (match UnigramTrainerConfig defaults)
DEFAULTS = {
    "init_vocab_algo": "corpus_long",
    "initial_vocab_factor": 10,
    "max_token_len": 32,
    "defensive_prune": False,
    "num_sub_iterations": 2,
    "m_step_digamma": True,
    "m_step_low_count_threshold": 0.5,
    "pre_final_vocab_factor": 1.1,
    "pruning_shrinking_factor": 0.75,
    "final_style_prune": False,
}

# Hyperparameter sweep configurations (each sweeps one parameter)
SWEEP_CONFIGS = {
    "initial_vocab_factor": [3, 10, 25, 50, 100],
    "m_step_digamma": [False, True],
    "m_step_low_count_threshold": [0.0, 0.5, 2.0, 10.0],
    "num_sub_iterations": [1, 2, 3, 5],
    "pre_final_vocab_factor": [1.0, 1.1, 1.25, 1.5, 2.0],
    "pruning_shrinking_factor": [0.5, 0.75, 0.9, 0.95],
    "additional_vocab_size": [fac * ADDITIONAL_VOCAB_SIZE for fac in [0.5, 1, 2, 4, 8]],
}

# ========== EXPERIMENT CONFIGURATION HELPERS ==========


def get_experiment_configs(experiment_name: str) -> list[dict]:
    """Get all valid parameter configurations for an experiment.

    Returns list of complete parameter dicts (including defaults).
    The baseline (all defaults) is included for regular sweeps.
    """
    # Regular parameter sweeps
    if experiment_name in SWEEP_CONFIGS:
        return [{**DEFAULTS, experiment_name: value} for value in SWEEP_CONFIGS[experiment_name]]

    if experiment_name in ["fsp", "fsp_vocab"]:
        overrides = dict(final_style_prune=True, pre_final_vocab_factor=1.0)
        if experiment_name == "fsp":
            return [
                {**DEFAULTS, **overrides, "pruning_shrinking_factor": val} for val in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
            ]
        if experiment_name == "fsp_vocab":
            return [
                {**DEFAULTS, **overrides, "additional_vocab_size": vocab_size}
                for vocab_size in SWEEP_CONFIGS["additional_vocab_size"]
            ]

    if experiment_name == "init_algo":
        algos = ["corpus_long_no_pt", "corpus_fallback_no_pt", "corpus_long", "corpus_fallback"]
        return [{**DEFAULTS, "init_vocab_algo": algo} for algo in algos]

    raise ValueError(f"Unknown experiment: {experiment_name}")


def get_model_path(corpus_name: str, params: dict) -> Path:
    """Get the file path for a model with given corpus and parameters."""
    vocab_size = params.get("additional_vocab_size", ADDITIONAL_VOCAB_SIZE)
    cache_dir = Path("results") / "unigram_sweeps" / corpus_name
    config_hash = get_config_hash(
        {
            "corpus": corpus_name,
            "pretokenizer": PRETOKENIZER_NAME,
            "n": vocab_size,
            **params,
        }
    )
    prefix = params.get("init_vocab_algo", "corpus_long")
    return cache_dir / f"{prefix}_n{vocab_size}_{config_hash}.model.json.gz"


def load_model_if_cached(corpus_name: str, params: dict) -> tuple[UnigramModel | None, Path]:
    """Load model from cache if it exists, otherwise return None and path.

    Returns:
        (model, path) if cached
        (None, path) if needs training
    """
    model_path = get_model_path(corpus_name, params)
    if USE_CACHE and model_path.exists():
        model = UnigramModel.load(str(model_path))
        return model, model_path

    return None, model_path


# ========== ANALYSIS UTILITIES (wrappers for utils.py) ==========


def load_experiment_results(
    experiment_keys: str | list[str],
    corpus_names: list[str] | None = None,
) -> pd.DataFrame:
    """Load experiment results based on experiment configuration."""
    if corpus_names is None:
        corpus_names = CORPUS_NAMES
    return _load_experiment_results(
        get_experiment_configs,
        load_model_if_cached,
        experiment_keys,
        corpus_names,
    )


def identify_baseline(df: pd.DataFrame, corpus_name: str) -> pd.Series | None:
    """Identify baseline configuration for a corpus (all defaults)."""
    return _identify_baseline(df, corpus_name, DEFAULTS)


def load_baseline_model(corpus_name: str) -> pd.Series | None:
    """Load baseline model (all defaults) for a corpus."""
    return _load_baseline_model(corpus_name, DEFAULTS, load_model_if_cached)


# ========== TRAINING UTILITIES ==========


def _train_and_evaluate(corpus_name: str, logger=None, **params) -> UnigramModel:
    if logger is None:
        logger = create_logger("train_eval", verbose=True)

    logger.info(f"Loading pretokenizer: {PRETOKENIZER_NAME}")
    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)

    logger.info(f"Loading corpus: {corpus_name}")
    corpus = load_corpus_by_name(corpus_name, pretokenizer)

    config_params = dict(additional_vocab_size=ADDITIONAL_VOCAB_SIZE)
    config_params.update({k: v for k, v in params.items() if v is not None})
    cfg = UnigramTrainerConfig(**config_params)

    # Handle _no_pt init algorithms (compare init without pretokenizer)
    if params.get("init_vocab_algo", "").endswith("_no_pt"):
        logger.info("Loading nosplit pretokenizer for _no_pt mode")
        ns_pretokenizer = get_pretokenizer("scriptenc_nosplit_cb")
        corpus = load_corpus_by_name(corpus_name, ns_pretokenizer)
        cfg.init_vocab_algo = (params["init_vocab_algo"], corpus)

    logger.info(f"Starting training with init_vocab_algo={params.get('init_vocab_algo', 'default')}")

    t0 = time.perf_counter()
    model = UnigramTrainer(pretokenizer, corpus, cfg).train()

    logger.info("Computing performance metrics")
    model.metadata["corpus"] = corpus_name
    model.metadata["time"] = time.perf_counter() - t0
    model.metadata["performance"] = model.corpus_performance(corpus)

    return model


def run_experiment(experiment_name: str, corpus_names: list[str]) -> list[dict]:
    """Run training experiments for an experiment."""
    logger = create_logger("experiment", verbose=True)

    configs = get_experiment_configs(experiment_name)
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
                model = _train_and_evaluate(corpus_name, logger=logger, **params)
                model.save(str(model_path))
                logger.info(f"📄 Saved to {model_path.name}")
                results.append(model.metadata)
            except Exception:
                logger.error(f"❌ {corpus_name}", exc_info=True)
                raise

    return sorted(results, key=lambda r: (r["corpus"], r.get("objective", 0)))


def run_bpe_baseline(corpora: list[str]) -> list[dict]:
    """Train BPE baseline tokenizers for comparison."""
    logger = create_logger("experiment", verbose=True)
    results = []

    for corpus_name in corpora:
        for vocab_size in SWEEP_CONFIGS["additional_vocab_size"]:
            cache_dir = RESULTS_DIR / corpus_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_file = cache_dir / f"bpe_n{vocab_size}.model.json.gz"

            if USE_CACHE and model_file.exists():
                logger.info(f"💾 {corpus_name}:bpe from {model_file}")
                tokenizer = BPETokenizer.load(str(model_file))
                results.append(tokenizer.metadata)
                continue

            logger.info(f"▶️  {corpus_name}:bpe")
            try:
                pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
                corpus = load_corpus_by_name(corpus_name, pretokenizer)

                cfg = BPETrainerConfig(additional_vocab_size=vocab_size, num_workers=4)
                t0 = time.perf_counter()
                tokenizer = BPETrainer(pretokenizer, corpus, cfg).train()

                tokenizer.metadata["corpus"] = corpus_name
                tokenizer.metadata["time"] = time.perf_counter() - t0
                tokenizer.metadata["performance"] = tokenizer.corpus_performance(corpus)
                tokenizer.save(str(model_file))
                logger.info(f"📄 Saved model to {model_file.name}")
                results.append(tokenizer.metadata)
            except Exception:
                logger.error(f"❌ {corpus_name}:bpe", exc_info=True)
                raise

    return sorted(results, key=lambda r: r["corpus"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train unigram hyperparameters")
    parser.add_argument("experiment", help="Experiment name (e.g., init_algo, bpe, all)")
    parser.add_argument(
        "--corpus-filter",
        nargs="+",
        help="Filter corpus names by these substrings (e.g., deu hin)",
    )
    parser.add_argument("--finewiki", action="store_true", help="Run on finewiki corpora")
    args = parser.parse_args()

    if args.finewiki:
        corpus_names = FINEWIKI_CORPUS_NAMES
    else:
        corpus_names = CORPUS_NAMES

    def apply_filter(names: list[str]) -> list[str]:
        if not args.corpus_filter:
            return names
        return [name for name in names if any(f in name for f in args.corpus_filter)]

    # BPE baseline
    if args.experiment in ["bpe", "all"]:
        active_corpora = apply_filter(corpus_names)
        if active_corpora:
            results = run_bpe_baseline(active_corpora)
            print("\n=== BPE BASELINE ===")
            print(tabulate(results, headers="keys", tablefmt="grid"))

    # Regular parameter sweeps
    for sweep_param in SWEEP_CONFIGS.keys():
        if args.experiment in [sweep_param, "all"]:
            active_corpora = apply_filter(corpus_names)
            if active_corpora:
                results = run_experiment(sweep_param, active_corpora)
                print(f"\n=== {sweep_param.upper().replace('_', ' ')} ===")
                print(tabulate(results, headers="keys", tablefmt="grid"))

    if args.experiment in ["init_algo", "all"]:
        active_corpora = apply_filter(corpus_names)
        if active_corpora:
            active_corpora = ["smol_" + c for c in active_corpora] + active_corpora
            results = run_experiment("init_algo", active_corpora)
            print("\n=== INIT ALGO ===")
            print(tabulate(results, headers="keys", tablefmt="grid"))

    if args.experiment.startswith("fsp") or args.experiment == "all":
        active_corpora = apply_filter(corpus_names)
        if active_corpora:
            results = run_experiment(args.experiment, active_corpora)
            print(f"\n=== {args.experiment} ===")
            print(tabulate(results, headers="keys", tablefmt="grid"))
