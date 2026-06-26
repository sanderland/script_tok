"""Shared utilities for unigram paper experiments.

Includes:
- Corpus mappings and constants
- Unified JSON caching
- Cached evaluation wrappers (FineWiki, MorphScore)
- Experiment analysis utilities
"""

import json
from pathlib import Path

import pandas as pd

from script_bpe.analysis import evaluate_on_corpus, flatten_model_metadata
from script_bpe.analysis.morphscore import MorphScore
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.unigram import UnigramModel

# ========== PRETOKENIZER & RESULTS CONFIG ==========

PRETOKENIZER_NAME = "scriptenc_cb"
NOSPLIT_PRETOKENIZER_NAME = "scriptenc_nosplit_cb"

RESULTS_DIR = Path("results/unigram_sweeps") / PRETOKENIZER_NAME

# ========== CORPUS MAPPINGS ==========

# Mapping from training corpus to FineWiki evaluation corpus
FINEWIKI_MAP = {
    "eng_latn_300mb": "finewiki_en_1gb",
    "deu_latn_300mb": "finewiki_de_1gb",
    "fin_latn_300mb": "finewiki_fi_1gb",
    "hun_latn_300mb": "finewiki_hu_1gb",
    "arb_arab_300mb": "finewiki_ar_1gb",
    "hin_deva_300mb": "finewiki_hi_1gb",
    "kor_hang_300mb": "finewiki_ko_1gb",
    "rus_cyrl_300mb": "finewiki_ru_1gb",
    "zho_hans_300mb": "finewiki_zh_1gb",
}

FINEWIKI_REVERSE_MAP = {v: k for k, v in FINEWIKI_MAP.items()}

# Language display names
LANG_NAMES = {
    "eng_latn_300mb": "English",
    "deu_latn_300mb": "German",
    "arb_arab_300mb": "Arabic",
    "hin_deva_300mb": "Hindi",
    "kor_hang_300mb": "Korean",
    "zho_hans_300mb": "Chinese",
}


def get_finewiki_corpus_name(corpus_name: str) -> str | None:
    """Get the FineWiki corpus name corresponding to a training corpus."""
    return FINEWIKI_MAP.get(corpus_name)


def get_300mb_corpus_name(finewiki_name: str) -> str:
    """Get the 300MB corpus name corresponding to a FineWiki corpus."""
    return FINEWIKI_REVERSE_MAP[finewiki_name]


# ========== UNIFIED JSON CACHE ==========

CACHE_DIR = RESULTS_DIR


class JsonCache:
    """Simple JSON file cache for evaluation results."""

    def __init__(self, name: str):
        self.path = CACHE_DIR / f"cache_{name}.json"
        self._cache: dict | None = None

    def _load(self) -> dict:
        if self._cache is None:
            if self.path.exists():
                with open(self.path) as f:
                    self._cache = json.load(f)
            else:
                self._cache = {}
        return self._cache

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, key: str) -> dict | None:
        return self._load().get(key)

    def set(self, key: str, value: dict):
        self._load()[key] = value
        self._save()


# Cache instances
_finewiki_cache = JsonCache("finewiki")
_morphscore_cache = JsonCache("morphscore")


# ========== CACHED EVALUATION FUNCTIONS ==========


def evaluate_on_corpus_cached(model, corpus_name: str, model_path: str) -> dict:
    """Cached version of evaluate_on_corpus."""
    key = f"{model_path}:{corpus_name}"
    cached = _finewiki_cache.get(key)
    if cached is not None:
        return cached
    print(f"  [cache miss] evaluate_on_corpus: {corpus_name} for {model_path}")
    result = evaluate_on_corpus(model, corpus_name)
    _finewiki_cache.set(key, result)
    return result


def evaluate_morphscore_cached(ms: MorphScore, model, lang_code: str, model_path: str) -> dict:
    """Cached MorphScore evaluation for a single language."""
    key = f"{model_path}:{lang_code}"
    cached = _morphscore_cache.get(key)
    if cached is not None:
        return cached
    print(f"  [cache miss] evaluate_morphscore: {lang_code} for {model_path}")
    ms.update_config(language_subset=[lang_code])
    result = ms.eval(model)[lang_code]
    _morphscore_cache.set(key, result)
    return result


# ========== MODEL LOADING UTILITIES ==========


def load_vocab_from_model_file(model_path: str | Path) -> set[tuple] | None:
    """Load vocabulary as set of token tuples from model file."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None

    if "bpe" in str(model_path.name):
        model = BPETokenizer.load(str(model_path))
    else:
        model = UnigramModel.load(str(model_path))
    return set(tuple(token.atomic_tokens) for token in model.tokens.values())


# ========== EXPERIMENT ANALYSIS UTILITIES ==========


def load_experiment_results(
    get_experiment_configs_fn,
    load_model_if_cached_fn,
    experiment_keys: str | list[str],
    corpus_names: list[str],
) -> pd.DataFrame:
    """Load experiment results based on experiment configuration.

    Args:
        get_experiment_configs_fn: Function(experiment_name) -> list[dict]
        load_model_if_cached_fn: Function(corpus_name, params) -> (model, path)
        experiment_keys: Single key or list of keys (experiment names)
        corpus_names: Corpus names to load
    """
    if isinstance(experiment_keys, str):
        experiment_keys = [experiment_keys]

    rows = []
    for exp_key in experiment_keys:
        valid_configs = get_experiment_configs_fn(exp_key)

        for corpus_name in corpus_names:
            for params in valid_configs:
                model, model_path = load_model_if_cached_fn(corpus_name, params)
                if model is None:
                    continue

                data = flatten_model_metadata(model.metadata)
                data["model_file"] = str(model_path)
                rows.append(data)

    return pd.DataFrame(rows)


def identify_baseline(df: pd.DataFrame, corpus_name: str, defaults: dict) -> pd.Series | None:
    """Identify baseline configuration for a corpus (all defaults)."""
    baseline = df[df["corpus"] == corpus_name].copy()

    # Filter for baseline configuration
    for param, default_value in defaults.items():
        if param in baseline.columns:
            baseline = baseline[baseline[param] == default_value]

    if len(baseline) == 0:
        return None
    if len(baseline) > 1:
        print(f"Warning: Multiple baselines found for {corpus_name}, using first")

    return baseline.iloc[0]


def load_baseline_model(
    corpus_name: str,
    defaults: dict,
    load_model_if_cached_fn,
) -> pd.Series | None:
    """Load baseline model (all defaults) for a corpus."""
    model, model_path = load_model_if_cached_fn(corpus_name, defaults)
    if model is None:
        return None

    data = flatten_model_metadata(model.metadata)
    data["model_file"] = str(model_path)
    data["corpus"] = corpus_name
    return pd.Series(data)


def compute_relative_performance(
    df: pd.DataFrame,
    baseline_loader_fn,
) -> pd.DataFrame:
    """Compute relative performance vs baseline for each corpus.

    Args:
        df: DataFrame with experiment results
        baseline_loader_fn: Function(corpus_name) -> pd.Series that loads baseline
    """
    relative_rows = []

    for corpus_name in df["corpus"].unique():
        corpus_data = df[df["corpus"] == corpus_name].copy()
        baseline = baseline_loader_fn(corpus_name)

        if baseline is None:
            print(f"Warning: No baseline found for {corpus_name}, skipping")
            continue

        baseline_objective = baseline["objective"]
        baseline_tokens = baseline["tokens"]
        baseline_cpt = baseline["chars_per_token"]

        for idx, row in corpus_data.iterrows():
            rel_data = {
                "corpus": corpus_name,
                "rel_objective": (row["objective"] - baseline_objective) / baseline_objective * 100,
                "rel_tokens": (row["tokens"] - baseline_tokens) / baseline_tokens * 100,
                "rel_chars_per_token": (row["chars_per_token"] - baseline_cpt) / baseline_cpt * 100,
                "abs_objective": row["objective"],
                "abs_tokens": row["tokens"],
                "abs_chars_per_token": row["chars_per_token"],
                "time": row.get("time", row.get("seconds", 0)),
                "model_file": row.get("model_file", ""),
            }

            # Copy over all other columns from original row
            for col in df.columns:
                if col not in rel_data:
                    rel_data[col] = row[col]

            relative_rows.append(rel_data)

    return pd.DataFrame(relative_rows)


def compute_vocab_overlap(
    df: pd.DataFrame,
    baseline_loader_fn,
) -> pd.DataFrame:
    """Compute vocabulary overlap with baseline for each corpus.

    Args:
        df: DataFrame with experiment results (must have model_file column)
        baseline_loader_fn: Function(corpus_name) -> pd.Series that loads baseline

    Returns:
        DataFrame with vocab_diff_pct: percentage of tokens that differ from baseline.
    """
    rows = []

    for corpus_name in df["corpus"].unique():
        corpus_data = df[df["corpus"] == corpus_name].copy()

        # Load baseline vocab
        baseline_series = baseline_loader_fn(corpus_name)
        if baseline_series is None:
            continue

        baseline_vocab = load_vocab_from_model_file(baseline_series["model_file"])
        if baseline_vocab is None:
            continue

        for idx, row in corpus_data.iterrows():
            row_vocab = load_vocab_from_model_file(row["model_file"])
            if row_vocab is None:
                continue

            # Compute overlap metrics
            total_tokens = len(baseline_vocab.union(row_vocab))
            common_tokens = len(baseline_vocab.intersection(row_vocab))
            diff_tokens = total_tokens - common_tokens

            # Percentage of vocabulary that differs (non-overlapping tokens)
            vocab_diff_pct = (diff_tokens / total_tokens * 100) if total_tokens > 0 else 0

            row_data = {
                "corpus": corpus_name,
                "vocab_diff_pct": vocab_diff_pct,
                "vocab_size": len(row_vocab),
                "baseline_vocab_size": len(baseline_vocab),
                "common_tokens": common_tokens,
                "unique_to_exp": len(row_vocab - baseline_vocab),
                "unique_to_baseline": len(baseline_vocab - row_vocab),
            }

            # Copy over parameter columns
            for col in df.columns:
                if col not in row_data and col not in ["model_file", "objective", "bytes_per_token"]:
                    row_data[col] = row[col]

            rows.append(row_data)

    return pd.DataFrame(rows)
