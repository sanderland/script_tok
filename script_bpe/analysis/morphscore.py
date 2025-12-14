"""MorphScore evaluation for morphological alignment of tokenizers.

Based on the MorphScore dataset: https://huggingface.co/datasets/catherinearnett/morphscore
"""

import json

import numpy as np
import pandas as pd
from datasets import load_dataset


# Cache the HF dataset at module level to avoid reloading
_HF_DATASET_CACHE: pd.DataFrame | None = None


def _get_hf_dataset() -> pd.DataFrame:
    """Load the MorphScore dataset from HuggingFace (cached)."""
    global _HF_DATASET_CACHE
    if _HF_DATASET_CACHE is None:
        print("Loading MorphScore dataset from HuggingFace...")
        ds = load_dataset("catherinearnett/morphscore", split="train")
        _HF_DATASET_CACHE = ds.to_pandas()
        print(f"Loaded {len(_HF_DATASET_CACHE)} rows")
    return _HF_DATASET_CACHE


class MorphScore:
    """
    MorphScore evaluator for measuring morphological alignment of tokenizers.

    Evaluates how well a tokenizer's segmentation aligns with morphological boundaries.
    """

    def __init__(self, config_path: str | None = None, **kwargs):
        """
        Initialize MorphScore evaluator.

        Args:
            config_path: Path to JSON config file (optional)
            **kwargs: Direct configuration arguments that override defaults/config
        """
        # Set defaults
        self.config = {
            # Filtering flags
            "unique_only": True,
            "stem_eq_lemma": True,
            "exclude_numbers": True,
            "language_subset": [],  # if not empty, run on this subset of languages only
            "splits": ["train", "dev", "test"],
            # Scoring flags
            "freq_scale": False,  # scale scoring by word frequency
            "exclude_single_tok": True,  # exclude single token words from scoring
            "exclude_single_morpheme": True,  # exclude single morpheme words from scoring
            "single_tok_point": 1,  # if exclude_single_tok is False, this is the score for single token words
            "correct_point": 1,  # all morpheme boundaries must be correct
            "partial_point": 0.5,  # only one morpheme boundary is correct
            # Breakdown flags
            "by_split": False,
            "by_pos": False,
            # tokenizer settings
            "subword_prefix": "",  # prefix for subwords, e.g. '##' for wordpiece
        }

        # Load config file if provided
        if config_path:
            self._load_config(config_path)

        # Override with any direct arguments
        self.config.update(kwargs)

        # Validate configuration
        self._validate_config()

    def _load_config(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, "r") as f:
            file_config = json.load(f)
        self.config.update(file_config)

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate point values
        if not isinstance(self.config["single_tok_point"], (int, float)):
            raise ValueError("single_tok_point must be numeric")
        if not isinstance(self.config["correct_point"], (int, float)):
            raise ValueError("correct_point must be numeric")
        if not isinstance(self.config["partial_point"], (int, float)):
            raise ValueError("partial_point must be numeric")

        # Validate lists
        if not isinstance(self.config["language_subset"], list):
            raise ValueError("language_subset must be a list")
        if not isinstance(self.config["splits"], list):
            raise ValueError("splits must be a list")

    def morph_eval(self, morphemes: list[str], tokens: list[str]) -> tuple[float, float]:
        """
        Evaluate morphological segmentation for a single word.

        Args:
            morphemes: Ground truth morpheme segmentation [preceding_part, stem, following_part],
                where preceding_part and following_part are optional
            tokens: Tokenizer output segments

        Returns:
            tuple: (morphscore_recall_point, morphscore_precision_point)
        """
        if len(tokens) == 1:
            return (
                (np.nan, np.nan)
                if self.config["exclude_single_tok"]
                else (self.config["single_tok_point"], self.config["single_tok_point"])
            )

        # find indices of predicted morpheme boundaries
        all_pred_boundaries = []
        idx = 0
        for t in range(len(tokens)):
            tok = tokens[t]
            this_idx = idx + len(tok)
            all_pred_boundaries.append(this_idx)
            idx = this_idx

        # find index of the gold morpheme boundary and score
        if len(morphemes) == 2:  # only 1 gold morpheme boundary
            gold_boundary_idx = len(morphemes[0])
            if gold_boundary_idx in all_pred_boundaries:
                return self.config["correct_point"], 1 / len(all_pred_boundaries)
            else:
                return 0, 0

        elif len(morphemes) == 3:
            gold_boundary_indices = [len(morphemes[0]), len(morphemes[0]) + len(morphemes[1])]

            # if both boundaries are in the predicted boundaries, score is correct
            if gold_boundary_indices[0] in all_pred_boundaries and gold_boundary_indices[1] in all_pred_boundaries:
                return self.config["correct_point"], 2 / len(all_pred_boundaries)
            # if one boundary is in the predicted boundaries, score is partial
            elif gold_boundary_indices[0] in all_pred_boundaries or gold_boundary_indices[1] in all_pred_boundaries:
                return self.config["partial_point"], 1 / len(all_pred_boundaries)
            else:
                return 0, 0

        # number of gold morphemes is 1
        else:
            if self.config["exclude_single_morpheme"]:
                return (np.nan, np.nan)
            else:
                return (
                    (self.config["single_tok_point"], self.config["single_tok_point"])
                    if morphemes == tokens
                    else (0, 0)
                )

    def _load_dataset(self, language: str) -> pd.DataFrame:
        """Load dataset for a specific language from HuggingFace."""
        full_df = _get_hf_dataset()

        # Filter by language (HF dataset uses format like "eng_latn", "deu_latn", etc.)
        language = language.lower()
        filtered = full_df[full_df["language"] == language]

        if len(filtered) == 0:
            # Try with common variations
            available = full_df["language"].unique()
            raise FileNotFoundError(f"No data for language '{language}'. Available: {sorted(available)[:20]}...")

        return filtered

    def _filter_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering based on configuration flags."""
        filtered_df = dataset.copy()

        # Filter by splits, if the column starts with the split name
        if "data_split" in filtered_df.columns:
            split_dfs = []
            for split in self.config["splits"]:
                split_dfs.append(filtered_df[filtered_df["data_split"].str.startswith(split)])
            filtered_df = pd.concat(split_dfs)

        # Filter unique only
        if self.config["unique_only"] and "unique" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["unique"] == "unique"]

        # Filter stem equals lemma
        if self.config["stem_eq_lemma"] and all(col in filtered_df.columns for col in ["stem", "lemma"]):
            filtered_df = filtered_df[filtered_df["stem"] == filtered_df["lemma"]]

        # Filter out numbers
        if self.config["exclude_numbers"] and "wordform" in filtered_df.columns:
            filtered_df = filtered_df[~filtered_df["wordform"].astype(str).str.contains(r"\d")]

        return filtered_df

    def get_morphscore(self, dataset: pd.DataFrame, tokenizer, return_df: bool = False) -> tuple | dict:
        """
        Calculate MorphScore for a dataset.

        Expects tokenizer to be a UnigramModel with encode() method.

        Args:
            dataset: DataFrame with morphological data
            tokenizer: Tokenizer with encode() method and pretokenizer
            return_df: If True, return (results_dict, scored_dataframe)

        Returns:
            dict with morphscore metrics, or tuple (dict, DataFrame) if return_df=True
        """
        required_cols = ["stem", "lemma", "preceding_part", "following_part", "wordform"]
        if not all(col in dataset.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}")

        # Score storage
        points_morphscore_recall = []
        points_morphscore_precision = []
        weights = []
        token_char_ratios = []
        matched_subwords = []
        gold_subwords = []
        pred_subwords = []
        all_gold_morphemes = []
        all_pred_tokens = []

        def add_nan_values():
            points_morphscore_recall.append(np.nan)
            points_morphscore_precision.append(np.nan)
            matched_subwords.append(np.nan)
            gold_subwords.append(np.nan)
            pred_subwords.append(np.nan)
            all_gold_morphemes.append(np.nan)
            all_pred_tokens.append(np.nan)
            weights.append(np.nan)
            token_char_ratios.append(np.nan)

        for idx in range(len(dataset)):
            row = dataset.iloc[idx]
            prefix = row["preceding_part"]
            suffix = row["following_part"]
            stem = row["stem"]
            wordform = row["wordform"]
            norm_freq = float(row["word_freq_norm"])

            if not isinstance(wordform, str):
                if pd.isna(wordform):
                    add_nan_values()
                    continue
                wordform = str(wordform).strip()
                if not wordform:
                    add_nan_values()
                    continue

            if not wordform or wordform.isspace():
                add_nan_values()
                continue

            # Assemble gold morphemes
            morphemes = []
            if not pd.isna(prefix):
                morphemes.append(prefix)
            morphemes.append(stem)
            if not isinstance(suffix, float):  # i.e., not NaN
                morphemes.append(suffix)

            # Tokenize using UnigramModel interface
            token_ids = tokenizer.encode(wordform)
            tokens = [tokenizer.pretokenizer.decode(tokenizer.tokens[tid].atomic_tokens) for tid in token_ids]

            # Apply subword prefix removal if configured
            if self.config["subword_prefix"]:
                tokens = [t.replace(self.config["subword_prefix"], "") for t in tokens]

            # Calculate token_char_ratio
            if len(wordform) > 0:
                token_char_ratios.append(len(tokens) / len(wordform))
            else:
                token_char_ratios.append(np.nan)

            # MorphScore
            point_recall, point_precision = self.morph_eval(morphemes, tokens)

            if self.config["freq_scale"] and not np.isnan(point_recall):
                weights.append(norm_freq)
            elif not np.isnan(point_recall):
                weights.append(1)
            else:
                weights.append(np.nan)  # Will be dropped later

            points_morphscore_recall.append(point_recall)
            points_morphscore_precision.append(point_precision)

            # Traditional metrics
            n_matched = len(set(tokens) & set(morphemes))
            n_gold = len(morphemes)
            n_pred = len(tokens)
            matched_subwords.append(n_matched)
            gold_subwords.append(n_gold)
            pred_subwords.append(n_pred)
            all_gold_morphemes.append(morphemes)
            all_pred_tokens.append(tokens)

        dataset = dataset.copy()
        dataset["morphscore_recall"] = points_morphscore_recall
        dataset["morphscore_precision"] = points_morphscore_precision
        dataset["token_char_ratio"] = token_char_ratios
        dataset["matched_subwords"] = matched_subwords
        dataset["gold_subwords"] = gold_subwords
        dataset["pred_subwords"] = pred_subwords
        dataset["gold_morphemes"] = all_gold_morphemes
        dataset["pred_morphemes"] = all_pred_tokens

        # Drop NaNs for final scoring
        new_dataset = dataset.dropna(
            subset=[
                "morphscore_recall",
                "morphscore_precision",
                "token_char_ratio",
                "matched_subwords",
                "gold_subwords",
                "pred_subwords",
            ]
        )
        valid_weights = [w for w in weights if not np.isnan(w)]
        assert len(new_dataset) == len(valid_weights)

        # Weighted MorphScores
        weighted_recall_points = [p * w for p, w in zip(new_dataset["morphscore_recall"], valid_weights)]
        weighted_precision_points = [p * w for p, w in zip(new_dataset["morphscore_precision"], valid_weights)]

        mean_morphscore_recall = float(np.sum(weighted_recall_points) / np.sum(valid_weights))
        mean_morphscore_precision = float(np.sum(weighted_precision_points) / np.sum(valid_weights))

        # Micro scores
        n_matched = np.sum(new_dataset["matched_subwords"])
        n_gold = np.sum(new_dataset["gold_subwords"])
        n_pred = np.sum(new_dataset["pred_subwords"])
        micro_precision = float(n_matched / n_pred)
        micro_recall = float(n_matched / n_gold)
        if micro_precision + micro_recall == 0:
            micro_f1 = 0.0
        else:
            micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall))

        # Macro scores
        all_precs = [row["matched_subwords"] / row["pred_subwords"] for _, row in new_dataset.iterrows()]
        all_recalls = [row["matched_subwords"] / row["gold_subwords"] for _, row in new_dataset.iterrows()]
        macro_precision = float(np.mean(all_precs))
        macro_recall = float(np.mean(all_recalls))
        if (macro_precision + macro_recall) == 0:
            macro_f1 = 0.0
        else:
            macro_f1 = float(2 * macro_precision * macro_recall / (macro_precision + macro_recall))

        mean_token_char_ratio = (
            np.mean(new_dataset["token_char_ratio"]) if len(new_dataset["token_char_ratio"]) > 0 else 0.0
        )

        results = {
            "morphscore_recall": mean_morphscore_recall,
            "morphscore_precision": mean_morphscore_precision,
            "morphscore_recall_std": np.std(weighted_recall_points) if len(weighted_recall_points) > 1 else 0.0,
            "morphscore_precision_std": np.std(weighted_precision_points)
            if len(weighted_precision_points) > 1
            else 0.0,
            "mean_token_char_ratio": mean_token_char_ratio,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "num_samples": len(new_dataset),
        }

        return (results, new_dataset) if return_df else results

    def eval(self, tokenizer, return_df: bool = False) -> dict:
        """
        Evaluate tokenizer on all configured languages.

        Args:
            tokenizer: Tokenizer with encode() method
            return_df: If True, return results with DataFrame from last language

        Returns:
            dict mapping language codes to result dicts, plus 'config' key
        """
        results_per_lang = {"config": self.config.copy()}
        new_dataset = None

        if self.config["language_subset"]:
            languages = self.config["language_subset"]
        else:
            # Get all available languages from HF dataset
            full_df = _get_hf_dataset()
            languages = sorted(full_df["language"].unique())

        for language in languages:
            dataset = self._load_dataset(language)
            filtered_data = self._filter_dataset(dataset)

            if len(filtered_data) == 0:
                results_per_lang[language] = {"num_samples": 0, "error": "No samples after filtering"}
                continue

            if return_df:
                results, new_dataset = self.get_morphscore(filtered_data, tokenizer, return_df)
            else:
                results = self.get_morphscore(filtered_data, tokenizer, return_df)

            # Breakdown by split
            if self.config["by_split"] and "data_split" in filtered_data.columns:
                results["by_split"] = {}
                for split in filtered_data["data_split"].unique():
                    split_data = filtered_data[filtered_data["data_split"] == split]
                    split_results = self.get_morphscore(split_data, tokenizer, return_df=False)
                    results["by_split"][split] = split_results

            # Breakdown by POS
            if self.config["by_pos"] and "pos" in filtered_data.columns:
                results["by_pos"] = {}
                for pos in filtered_data["pos"].unique():
                    pos_data = filtered_data[filtered_data["pos"] == pos]
                    pos_results = self.get_morphscore(pos_data, tokenizer, return_df=False)
                    results["by_pos"][pos] = pos_results

            results_per_lang[language] = results

        if return_df:
            return results_per_lang, new_dataset
        else:
            return results_per_lang

    def _get_filtered_dataset(self) -> pd.DataFrame:
        """Get the filtered dataset based on current config."""
        if self.config["language_subset"]:
            languages = self.config["language_subset"]
        else:
            full_df = _get_hf_dataset()
            languages = sorted(full_df["language"].unique())

        dfs = []
        for language in languages:
            dataset = self._load_dataset(language)
            filtered = self._filter_dataset(dataset)
            if len(filtered) > 0:
                dfs.append(filtered)

        return pd.concat(dfs) if dfs else pd.DataFrame()

    def get_word_frequencies(self) -> dict[str, float]:
        """Return word -> normalized frequency dict for the filtered dataset."""
        dataset = self._get_filtered_dataset()
        return dict(zip(dataset["wordform"], dataset["word_freq_norm"]))

    def analyze_tokenizer(self, tokenizer) -> list[dict]:
        """
        Analyze a single tokenizer, returning per-word results.

        Args:
            tokenizer: Tokenizer with encode() method

        Returns:
            list of dicts, each containing:
            - word: str
            - gold: list[str] (gold morphemes)
            - predicted: list[str] (tokenizer output)
            - recall: float (morphscore recall for this word, or None if excluded)
            - is_correct: bool (recall == 1.0)
            - is_single: bool (single token output)
        """
        dataset = self._get_filtered_dataset()
        results = []

        for idx in range(len(dataset)):
            row = dataset.iloc[idx]
            wordform = row["wordform"]

            if not isinstance(wordform, str) or not wordform or wordform.isspace():
                continue

            # Assemble gold morphemes
            prefix = row["preceding_part"]
            suffix = row["following_part"]
            stem = row["stem"]
            morphemes = []
            if not pd.isna(prefix):
                morphemes.append(prefix)
            morphemes.append(stem)
            if not isinstance(suffix, float):  # i.e., not NaN
                morphemes.append(suffix)

            # Tokenize
            token_ids = tokenizer.encode(wordform)
            tokens = [tokenizer.pretokenizer.decode(tokenizer.tokens[tid].atomic_tokens) for tid in token_ids]

            if self.config["subword_prefix"]:
                tokens = [t.replace(self.config["subword_prefix"], "") for t in tokens]

            # Compute recall
            recall_val, _ = self.morph_eval(morphemes, tokens)
            recall = None if np.isnan(recall_val) else float(recall_val)

            results.append(
                {
                    "word": wordform,
                    "gold": morphemes,
                    "predicted": tokens,
                    "recall": recall,
                    "is_correct": recall == 1.0 if recall is not None else False,
                    "is_single": len(tokens) == 1,
                }
            )

        return results

    def update_config(self, **kwargs):
        """Update configuration parameters."""
        self.config.update(kwargs)
        self._validate_config()

    def get_config(self) -> dict:
        """Get current configuration."""
        return self.config.copy()
