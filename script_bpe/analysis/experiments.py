"""Experiment utilities for model training and analysis."""

import hashlib
import json


def get_config_hash(config_dict: dict) -> str:
    """
    Generate a stable 16-character hash for a configuration dictionary.

    Args:
        config_dict: Configuration dictionary to hash

    Returns:
        16-character hex string hash
    """
    # Sort keys for stable hash
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def flatten_model_metadata(metadata: dict) -> dict:
    """
    Flatten nested config and performance metadata for easier DataFrame access.

    Extracts values from nested 'config' and 'performance' dicts into top-level keys.

    Args:
        metadata: Model metadata dict with nested 'config' and 'performance' dicts

    Returns:
        Flattened dict with all values at top level
    """
    data = dict(metadata)

    # Flatten config
    if "config" in data:
        data.update(data["config"])

    # Flatten performance stats
    if "performance" in data:
        perf = data["performance"]
        data.update(
            {
                "tokens": perf.get("total_tokens_len"),
                "chars": perf.get("total_char_len"),
                "bytes": perf.get("total_byte_len"),
                "bytes_per_token": perf.get("bytes_per_token"),
                "chars_per_token": perf.get("chars_per_token"),
                "tokens_per_char": perf.get("tokens_per_char"),
                "bytes_per_char": perf.get("bytes_per_char"),
            }
        )

    # Include training stats if available
    if "stats" in data:
        stats = data["stats"]
        if "num_iterations" in stats:
            data["num_iterations"] = stats["num_iterations"]

    return data

