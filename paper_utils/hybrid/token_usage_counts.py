"""Shared helpers for paper token-usage count artifacts."""

from collections.abc import Iterable
from itertools import permutations
from pathlib import Path

import polars as pl

from paper_utils.hybrid.utils import REPO_ROOT

TOKEN_USAGE_COUNTS_PARQUET = REPO_ROOT / "results" / "downstream" / "token_usage_counts.parquet"
RARE_TOKEN_FREQUENCY_THRESHOLD = 1e-7

TOKENIZER_TO_METHOD = {
    "bpe": "bpe",
    "unigram": "unigram_default",
    "fsp": "unigram_fsp",
    "bpe_init_f1.15": "bpe_init_f1.15",
    "fsp_bpe_init_f1.15": "fsp_bpe_init_f1.15",
    "mingram_f1.15": "mingram_f1.15",
    "mingram_pp_f8": "mingram_pp_f8",
    "pathpiece_pb0.1": "pathpiece",
    "convextok": "convextok",
}
METHOD_TO_TOKENIZER = {method: tokenizer for tokenizer, method in TOKENIZER_TO_METHOD.items()}


def load_token_usage_counts(path: Path = TOKEN_USAGE_COUNTS_PARQUET) -> pl.DataFrame:
    return pl.read_parquet(path)


def rare_token_cutoff(df: pl.DataFrame) -> float:
    totals = df.group_by("tokenizer").agg(pl.col("count").sum().alias("total"))
    return float(totals["total"].mean() * RARE_TOKEN_FREQUENCY_THRESHOLD)


def rare_multi_unit_tokens(path: Path = TOKEN_USAGE_COUNTS_PARQUET) -> pl.DataFrame:
    df = load_token_usage_counts(path)
    cutoff = rare_token_cutoff(df)
    return (
        df.filter((pl.col("n_units") > 1) & (pl.col("count") < cutoff))
        .with_columns(pl.col("tokenizer").replace_strict(TOKENIZER_TO_METHOD).alias("method"))
        .sort(["method", "count", "token_id"])
    )


def rare_multi_unit_counts_by_method(path: Path = TOKEN_USAGE_COUNTS_PARQUET) -> dict[str, int]:
    rows = rare_multi_unit_tokens(path).group_by("method").len(name="count").rows(named=True)
    return {row["method"]: row["count"] for row in rows}


def least_common_tokens(path: Path = TOKEN_USAGE_COUNTS_PARQUET, n: int = 10) -> pl.DataFrame:
    df = load_token_usage_counts(path).filter(pl.col("n_units") > 1)
    return (
        df.sort(["tokenizer", "count", "token_id"])
        .with_columns((pl.int_range(pl.len()).over("tokenizer") + 1).alias("rank"))
        .filter(pl.col("rank") <= n)
        .with_columns(
            pl.col("tokenizer").replace_strict(TOKENIZER_TO_METHOD).alias("method"),
            (pl.lit("'") + pl.col("token_str") + pl.lit("' (") + pl.col("count").cast(pl.String) + pl.lit(")")).alias(
                "token_count"
            ),
        )
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right)


def similar_neighbour_order(rows: pl.DataFrame, methods: Iterable[str]) -> list[str]:
    methods = list(methods)
    token_sets = {
        method: frozenset(
            rows.filter(pl.col("method") == method).select("token_str").to_series().to_list()
        )
        for method in methods
    }
    return list(
        max(
            permutations(methods),
            key=lambda order: sum(_jaccard(token_sets[left], token_sets[right]) for left, right in zip(order, order[1:])),
        )
    )
