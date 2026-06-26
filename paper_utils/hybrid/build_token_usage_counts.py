#!/usr/bin/env python3
"""Build the token-usage count artifact used by downstream paper tables."""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from paper_utils.hybrid.token_usage_counts import TOKEN_USAGE_COUNTS_PARQUET
from paper_utils.hybrid.train_hybrid import (
    get_bpe_model_path,
    get_model_path as get_hybrid_model_path,
)
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import FSP_PARAMS, REPO_ROOT
from paper_utils.unigram.train_hyperparameters import (
    ADDITIONAL_VOCAB_SIZE,
    DEFAULTS,
    get_model_path as get_unigram_model_path,
)
from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers import load_tokenizer

DEFAULT_CORPUS = "fineweb_en_5gb"
PRETOKENIZER_NAME = "scriptenc_cb"


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    path: Path


def tokenizer_specs(corpus_name: str) -> list[TokenizerSpec]:
    return [
        TokenizerSpec("bpe", get_bpe_model_path(corpus_name, ADDITIONAL_VOCAB_SIZE)),
        TokenizerSpec("unigram", get_unigram_model_path(corpus_name, DEFAULTS)),
        TokenizerSpec("fsp", get_unigram_model_path(corpus_name, FSP_PARAMS)),
        TokenizerSpec("bpe_init_f1.15", get_hybrid_model_path(corpus_name, {**DEFAULTS, "overshoot_factor": 1.15})),
        TokenizerSpec("fsp_bpe_init_f1.15", get_hybrid_model_path(corpus_name, {**FSP_PARAMS, "overshoot_factor": 1.15})),
        TokenizerSpec("mingram_f1.15", get_mingram_model_path(corpus_name, 1.15, 2, 0.0)),
        TokenizerSpec("mingram_mi_f8", get_mingram_model_path(corpus_name, 8.0, 2, 0.9, prune_criterion="mi")),
        TokenizerSpec("pathpiece_pb0.1", get_pathpiece_model_path(corpus_name, init="bpe")),
        TokenizerSpec(
            "convextok",
            REPO_ROOT / "results" / "convextok_tokenizers" / corpus_name / "n32768_cmin50_mp200000_L32_det.json.gz",
        ),
    ]


def _token_text(model, token) -> str:
    return model.pretokenizer.decode(token.atomic_tokens)


def _count_model_tokens(model, corpus, limit_chunks: int | None) -> Counter[int]:
    counts: Counter[int] = Counter()
    for i, (atomic_tokens, freq) in enumerate(corpus):
        if limit_chunks is not None and i >= limit_chunks:
            break
        text = corpus.pretokenizer.decode(atomic_tokens)
        encoded = model.encode(text)
        chunk_counts = Counter(int(token_id) for token_id in encoded)
        counts.update({token_id: count * freq for token_id, count in chunk_counts.items()})
    return counts


def build_counts(corpus_name: str, methods: set[str] | None, limit_chunks: int | None) -> pl.DataFrame:
    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(corpus_name, pretokenizer)
    rows = []
    for spec in tokenizer_specs(corpus_name):
        if methods is not None and spec.name not in methods:
            continue
        model = load_tokenizer(str(spec.path))
        counts = _count_model_tokens(model, corpus, limit_chunks)
        for token in model.tokens.values():
            token_str = _token_text(model, token)
            rows.append(
                {
                    "tokenizer": spec.name,
                    "token_id": int(token.id),
                    "token_str": token_str,
                    "n_units": len(token.atomic_tokens),
                    "is_multichar": len(token_str) > 1,
                    "count": int(counts[token.id]),
                }
            )
    return pl.DataFrame(rows).sort(["tokenizer", "token_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=TOKEN_USAGE_COUNTS_PARQUET)
    parser.add_argument("--methods", nargs="+", default=None, help="Subset of tokenizer names to rebuild.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Debug/smoke-test limit over unique corpus chunks.")
    args = parser.parse_args()

    methods = set(args.methods) if args.methods is not None else None
    df = build_counts(args.corpus, methods, args.limit_chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output)
    print(f"wrote {args.output} ({df.height:,} rows)")


if __name__ == "__main__":
    main()
