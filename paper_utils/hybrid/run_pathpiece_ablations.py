"""Run a PathPiece configuration ablation on English-only and report CTC.

Usage:
    uv run python -m paper_utils.hybrid.run_pathpiece_ablations \
        --tag 512k_L16_ngram --init ngram --init-vocab 524288 --L 16

Always trains fineweb_en_5gb, evaluates on eng_latn_fishfood, reports
``tokens`` and ``delta vs Unigram baseline``.
"""

import argparse
import json
import time
from pathlib import Path

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.pathpiece import PathPieceModel, PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import create_logger

PRETOKENIZER_NAME = "scriptenc_cb"
TRAIN_CORPUS = "fineweb_en_5gb"
EVAL_CORPUS = "eng_latn_fishfood"
UNIGRAM_BASELINE_TOKENS_ENG = 69_729_749  # from results/hybrid/cache_train_eval_compression_grid.json
OUT_DIR = Path("results/pathpiece/ablations")


def run(
    tag: str,
    init: str,
    init_vocab: int,
    L: int,
    num_workers: int,
    prune_batch_fraction: float = 0.20,
    ngram_init_rank: str = "count",
) -> None:
    logger = create_logger(f"pp_abl[{tag}]", verbose=True)
    out_path = OUT_DIR / f"{tag}.model.json.gz"
    summary_path = OUT_DIR / f"{tag}.summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    train_corpus = load_corpus_by_name(TRAIN_CORPUS, pretokenizer)

    if out_path.exists():
        logger.info(f"Loading cached {out_path}")
        model = PathPieceModel.load(str(out_path))
    else:
        cfg = PathPieceTrainerConfig(
            # additional_vocab_size matches the paper grid (ADDITIONAL_VOCAB_SIZE=32768);
            # total vocab = atomic + 32768 = 34,684, same as every main-table model.
            additional_vocab_size=32768,
            num_workers=num_workers,
            init=init,
            init_vocab_size=init_vocab,
            max_token_width=L,
            prune_batch_fraction=prune_batch_fraction,
            ngram_init_rank=ngram_init_rank,
            verbose=True,
        )
        t0 = time.perf_counter()
        model = PathPieceTrainer(pretokenizer, train_corpus, cfg).train()
        elapsed = time.perf_counter() - t0
        model.metadata["abl_tag"] = tag
        model.metadata["abl_time"] = elapsed
        model.save(str(out_path))
        logger.info(f"Trained in {elapsed/60:.1f}min; saved to {out_path}")

    eval_corpus = load_corpus_by_name(EVAL_CORPUS, pretokenizer)
    t0 = time.perf_counter()
    perf = model.corpus_performance(eval_corpus)
    enc_t = time.perf_counter() - t0
    tokens = int(perf["total_tokens_len"])
    delta = (tokens - UNIGRAM_BASELINE_TOKENS_ENG) / UNIGRAM_BASELINE_TOKENS_ENG * 100

    summary = {
        "tag": tag,
        "init": init,
        "init_vocab_size": init_vocab,
        "max_token_width": L,
        "final_vocab_size": len(model.tokens),
        "eval_tokens": tokens,
        "unigram_baseline_tokens": UNIGRAM_BASELINE_TOKENS_ENG,
        "delta_vs_unigram_pct": delta,
        "tokens_per_char": perf["tokens_per_char"],
        "encode_time_s": enc_t,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"RESULT[{tag}] |V|={len(model.tokens):,}  tokens={tokens:,}  delta={delta:+.2f}%  tok/char={perf['tokens_per_char']:.5f}")
    print(f"\n*** {tag}: delta={delta:+.2f}% (|V|={len(model.tokens):,}, tok={tokens:,}, tok/char={perf['tokens_per_char']:.5f}) ***\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--init", choices=["ngram", "bpe"], required=True)
    parser.add_argument("--init-vocab", type=int, required=True)
    parser.add_argument("-L", "--max-token-width", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prune-batch-fraction", type=float, default=0.20)
    parser.add_argument("--ngram-init-rank", choices=["count", "count_width"], default="count")
    args = parser.parse_args()
    run(
        args.tag,
        args.init,
        args.init_vocab,
        args.max_token_width,
        args.num_workers,
        args.prune_batch_fraction,
        ngram_init_rank=args.ngram_init_rank,
    )


if __name__ == "__main__":
    main()
