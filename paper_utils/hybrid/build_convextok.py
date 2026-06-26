"""Train the ConvexTok baseline used by the Mingram paper tables and figures.

This is the held-out-optimal ConvexTok setup used in the paper comparison:
SCRIPT pretokenization, 32k selected tokens, cmin=50, mp=200k, L=32, and
deterministic rounding. The output path matches the paper generators.

Usage: build_convextok.py <corpus_name> [max_pretokens] [num_workers]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.convextok import ConvexTokTrainer, ConvexTokTrainerConfig

from paper_utils.unigram.train_hyperparameters import ADDITIONAL_VOCAB_SIZE

PRETOKENIZER_NAME = "scriptenc_cb"
C_MIN = 50
MAX_TOKEN_WIDTH = 32
ROUNDING = "det"
LP_METHOD = "pdlp"
PDLP_TOL = 1e-7


def get_model_path(corpus_name: str, max_pretokens: int) -> Path:
    return (
        Path("results/convextok_tokenizers")
        / corpus_name
        / f"n{ADDITIONAL_VOCAB_SIZE}_cmin{C_MIN}_mp{max_pretokens}_L{MAX_TOKEN_WIDTH}_{ROUNDING}.json.gz"
    )


def main() -> None:
    corpus_name = sys.argv[1] if len(sys.argv) > 1 else "eng_latn_300mb"
    max_pretokens = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    out_path = get_model_path(corpus_name, max_pretokens)

    if out_path.exists():
        print(f"[skip] already solved: {out_path}", flush=True)
        return

    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    config = ConvexTokTrainerConfig(
        additional_vocab_size=ADDITIONAL_VOCAB_SIZE,
        cmin=C_MIN,
        max_pretokens=max_pretokens,
        max_token_width=MAX_TOKEN_WIDTH,
        rounding=ROUNDING,
        lp_method=LP_METHOD,
        num_workers=num_workers,
        pdlp_tol=PDLP_TOL,
    )

    start = time.time()
    print(f"[load] {corpus_name} ({PRETOKENIZER_NAME}) ...", flush=True)
    corpus = load_corpus_by_name(corpus_name, pretokenizer)
    model = ConvexTokTrainer(pretokenizer, corpus, config).train()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    metadata = model.metadata
    print(
        f"[done] {time.time() - start:.0f}s  "
        f"bound={metadata['lp_objective_lower_bound']:,.1f}  "
        f"rounded_CTC={metadata['rounded_corpus_token_count']:,}  "
        f"ratio={metadata['optimality_ratio']:.5f}  "
        f"n_pretokens={metadata['n_pretokens']:,} -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
