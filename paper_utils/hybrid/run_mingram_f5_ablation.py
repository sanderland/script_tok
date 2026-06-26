"""Investigate MinGram's compression drop at high f (English, f=5).

MinGram-score peaks at f~=1.15 (-1.88%) and falls to -1.05% at f=5. Two
candidate causes, both tested here by sweeping the relevant knob at fixed f=5:
  * EM under-convergence -> sweep num_em_iterations (2, 4, 8)
  * too-aggressive one-shot pruning -> sweep pruning_shrinking_factor
    (0.0 = prune straight to target; higher = more gradual, prune a smaller
    fraction per outer iteration)
"""

import argparse
import json
import time
from pathlib import Path

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.mingram import MinGramModel, MinGramTrainer, MinGramTrainerConfig
from script_bpe.utils import create_logger

PRETOKENIZER_NAME = "scriptenc_cb"
TRAIN_CORPUS = "fineweb_en_5gb"
EVAL_CORPUS = "eng_latn_fishfood"
UNIGRAM_BASELINE_TOKENS_ENG = 69_729_749
OUT_DIR = Path("results/mingram/ablations")


def run(tag: str, overshoot_factor: float, num_em_iterations: int,
        pruning_shrinking_factor: float, num_workers: int) -> None:
    logger = create_logger(f"mg_f5[{tag}]", verbose=True)
    out_path = OUT_DIR / f"{tag}.model.json.gz"
    summary_path = OUT_DIR / f"{tag}.summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pretokenizer = get_pretokenizer(PRETOKENIZER_NAME)
    train_corpus = load_corpus_by_name(TRAIN_CORPUS, pretokenizer)

    if out_path.exists():
        model = MinGramModel.load(str(out_path))
    else:
        cfg = MinGramTrainerConfig(
            additional_vocab_size=32768,  # matches paper grid -> total 34,684
            num_workers=num_workers,
            overshoot_factor=overshoot_factor,
            num_em_iterations=num_em_iterations,
            pruning_shrinking_factor=pruning_shrinking_factor,
            verbose=True,
        )
        t0 = time.perf_counter()
        model = MinGramTrainer(pretokenizer, train_corpus, cfg).train()
        model.metadata["abl_tag"] = tag
        model.metadata["abl_time"] = time.perf_counter() - t0
        model.save(str(out_path))

    eval_corpus = load_corpus_by_name(EVAL_CORPUS, pretokenizer)
    perf = model.corpus_performance(eval_corpus)
    tokens = int(perf["total_tokens_len"])
    delta = (tokens - UNIGRAM_BASELINE_TOKENS_ENG) / UNIGRAM_BASELINE_TOKENS_ENG * 100
    summary = {
        "tag": tag, "overshoot_factor": overshoot_factor,
        "num_em_iterations": num_em_iterations,
        "pruning_shrinking_factor": pruning_shrinking_factor,
        "final_vocab_size": len(model.tokens),
        "eval_tokens": tokens, "delta_vs_unigram_pct": delta,
        "tokens_per_char": perf["tokens_per_char"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"RESULT[{tag}] |V|={len(model.tokens):,} delta={delta:+.2f}%")
    print(f"\n*** {tag}: em={num_em_iterations} psf={pruning_shrinking_factor} delta={delta:+.2f}% ***\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--overshoot-factor", type=float, default=5.0)
    p.add_argument("--num-em-iterations", type=int, default=2)
    p.add_argument("--pruning-shrinking-factor", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    a = p.parse_args()
    run(a.tag, a.overshoot_factor, a.num_em_iterations, a.pruning_shrinking_factor, a.num_workers)


if __name__ == "__main__":
    main()
