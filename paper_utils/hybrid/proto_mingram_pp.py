#!/usr/bin/env python3
"""PROTOTYPE: stock MinGram (usage-count prune) vs MinGram-PP (MI prune).

Trains both variants with an identical config on one corpus, evaluates compression on
a held-out corpus, and reports |V|, tokens/pretoken, held-out token count, and how much
the two vocabularies actually differ (to see whether MinGram-PP diverges from the
usage-count prune -- and how close it lands to a PathPiece-style vocab).

Separate, throwaway harness (delete to revert). Example:
  uv run python -m paper_utils.hybrid.proto_mingram_pp \
      --corpus eng_latn_300mb --eval flores_plus_eng_latn --vocab 32768 --factor 1.15
"""

import argparse
import copy
import time

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.utils import create_logger

PRETOKENIZER_NAME = "scriptenc_cb"


def _vocab_keys(model) -> set:
    return {tuple(t.atomic_tokens) for t in model.tokens.values()}


def _train(label, pretok, corpus, cfg, init_tokens, logger):
    t0 = time.time()
    trainer = MinGramTrainer(pretok, corpus, cfg)
    # Share ONE BPE init across variants (deep copy: train() mutates log_prob), so the
    # only difference is the prune criterion -- not init noise.
    trainer._build_bpe_init_tokens = lambda: copy.deepcopy(init_tokens)
    model = trainer.train()
    removed = model.metadata.get("totals_removed", {})
    logger.info(f"[{label}] |V|={len(model.tokens):,} in {time.time()-t0:.0f}s  removed={dict(removed)}")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eng_latn_300mb")
    ap.add_argument("--eval", default="flores_plus_eng_latn")
    ap.add_argument("--vocab", type=int, default=32768, help="additional_vocab_size")
    ap.add_argument("--factor", type=float, default=1.15, help="overshoot_factor")
    ap.add_argument("--em", type=int, default=2)
    ap.add_argument("--p", type=float, default=0.0, help="pruning_shrinking_factor")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    logger = create_logger("proto_mingram")
    pretok = get_pretokenizer(PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(args.corpus, pretok)
    eval_corpus = load_corpus_by_name(args.eval, pretok)

    def mk_cfg():
        return MinGramTrainerConfig(
            additional_vocab_size=args.vocab,
            overshoot_factor=args.factor,
            num_em_iterations=args.em,
            pruning_shrinking_factor=args.p,
            num_workers=args.workers,
            verbose=False,
        )

    # Build the shared BPE-init ONCE, reuse (deep-copied) for both variants.
    logger.info("building shared BPE init...")
    init_tokens = MinGramTrainer(pretok, corpus, mk_cfg())._build_bpe_init_tokens()
    logger.info(f"shared init |V|={len(init_tokens):,}")

    pp_cfg = mk_cfg()
    pp_cfg.prune_criterion = "mi"
    stock = _train("stock", pretok, corpus, mk_cfg(), init_tokens, logger)
    mingram_pp = _train("mingram_pp", pretok, corpus, pp_cfg, init_tokens, logger)

    ps = stock.corpus_performance(eval_corpus)
    pc = mingram_pp.corpus_performance(eval_corpus)
    ks, kc = _vocab_keys(stock), _vocab_keys(mingram_pp)
    inter = len(ks & kc)
    jacc = inter / len(ks | kc) if (ks | kc) else 0.0

    print("\n========== MinGram: usage-count vs MinGram-PP prune ==========")
    print(f"corpus(train)={args.corpus}  eval={args.eval}  vocab+{args.vocab}  f={args.factor} em={args.em} p={args.p}")
    print(f"{'variant':10}{'|V|':>9}{'eval_tokens':>14}{'tok/pretoken':>14}")
    for label, m, p in [("stock", stock, ps), ("mingram_pp", mingram_pp, pc)]:
        tpp = m.metadata.get("tokens/pretoken", float("nan"))
        print(f"{label:10}{len(m.tokens):>9,}{p['total_tokens_len']:>14,}{tpp:>14.5f}")
    base = ps["total_tokens_len"]
    delta = (pc["total_tokens_len"] - base) / base * 100
    print(f"\nMinGram-PP vs stock held-out token count: {delta:+.3f}%  (negative = MinGram-PP compresses better)")
    print(f"vocab overlap: {inter:,} shared  |  Jaccard={jacc:.3f}  "
          f"(stock-only {len(ks-kc):,}, MinGram-PP-only {len(kc-ks):,})")
    print("eval_perf keys:", list(ps.keys()))


if __name__ == "__main__":
    main()
