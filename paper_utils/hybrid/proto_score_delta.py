#!/usr/bin/env python3
"""Test the score_delta reparameterization of MinGram-PP:
  - delta = 1/100_000 (default) should match the historical behaviour
  - delta = 0 should recover PathPiece (min-token + longest tiebreak + MI prune)
All share ONE BPE init so the only differences are the score/criterion. Compares vocab overlap
and held-out compression against a PathPiece pb=0.1 model trained on the same init.
"""

import argparse
import copy
import time

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.pathpiece.trainer import PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import create_logger

PRETOK = "scriptenc_cb"
VOCAB = 32768
ATOMS = 1916


def keyset(m):
    return {tuple(t.atomic_tokens) for t in m.tokens.values()}


def jacc(a, b):
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="eng_latn_300mb")
    ap.add_argument("--eval", default="flores_plus_eng_latn")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    log = create_logger("proto_sd")
    pretok = get_pretokenizer(PRETOK)
    corpus = load_corpus_by_name(a.corpus, pretok)
    ev = load_corpus_by_name(a.eval, pretok)
    iv = round(a.factor * VOCAB) + ATOMS

    # one shared BPE init (UnigramTokens) for all MinGram runs and PathPiece
    mg_init = MinGramTrainer(pretok, corpus, MinGramTrainerConfig(
        additional_vocab_size=VOCAB, overshoot_factor=a.factor, num_em_iterations=2,
        pruning_shrinking_factor=0.9, prune_criterion="mi", num_workers=a.workers, verbose=False)
    )._build_bpe_init_tokens()
    log.info(f"shared init |V|={len(mg_init):,}")

    def train_mg(sd):
        cfg = MinGramTrainerConfig(additional_vocab_size=VOCAB, overshoot_factor=a.factor, num_em_iterations=2,
                                   pruning_shrinking_factor=0.9, prune_criterion="mi", score_delta=sd,
                                   num_workers=a.workers, verbose=False)
        tr = MinGramTrainer(pretok, corpus, cfg)
        tr._build_bpe_init_tokens = lambda init=mg_init: copy.deepcopy(init)
        return tr.train()

    def train_pp():
        cfg = PathPieceTrainerConfig(additional_vocab_size=VOCAB, num_workers=a.workers, init="bpe",
                                     init_vocab_size=iv, max_token_width=1024, prune_batch_fraction=0.1, verbose=False)
        tr = PathPieceTrainer(pretok, corpus, cfg)
        tr._build_bpe_init_vocab = lambda _iv, _L, init=mg_init: copy.deepcopy(init)
        return tr.train()

    runs = {}
    for label, fn in [("mg_default(1e-5)", lambda: train_mg(1e-5)),
                      ("mg_delta0", lambda: train_mg(0.0)),
                      ("pathpiece", train_pp)]:
        t0 = time.time()
        m = fn()
        runs[label] = (m, keyset(m), int(m.corpus_performance(ev)["total_tokens_len"]))
        log.info(f"{label}: |V|={len(m.tokens):,} tok={runs[label][2]:,} ({time.time()-t0:.0f}s)")

    pp_keys = runs["pathpiece"][1]
    pp_tok = runs["pathpiece"][2]
    print(f"\n===== score_delta test (corpus={a.corpus} eval={a.eval} f={a.factor}) =====")
    print(f"{'variant':20}{'|V|':>9}{'eval_tokens':>13}{'vs PathPiece':>14}{'Jaccard vs PP':>15}")
    for label, (m, ks, tok) in runs.items():
        print(f"{label:20}{len(m.tokens):>9,}{tok:>13,}{(tok-pp_tok)/pp_tok*100:>+13.3f}%{jacc(ks, pp_keys):>15.4f}")


if __name__ == "__main__":
    main()
