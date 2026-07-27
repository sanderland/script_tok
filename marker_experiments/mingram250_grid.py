"""MinGram vs BPE for every boundary variant, at a scale that completes.

The 1 GB grid's MinGram half needs ~26 min per cell, and the container clears the
working tree every ~30-60 min, so cells kept dying mid-flight: one of 24 landed. This
runs the same comparison at 250M characters, where a MinGram cell is ~8 min and a BPE
cell ~2 min, so a whole language finishes inside one window.

  languages     : en, ru, ko  (Latin, Cyrillic, Hangul -- one per script family)
  pretokenizers : plain, bnd_w, bnd_wp, bnd_wpd
  trainers      : bpe, mingram   (both, so the table is internally comparable)
  vocabulary    : 32,768 additional
  => 24 cells

BPE and MinGram share a language's corpora, and both trainers run before the corpora are
freed, so each corpus is built once. Evaluation documents are withheld from training via
train_batches. These numbers are NOT comparable to the 1 GB table in §4.1, which is a
different scale and trains on its own eval slice.
"""

import gc
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.mingram.trainer import MinGramTrainerConfig

import finewiki1gb_grid as G
from boundary_pretokenizer import get_boundary_pretokenizer
from finewiki1gb_grid import (
    CORPORA,
    NUM_WORKERS,
    OVERSHOOT,
    VOCAB,
    CachedInitMinGramTrainer,
    analyse_vocab,
    commit_cell,
    ensure_eval,
    log,
    train_batches,
)

LANGS = ["en", "ru", "ko"]
CHARS = 250_000_000
RESULT_PATH = os.path.join(HERE, "mingram250_result.json")
TOKENIZERS = os.path.join(HERE, "tokenizers")

PRETOKENIZERS = {"plain": lambda: get_pretokenizer("scriptenc3_cb")}
PRETOKENIZERS.update({n: (lambda n=n: get_boundary_pretokenizer(n)) for n in ["bnd_w", "bnd_wp", "bnd_wpd"]})


def main():
    os.makedirs(TOKENIZERS, exist_ok=True)
    G.CHARS_PER_LANG = CHARS
    results = json.load(open(RESULT_PATH)) if os.path.exists(RESULT_PATH) else {}

    for lang in LANGS:
        if all(f"{lang}_{tag}_{m}" in results for tag in PRETOKENIZERS for m in ("bpe", "mingram")):
            log(f"{lang}: complete, skipping")
            continue
        eval_texts = ensure_eval(lang)
        eval_chars = sum(map(len, eval_texts))

        for tag, make_pt in PRETOKENIZERS.items():
            pt = make_pt()
            corpus_name = f"mg250_{lang}_{tag}"
            corpus = None
            for method in ["bpe", "mingram"]:
                key = f"{lang}_{tag}_{method}"
                if key in results:
                    continue
                try:
                    if corpus is None:
                        try:
                            corpus = PretokenizedCorpus(name=corpus_name, base_path=CORPORA, pretokenizer=pt)
                        except FileNotFoundError:
                            t = time.time()
                            corpus = PretokenizedCorpus.from_text_batches(
                                name=corpus_name, base_path=CORPORA, pretokenizer=pt,
                                text_batches=train_batches(lang), num_workers=NUM_WORKERS,
                            )
                            log(f"{lang}/{tag}: corpus built in {time.time()-t:.0f}s "
                                f"unique_chunks={corpus.metadata.get('unique_chunks'):,}")

                    t = time.time()
                    if method == "bpe":
                        tokenizer = BPETrainer(
                            pt, corpus, BPETrainerConfig(additional_vocab_size=VOCAB, num_workers=NUM_WORKERS)
                        ).train()
                    else:
                        tr = CachedInitMinGramTrainer(
                            pt, corpus,
                            MinGramTrainerConfig(additional_vocab_size=VOCAB, num_workers=NUM_WORKERS,
                                                 overshoot_factor=OVERSHOOT),
                        )
                        tr.cache_tag = f"mg250_{lang}_{tag}"
                        tokenizer = tr.train()
                    train_time = time.time() - t

                    out = os.path.join(TOKENIZERS, f"mg250_{key}_32k.json.gz")
                    tokenizer.save(out)

                    toks = fails = 0
                    for text in eval_texts:
                        ids = tokenizer.encode(text)
                        toks += len(ids)
                        if tokenizer.decode(ids) != text:
                            fails += 1

                    results[key] = {
                        "lang": lang, "pretokenizer": tag, "method": method,
                        "additional_vocab_size": VOCAB, "vocab_size": len(tokenizer.tokens),
                        "train_seconds": round(train_time),
                        "unique_chunks": corpus.metadata.get("unique_chunks"),
                        "eval_docs": len(eval_texts), "eval_chars": eval_chars, "eval_tokens": toks,
                        "eval_chars_per_token": eval_chars / toks,
                        "roundtrip_failures": fails,
                        "tokenizer_file": os.path.relpath(out, os.path.dirname(HERE)),
                        **analyse_vocab(tokenizer, pt),
                    }
                    with open(RESULT_PATH, "w") as f:
                        json.dump(results, f, indent=2)
                    log(f"  {key}: {eval_chars/toks:.4f} ch/tok  {round(train_time)}s  rt={fails}")
                    del tokenizer
                    gc.collect()
                    commit_cell(key)
                except Exception as e:
                    log(f"  {key}: FAILED ({type(e).__name__}: {e}); continuing")
                    gc.collect()
        import shutil
        for tag in PRETOKENIZERS:
            shutil.rmtree(os.path.join(CORPORA, f"mg250_{lang}_{tag}"), ignore_errors=True)
        log(f"[{lang}] corpora removed")

    log(f"DONE: {len(results)} cells")


if __name__ == "__main__":
    main()
