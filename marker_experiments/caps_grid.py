"""Do caps codes pay for themselves?

Same duplication argument as the leading space, applied to case: without caps codes a
vocabulary holds 'The' and 'the', 'NASA' and 'nasa' as separate entries. With them, a
title-case span is a shift code plus the lowercased form, so the pieces are shared.

The cost is one extra token per capitalised span, and sentence-initial capitals are very
frequent, so this can easily come out negative. Section 5.3 already showed that reclaiming
vocabulary does not automatically buy compression: removing a 3.17% digit-variant tax was
worth +0.33pp.

en, 250M characters, 32,768 additional vocabulary, BPE, evaluation withheld from training
-- the same setup as the digit axis, so the plain and bnd_wpd cells there are directly
comparable and are reused.
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig

import finewiki1gb_grid as G
from boundary_pretokenizer import BoundaryScriptPretokenizer, BoundaryScriptPretokenizerConfig
from finewiki1gb_grid import CORPORA, NUM_WORKERS, VOCAB, analyse_vocab, commit_cell, ensure_eval, log, train_batches

LANG = "en"
CHARS = 250_000_000
RESULT_PATH = os.path.join(HERE, "caps_result.json")
TOKENIZERS = os.path.join(HERE, "tokenizers")

VARIANTS = {
    "bnd_wpd": dict(caps_codes=False),
    "bnd_wpd_caps": dict(caps_codes=True),
}


def make_pt(**kw):
    return BoundaryScriptPretokenizer(
        BoundaryScriptPretokenizerConfig(
            script_config=ScriptEncodingV3, boundary_targets=("word", "punct", "digit"), **kw
        )
    )


def case_stats(tokenizer, pt):
    """Vocabulary spent on case variants of the same word.

    Counts entries whose decoded text has a distinct-cased counterpart also in the
    vocabulary ('The'/'the'), which is the case analogue of the ' X'/'X' pair count.
    """
    marker = getattr(pt, "marker_token_id", None)
    codes = {pt.shift_token_id, pt.caps_token_id} - {None}
    texts = {}
    for t in tokenizer.tokens.values():
        ids = [x for x in t.atomic_tokens if x != marker and x not in codes]
        if not ids:
            continue
        txt = pt.try_decode_strict(ids)
        if txt and txt.isalpha():
            texts.setdefault(txt, 0)
            texts[txt] += 1
    pairs = 0
    seen = set()
    for txt in texts:
        if txt.islower() or txt in seen:
            continue
        low = txt.lower()
        if low != txt and low in texts:
            pairs += 1
            seen.add(txt)
            seen.add(low)
    return {
        "alpha_entries": len(texts),
        "case_dup_pairs": pairs,
        "case_dup_vocab_frac": 2 * pairs / len(tokenizer.tokens),
    }


def main():
    os.makedirs(TOKENIZERS, exist_ok=True)
    G.CHARS_PER_LANG = CHARS
    results = json.load(open(RESULT_PATH)) if os.path.exists(RESULT_PATH) else {}
    eval_texts = ensure_eval(LANG)
    eval_chars = sum(map(len, eval_texts))
    log(f"eval: {len(eval_texts)} docs, {eval_chars:,} chars")

    for tag, kw in VARIANTS.items():
        key = f"{LANG}_{tag}"
        if key in results:
            log(f"{key}: done, skipping")
            continue
        pt = make_pt(**kw)
        corpus_name = f"caps250_{LANG}_{tag}"
        try:
            corpus = PretokenizedCorpus(name=corpus_name, base_path=CORPORA, pretokenizer=pt)
        except FileNotFoundError:
            t = time.time()
            corpus = PretokenizedCorpus.from_text_batches(
                name=corpus_name, base_path=CORPORA, pretokenizer=pt,
                text_batches=train_batches(LANG), num_workers=NUM_WORKERS,
            )
            log(f"{key}: corpus built in {time.time()-t:.0f}s "
                f"unique_chunks={corpus.metadata.get('unique_chunks'):,}")

        t = time.time()
        tokenizer = BPETrainer(
            pt, corpus, BPETrainerConfig(additional_vocab_size=VOCAB, num_workers=NUM_WORKERS)
        ).train()
        train_time = time.time() - t
        out = os.path.join(TOKENIZERS, f"{key}_bpe_32k.json.gz")
        tokenizer.save(out)

        toks = fails = 0
        for text in eval_texts:
            ids = tokenizer.encode(text)
            toks += len(ids)
            if tokenizer.decode(ids) != text:
                fails += 1

        results[key] = {
            "lang": LANG, "variant": tag, "caps_codes": kw["caps_codes"],
            "atomic_vocab": len(pt.atomic_tokens), "vocab_size": len(tokenizer.tokens),
            "train_seconds": round(train_time),
            "unique_chunks": corpus.metadata.get("unique_chunks"),
            "eval_chars": eval_chars, "eval_tokens": toks,
            "eval_chars_per_token": eval_chars / toks,
            "roundtrip_failures": fails,
            **analyse_vocab(tokenizer, pt), **case_stats(tokenizer, pt),
        }
        with open(RESULT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  {key}: {eval_chars/toks:.4f} ch/tok  case_pairs="
            f"{results[key]['case_dup_pairs']}  {round(train_time)}s  rt={fails}")
        commit_cell(key)

    log(f"DONE: {len(results)} cells")


if __name__ == "__main__":
    main()
