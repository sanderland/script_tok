"""Does splitting digits remove the digit-variant tax?

Measured on the 1 GB grid, delimiting whole digit runs costs English 1,093 vocabulary
slots (3.17%): every distinct number acquires up to four marked forms, so bnd_wpd spends
more entries on numbers than the baseline while covering fewer than half as many.

digit_handling bounds the markable set, because only a run's first and last GROUP can
carry a marker:

    None    every distinct number is markable      (what the 1 GB grid used)
    SPLIT   10 markable strings
    RTL3    1110 markable strings (pretokenizer.py registers exactly 1000+100+10)

This runs en at 1 GB, 32,768 additional vocabulary, BPE, over
{plain, bnd_wpd} x {None, SPLIT, RTL3}. The None cells already exist in the main grid
and are reused.

Fairness note: ScriptPretokenizer.decode has no path for digit-group tokens, so the
stock baseline cannot round-trip with digit_handling set at all (digit_handling was
only ever exercised with UTF8Pretokenizer). The baseline used here is the stock one plus
exactly that decode fix and nothing else, so both sides of the comparison are equally
able to use digit splitting.
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

from marker_experiments.digit_pretokenizer import (
    DigitAwareScriptPretokenizer,
    DigitAwareScriptPretokenizerConfig,
)

from boundary_pretokenizer import BoundaryScriptPretokenizer, BoundaryScriptPretokenizerConfig
import finewiki1gb_grid as G
from finewiki1gb_grid import CORPORA, VOCAB, NUM_WORKERS, analyse_vocab, commit_cell, ensure_eval, log, train_batches

LANG = "en"
# 250M chars, not the 1 GB of the main grid: the digit-variant tax is a property of the
# markable set, not of scale, and at 1 GB each cell needed a 716s corpus build that the
# container's ~30-60 min working-tree wipes kept destroying mid-flight. All six cells here
# are rebuilt at this size so the three digit_handling settings are directly comparable to
# each other; they are NOT comparable to the 1 GB numbers in the main grid.
DIGIT_AXIS_CHARS = 250_000_000
RESULT_PATH = os.path.join(HERE, "digit_split_result.json")
TOKENIZERS = os.path.join(HERE, "tokenizers")


def make_pt(tag, digit_handling):
    if tag == "plain":
        return DigitAwareScriptPretokenizer(
            DigitAwareScriptPretokenizerConfig(
                script_config=ScriptEncodingV3, enforce_char_boundaries=True, digit_handling=digit_handling
            )
        )
    return BoundaryScriptPretokenizer(
        BoundaryScriptPretokenizerConfig(
            script_config=ScriptEncodingV3,
            boundary_targets=("word", "punct", "digit"),
            digit_handling=digit_handling,
        )
    )


def digit_stats(tokenizer, pt):
    """Pure-digit vocabulary entries, distinct numbers covered, slots lost to variants."""
    m = getattr(pt, "marker_token_id", None)
    forms = {}
    entries = 0
    for t in tokenizer.tokens.values():
        ids = list(t.atomic_tokens)
        core = [x for x in ids if x != m] if m is not None else ids
        if not core:
            continue
        txt = pt.try_decode_strict(core)
        if not txt or not txt.isdigit():
            continue
        entries += 1
        key = ("<|>" if m is not None and ids[0] == m else "") + txt + ("<|>" if m is not None and ids[-1] == m else "")
        forms.setdefault(txt, set()).add(key)
    return {
        "pure_digit_entries": entries,
        "distinct_numbers": len(forms),
        "digit_variant_extra_slots": sum(len(v) - 1 for v in forms.values()),
    }


def main():
    os.makedirs(TOKENIZERS, exist_ok=True)
    results = json.load(open(RESULT_PATH)) if os.path.exists(RESULT_PATH) else {}
    G.CHARS_PER_LANG = DIGIT_AXIS_CHARS
    eval_texts = ensure_eval(LANG)
    eval_chars = sum(map(len, eval_texts))

    G.CHARS_PER_LANG = DIGIT_AXIS_CHARS  # applies to train_batches/ensure_eval below
    for digit_handling in ["None", "SPLIT", "RTL3"]:
        for tag in ["plain", "bnd_wpd"]:
            key = f"{LANG}_{tag}_{digit_handling}"
            if key in results:
                log(f"{key}: done, skipping")
                continue
            pt = make_pt(tag, None if digit_handling == "None" else digit_handling)
            corpus_name = f"digitsplit250_{LANG}_{tag}_{digit_handling}"
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
                "lang": LANG, "pretokenizer": tag, "digit_handling": digit_handling,
                "atomic_vocab": len(pt.atomic_tokens), "vocab_size": len(tokenizer.tokens),
                "train_seconds": round(train_time),
                "unique_chunks": corpus.metadata.get("unique_chunks"),
                "eval_chars": eval_chars, "eval_tokens": toks,
                "eval_chars_per_token": eval_chars / toks,
                "roundtrip_failures": fails,
                **analyse_vocab(tokenizer, pt), **digit_stats(tokenizer, pt),
            }
            with open(RESULT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            log(f"  {key}: {eval_chars/toks:.4f} ch/tok  digit_variants="
                f"{results[key]['digit_variant_extra_slots']}  {round(train_time)}s  rt={fails}")
            commit_cell(key)

    log(f"DONE: {len(results)} cells")


if __name__ == "__main__":
    main()
