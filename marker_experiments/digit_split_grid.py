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
from script_bpe.pretokenize.pretokenizer import ScriptPretokenizer, ScriptPretokenizerConfig
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig

from boundary_pretokenizer import BoundaryScriptPretokenizer, BoundaryScriptPretokenizerConfig
from finewiki1gb_grid import CORPORA, VOCAB, NUM_WORKERS, analyse_vocab, commit_cell, ensure_eval, log, stream_batches

LANG = "en"
RESULT_PATH = os.path.join(HERE, "digit_split_result.json")
TOKENIZERS = os.path.join(HERE, "tokenizers")


class DigitAwareScriptPretokenizerConfig(ScriptPretokenizerConfig):
    cls: str = "DigitAwareScriptPretokenizer"


class DigitAwareScriptPretokenizer(ScriptPretokenizer, config_type=DigitAwareScriptPretokenizerConfig):
    """Stock baseline plus the two fixes `digit_handling` needs on ScriptPretokenizer.

    ScriptPretokenizer does not support digit_handling at all -- it was only ever
    exercised with UTF8Pretokenizer -- and fails in two independent places:

      * split_encoded raises ValueError on any chunk that is not entirely ScriptCharEnc,
        and a digit chunk is entirely DigitsEnc;
      * decode has no path for digit-group tokens, so every digit becomes U+FFFD.

    Both are fixed here and nothing else differs from scriptenc3_cb, so the baseline and
    the boundary variant are equally able to use digit splitting.
    """

    def __init__(self, config):
        super().__init__(config)
        self.digit_token_ids = {tid for tid, txt in self.atomic_tokens.items() if txt.isdigit()}

    def split_encoded(self, encoding):
        if any(getattr(c, "script_id", 0) == -1 for c in encoding):
            return [encoding]  # a digit chunk is already its own pretoken
        return super().split_encoded(encoding)

    def decode(self, tokenization, errors="replace") -> str:
        decoded = ""
        i = 0
        n = len(tokenization)
        while i < n:
            if tokenization[i] in self.digit_token_ids:
                decoded += self.atomic_tokens[tokenization[i]]
                i += 1
                continue
            script_tok = tokenization[i]
            ix_tok = tokenization[i + 1] if i + 1 < n else None
            if (script_tok, ix_tok) in self.detokenize_map:
                decoded += self.detokenize_map[(script_tok, ix_tok)]
                i += 2
            else:
                if errors == "backslashreplace":
                    decoded += self.atomic_tokens[script_tok]
                elif errors == "replace":
                    decoded += "�"
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok})")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        return decoded


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
    eval_texts = ensure_eval(LANG)
    eval_chars = sum(map(len, eval_texts))

    for digit_handling in ["SPLIT", "RTL3"]:
        for tag in ["plain", "bnd_wpd"]:
            key = f"{LANG}_{tag}_{digit_handling}"
            if key in results:
                log(f"{key}: done, skipping")
                continue
            pt = make_pt(tag, digit_handling)
            corpus_name = f"digitsplit_{LANG}_{tag}_{digit_handling}"
            try:
                corpus = PretokenizedCorpus(name=corpus_name, base_path=CORPORA, pretokenizer=pt)
            except FileNotFoundError:
                t = time.time()
                corpus = PretokenizedCorpus.from_text_batches(
                    name=corpus_name, base_path=CORPORA, pretokenizer=pt,
                    text_batches=stream_batches(LANG), num_workers=NUM_WORKERS,
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
