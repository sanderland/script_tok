"""Boundary markers across the hybrid/ 6-language FineWiki set.

Languages are the ones in FINEWIKI_HYBRID6_CORPORA (script_bpe/corpus/registry.py):
en, de, fi, ru, ar, ko -- Latin x3, Cyrillic, Arabic, Hangul. All six are in
DEFAULT_SCRIPTS_LM_WITH_SPACES, so all six are word-wrapped by the marker schemes.

Deviations from the registry's finewiki_{lang}_1gb, and why:
  * CHARS_PER_LANG below the registry's 1 GB cap, for compute budget. Everything
    else is matched: same dataset, same normalize_whitespace transform the
    registry applies to finewiki (which collapses [ \\t]+ to a single space, so
    multi-space runs are absent by construction here).
  * Data is read via direct parquet row-group range requests rather than
    load_dataset(name=...). FineWiki has hundreds of language configs and config
    resolution times out on a cold cache in this environment, while row-group
    range reads run at ~5M chars/s.

Three pretokenizers, all on ScriptEncodingV3 with enforce_char_boundaries:
  plain : scriptenc3_cb, the current baseline
  v4    : markers on word spans (unconditional) + punctuation (space side only)
  v5    : v4 + digits markable (space side only)

Results are written after every cell, and completed cells are skipped on re-run,
so this survives interruption.
"""

import os
import sys
import json
import time
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fsspec
import pyarrow.parquet as pq

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3
from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.registry import normalize_whitespace
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array

from scriptenc_marker_v4 import MarkerV4Pretokenizer, MarkerV4PretokenizerConfig
from scriptenc_marker_v5 import MarkerV5Pretokenizer, MarkerV5PretokenizerConfig

LANGS = ["en", "de", "fi", "ru", "ar", "ko"]
CHARS_PER_LANG = 100_000_000
EVAL_DOCS = 500
BPE_VOCABS = [16_000, 32_000, 64_000]
MINGRAM_VOCABS = [64_000]
MINGRAM_LANGS = ["en", "ru", "ko"]  # one per script family, to check trainer-independence
OVERSHOOT = 1.15
NUM_WORKERS = 4

CORPORA = os.path.join(HERE, "corpora")
EVAL_DIR = os.path.join(HERE, "eval_texts")
RESULT_PATH = os.path.join(HERE, "multilang_result.json")
URL = "https://huggingface.co/datasets/HuggingFaceFW/finewiki/resolve/main/data/{lang}wiki/000_00000.parquet"

PRETOKENIZERS = {
    "plain": lambda: get_pretokenizer("scriptenc3_cb"),
    "v4": lambda: MarkerV4Pretokenizer(MarkerV4PretokenizerConfig(script_config=ScriptEncodingV3)),
    "v5": lambda: MarkerV5Pretokenizer(MarkerV5PretokenizerConfig(script_config=ScriptEncodingV3)),
}


def read_lang(lang):
    """Stream row groups until CHARS_PER_LANG, applying the registry's finewiki transform."""
    t = time.time()
    f = fsspec.open(URL.format(lang=lang)).open()
    pf = pq.ParquetFile(f)
    texts, total = [], 0
    for rg in range(pf.num_row_groups):
        for x in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            if not x:
                continue
            x = normalize_whitespace(x)
            if not x:
                continue
            texts.append(x)
            total += len(x)
        if total >= CHARS_PER_LANG:
            break
    print(f"[{lang}] {len(texts):,} docs, {total:,} chars in {time.time()-t:.0f}s", flush=True)
    return texts


class CachedInitMinGramTrainer(MinGramTrainer):
    cache_tag = "unknown"

    def _build_bpe_init_tokens(self):
        size = int(self.config.additional_vocab_size * self.config.overshoot_factor)
        path = os.path.join(HERE, "bpe_init_cache", f"{self.cache_tag}_{size}.json.gz")
        if os.path.exists(path):
            self.logger.info(f"Reusing cached BPE init {path}")
            m = BPETokenizer.load(path)
            m.pretokenizer = self.pretokenizer
        else:
            cfg = BPETrainerConfig(additional_vocab_size=size, num_workers=self.config.num_workers)
            m = BPETrainer(self.pretokenizer, self.corpus, cfg).train()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            m.save(path)
        tot = sum(max(1, t.current_count) for t in m.tokens.values())
        return [
            UnigramToken(id=t.id, atomic_tokens=token_array(t.atomic_tokens),
                         log_prob=math.log(max(1, t.current_count) / tot),
                         required=len(t.atomic_tokens) == 1)
            for t in m.tokens.values()
        ]


def analyse_vocab(tokenizer, pt):
    """Duplicate-pair vocabulary tax, marker-variant cost, and word coverage."""
    marker_id = getattr(pt, "marker_token_id", None)
    words, variants, ws_only = set(), {}, 0
    by_text = {}
    for t in tokenizer.tokens.values():
        ids = list(t.atomic_tokens)
        by_text.setdefault(pt.decode(ids), []).append(t)
        core = [x for x in ids if x != marker_id] if marker_id is not None else ids
        if not core:
            continue
        txt = pt.try_decode_strict(core)
        if txt is None:
            continue
        if txt and txt.strip() == "":
            ws_only += 1
        elif marker_id is not None:
            if txt.isalpha() and ids[0] == marker_id and ids[-1] == marker_id and ids.count(marker_id) == 2:
                words.add(txt)
            elif txt and not any(c.isalnum() or c.isspace() for c in txt):
                key = ("<|>" if ids[0] == marker_id else "") + txt + ("<|>" if ids[-1] == marker_id else "")
                variants.setdefault(txt, set()).add(key)
        else:
            words.add(txt[1:] if txt.startswith(" ") else txt) if (
                (txt[1:] if txt.startswith(" ") else txt).isalpha()) else None
    dups = [x for x in by_text if x.startswith(" ") and len(x) > 1 and x[1:] in by_text]
    dup_slots = sum(len(by_text[x]) + len(by_text[x[1:]]) for x in dups)
    return {
        "distinct_alpha_words_with_own_token": len(words),
        "space_dup_pairs": len(dups),
        "space_dup_vocab_frac": dup_slots / len(tokenizer.tokens),
        "marker_variant_extra_slots": sum(len(v) - 1 for v in variants.values()),
        "whitespace_only_vocab_entries": ws_only,
    }


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    results = json.load(open(RESULT_PATH)) if os.path.exists(RESULT_PATH) else {}
    jobs = [("bpe", v) for v in BPE_VOCABS] + [("mingram", v) for v in MINGRAM_VOCABS]

    for lang in LANGS:
        needed = [
            f"{lang}_{tag}_{m}_{v//1000}k"
            for tag in PRETOKENIZERS for m, v in jobs
            if not (m == "mingram" and lang not in MINGRAM_LANGS)
        ]
        if all(k in results for k in needed):
            print(f"=== {lang}: all cells present, skipping ===", flush=True)
            continue

        eval_path = os.path.join(EVAL_DIR, f"{lang}.json")
        train_texts = None
        if os.path.exists(eval_path):
            eval_texts = json.load(open(eval_path))
        else:
            texts = read_lang(lang)
            eval_texts, train_texts = texts[-EVAL_DOCS:], texts[:-EVAL_DOCS]
            json.dump(eval_texts, open(eval_path, "w"))
        eval_chars = sum(map(len, eval_texts))

        for tag, make_pt in PRETOKENIZERS.items():
            corpus_name = f"finewiki6_{lang}_{tag}"
            pt = make_pt()
            try:
                corpus = PretokenizedCorpus(name=corpus_name, base_path=CORPORA, pretokenizer=pt)
            except FileNotFoundError:
                if train_texts is None:
                    texts = read_lang(lang)
                    eval_texts = texts[-EVAL_DOCS:]
                    train_texts = texts[:-EVAL_DOCS]
                    eval_chars = sum(map(len, eval_texts))
                t = time.time()
                corpus = PretokenizedCorpus.from_texts(
                    name=corpus_name, base_path=CORPORA, pretokenizer=pt,
                    texts=train_texts, num_workers=NUM_WORKERS,
                )
                print(f"[{lang}/{tag}] corpus built in {time.time()-t:.0f}s "
                      f"unique_chunks={corpus.metadata.get('unique_chunks'):,}", flush=True)

            for method, vocab in jobs:
                if method == "mingram" and lang not in MINGRAM_LANGS:
                    continue
                key = f"{lang}_{tag}_{method}_{vocab//1000}k"
                if key in results:
                    continue
                t = time.time()
                if method == "bpe":
                    tokenizer = BPETrainer(
                        pt, corpus, BPETrainerConfig(additional_vocab_size=vocab, num_workers=NUM_WORKERS)
                    ).train()
                else:
                    tr = CachedInitMinGramTrainer(
                        pt, corpus,
                        MinGramTrainerConfig(additional_vocab_size=vocab, num_workers=NUM_WORKERS,
                                             overshoot_factor=OVERSHOOT),
                    )
                    tr.cache_tag = f"{lang}_{tag}"
                    tokenizer = tr.train()
                train_time = time.time() - t

                toks = fails = 0
                for text in eval_texts:
                    ids = tokenizer.encode(text)
                    toks += len(ids)
                    if tokenizer.decode(ids) != text:
                        fails += 1

                results[key] = {
                    "lang": lang, "pretokenizer": tag, "method": method,
                    "additional_vocab_size": vocab, "vocab_size": len(tokenizer.tokens),
                    "train_seconds": round(train_time),
                    "unique_chunks": corpus.metadata.get("unique_chunks"),
                    "eval_docs": len(eval_texts), "eval_chars": eval_chars, "eval_tokens": toks,
                    "eval_chars_per_token": eval_chars / toks,
                    "roundtrip_failures": fails,
                    **analyse_vocab(tokenizer, pt),
                }
                print(f"  {key}: {eval_chars/toks:.4f} ch/tok  dup={results[key]['space_dup_pairs']}  "
                      f"{round(train_time)}s  rt={fails}", flush=True)
                with open(RESULT_PATH, "w") as f:
                    json.dump(results, f, indent=2)
        train_texts = None  # free before the next language

    print("\n=== DONE ===")
    print(f"{len(results)} cells in {RESULT_PATH}")


if __name__ == "__main__":
    main()
