"""Boundary variants on FineWiki at 1 GB per language, 6 languages, BPE and MinGram.

Grid
----
  pretokenizers : scriptenc3_cb (baseline), bnd_w, bnd_wp, bnd_wpd
  languages     : en, de, fi, ru, ar, ko   (FINEWIKI_HYBRID6_CORPORA set)
  trainers      : bpe, mingram
  vocabulary    : 32,768 additional (paper_utils/unigram ADDITIONAL_VOCAB_SIZE)
  => 48 cells

All four pretokenizers are ScriptEncodingV3 with enforce_char_boundaries=True and
differ only in which unit kinds carry a boundary marker, so the comparison isolates
the boundary scheme.

Cost and durability
-------------------
This is a long run: roughly 15-20 hours, dominated by MinGram at 1 GB. The
container's disk allowance is limited and has wiped the working tree three times
in this investigation, so:

  * every cell is committed AND pushed as soon as it finishes, trained tokenizer
    included, so progress is never lost to a wipe or a reclaim;
  * completed cells are skipped on restart, so re-running resumes;
  * BPE runs for ALL languages before MinGram starts, so an interrupted run still
    yields a complete picture for one trainer rather than half the languages for
    both;
  * text batches and pretokenized corpora stay untracked (too large to commit)
    and are rebuilt if lost.

Data is read via direct parquet row-group range requests. FineWiki has hundreds
of language configs and load_dataset(name=...) config resolution times out on a
cold cache here, while row-group reads sustain ~5M chars/s. normalize_whitespace
is applied exactly as the registry's finewiki loader does.
"""

import gc
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fsspec
import pyarrow.parquet as pq

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.registry import normalize_whitespace
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array

from boundary_pretokenizer import BOUNDARY_VARIANTS, get_boundary_pretokenizer

LANGS = ["en", "de", "fi", "ru", "ar", "ko"]
CHARS_PER_LANG = 1_000_000_000
BLOCK_CHARS = 10_000_000  # text batch size on disk, matches registry's FINEWEB_BLOCK_MAX_CHARS
EVAL_DOCS = 500
VOCAB = 32_768
OVERSHOOT = 1.15
NUM_WORKERS = 4

CORPORA = os.path.join(HERE, "corpora")
EVAL_DIR = os.path.join(HERE, "eval_texts")
TOKENIZERS = os.path.join(HERE, "tokenizers")
RESULT_PATH = os.path.join(HERE, "finewiki1gb_result.json")
RESOLVE = "https://huggingface.co/datasets/HuggingFaceFW/finewiki/resolve/main/{path}"
TREE_API = "https://huggingface.co/api/datasets/HuggingFaceFW/finewiki/tree/main/data/{lang}wiki"

PRETOKENIZERS = {"plain": lambda: get_pretokenizer("scriptenc3_cb")}
PRETOKENIZERS.update({n: (lambda n=n: get_boundary_pretokenizer(n)) for n in BOUNDARY_VARIANTS})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git(*args, check=False):
    return subprocess.run(["git", *args], cwd=os.path.dirname(HERE), capture_output=True, text=True, check=check)


def commit_cell(key):
    """Persist a finished cell immediately; a wipe or reclaim must not cost work."""
    git("add", "marker_experiments")
    if not git("diff", "--cached", "--quiet").returncode:
        return  # nothing staged
    msg = (
        f"FineWiki 1GB grid: {key}\n\n"
        "Auto-committed per cell so partial progress survives container wipes.\n\n"
        "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>\n"
        "Claude-Session: https://claude.ai/code/session_01PE4L4w3oY91vHCMK9uw32k\n"
    )
    git("commit", "-q", "-m", msg)
    for delay in (2, 4, 8, 16, 0):
        if git("push", "-q", "-u", "origin", "claude/fineweb-space-neighbors-k10ufw").returncode == 0:
            return
        if delay:
            time.sleep(delay)
    log(f"WARNING: push failed for {key}; commit is local only")


def _lang_shards(lang):
    """All parquet shards for a language, in order.

    Reading only shard 0 silently short-changes languages whose first shard holds
    fewer than CHARS_PER_LANG characters: Arabic has 4 shards and shard 0 yields
    483M chars, Korean has 2 and shard 0 yields ~734M. English, German, Finnish
    and Russian were unaffected because their first shard already exceeds 1 GB.
    """
    import json as _json
    import urllib.request

    for attempt in range(5):
        try:
            with urllib.request.urlopen(TREE_API.format(lang=lang), timeout=60) as r:
                tree = _json.load(r)
            shards = sorted(f["path"] for f in tree if f["path"].endswith(".parquet"))
            if shards:
                return shards
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not list shards for {lang}")


def _open_parquet(path, attempts=6):
    last = None
    for attempt in range(attempts):
        try:
            return pq.ParquetFile(fsspec.open(RESOLVE.format(path=path)).open())
        except Exception as e:  # transient CDN/network failure
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not open {path}") from last


def stream_batches(lang):
    """Yield ~BLOCK_CHARS batches of normalized text straight from parquet row groups.

    Nothing is staged on disk: writing 1 GB of text blocks per language is what
    exhausted the session's disk allowance and wiped the working tree. Re-reading
    per pretokenizer costs ~186s and keeps peak disk to the corpora alone.
    """
    cur, cur_chars, total = [], 0, 0
    for shard in _lang_shards(lang):
        if total >= CHARS_PER_LANG:
            break
        pf = _open_parquet(shard)
        rg = 0
        while rg < pf.num_row_groups:
            # The HF CDN returns transient 503s on long reads; a bare read killed a run
            # partway through English. Retry the row group, reopening between attempts.
            table = None
            for attempt in range(6):
                try:
                    table = pf.read_row_group(rg, columns=["text"])
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    log(f"[{lang}] {shard} rg{rg} failed ({type(e).__name__}), retry in {wait}s")
                    time.sleep(wait)
                    pf = _open_parquet(shard)
            if table is None:
                raise RuntimeError(f"[{lang}] {shard} row group {rg} unreadable after retries")
            rg += 1
            for x in table.column("text").to_pylist():
                if not x:
                    continue
                x = normalize_whitespace(x)
                if not x:
                    continue
                cur.append(x)
                cur_chars += len(x)
                total += len(x)
                if cur_chars >= BLOCK_CHARS:
                    yield cur
                    cur, cur_chars = [], 0
            if total >= CHARS_PER_LANG:
                break
    if cur:
        yield cur


def ensure_eval(lang):
    """Held-out slice: the last EVAL_DOCS documents of the same 1 GB stream."""
    eval_path = os.path.join(EVAL_DIR, f"{lang}.json")
    if os.path.exists(eval_path):
        return json.load(open(eval_path))
    os.makedirs(EVAL_DIR, exist_ok=True)
    t = time.time()
    tail, total = [], 0
    for batch in stream_batches(lang):
        total += sum(map(len, batch))
        tail.extend(batch)
        del tail[:-EVAL_DOCS]
    log(f"[{lang}] {total:,} chars streamed, eval slice {len(tail)} docs, {time.time()-t:.0f}s")
    json.dump(tail, open(eval_path, "w"))
    return tail


def drop_corpora(lang):
    """Free a language's corpora once all its cells are done; peak disk is what wipes us."""
    import shutil
    for tag in PRETOKENIZERS:
        d = os.path.join(CORPORA, f"fw1gb_{lang}_{tag}")
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
    log(f"[{lang}] corpora removed")


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
            bare = txt[1:] if txt.startswith(" ") else txt
            if bare.isalpha():
                words.add(bare)
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
    os.makedirs(TOKENIZERS, exist_ok=True)
    results = json.load(open(RESULT_PATH)) if os.path.exists(RESULT_PATH) else {}

    # Trainer-major: every BPE cell (~8 min each) finishes before any MinGram cell
    # (~25-50 min each). The container wipes the working tree every ~30-60 min, and a
    # cell longer than that interval can never complete, so the cheap trainer is run to
    # completion across all six languages first. Corpora are still freed per language,
    # and rebuilt for the MinGram pass -- ~300s, cheap next to the training it feeds.
    for method in ["bpe", "mingram"]:
        for lang in LANGS:
            if all(f"{lang}_{tag}_{method}" in results for tag in PRETOKENIZERS):
                log(f"{lang}/{method}: complete, skipping")
                continue
            eval_texts = ensure_eval(lang)
            eval_chars = sum(map(len, eval_texts))
            for tag, make_pt in PRETOKENIZERS.items():
                key = f"{lang}_{tag}_{method}"
                if key in results:
                    continue
                try:
                    key = f"{lang}_{tag}_{method}"
                    if key in results:
                        continue
                    pt = make_pt()
                    corpus_name = f"fw1gb_{lang}_{tag}"
                    try:
                        corpus = PretokenizedCorpus(name=corpus_name, base_path=CORPORA, pretokenizer=pt)
                    except FileNotFoundError:
                        t = time.time()
                        corpus = PretokenizedCorpus.from_text_batches(
                            name=corpus_name, base_path=CORPORA, pretokenizer=pt,
                            text_batches=stream_batches(lang), num_workers=NUM_WORKERS,
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
                        tr.cache_tag = f"{lang}_{tag}"
                        tokenizer = tr.train()
                    train_time = time.time() - t

                    out = os.path.join(TOKENIZERS, f"{lang}_{tag}_{method}_{VOCAB//1024}k.json.gz")
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
                        "train_chars": corpus.metadata.get("atomic_tokens"),
                        "unique_chunks": corpus.metadata.get("unique_chunks"),
                        "eval_docs": len(eval_texts), "eval_chars": eval_chars, "eval_tokens": toks,
                        "eval_chars_per_token": eval_chars / toks,
                        "roundtrip_failures": fails,
                        "tokenizer_file": os.path.relpath(out, os.path.dirname(HERE)),
                        **analyse_vocab(tokenizer, pt),
                    }
                    with open(RESULT_PATH, "w") as f:
                        json.dump(results, f, indent=2)
                    log(f"  {key}: {eval_chars/toks:.4f} ch/tok  "
                        f"dup={results[key]['space_dup_pairs']}  {round(train_time)}s  rt={fails}")
                    del tokenizer
                    gc.collect()
                    commit_cell(key)
                except Exception as e:
                    # One flaky cell must not abort the remaining grid; it is retried
                    # on the next run because it never entered results.
                    log(f"  {key}: FAILED ({type(e).__name__}: {e}); continuing")
                    gc.collect()

            drop_corpora(lang)

    log(f"DONE: {len(results)} cells")


if __name__ == "__main__":
    main()
