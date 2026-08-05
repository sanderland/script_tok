#!/usr/bin/env python3
"""Measure compression of the FineWeb grid on a held-out Goldfish slice.

Why not the slice the manifest already records
----------------------------------------------
`train_multilang.py` can record compression against an `--eval-texts` slice as it trains,
but that number is not one measurement: the tokenizers train on `fineweb_{lang}_5gb` while
the slice held out by the older grids was FineWiki, so the pairing is cross-domain by
accident rather than by design; the slice differed per language; and most of the grid was
trained on machines that had no slice at all and so recorded nothing.

Goldfish "fish-food" (Chang et al., LREC 2026, arXiv 2408.10441) is the held-out eval the
hybrid paper's headline `fineweb -> fishfood` pairing already uses, so this puts the
marker grid on that same axis and gives every cell one comparable number.

The slice
---------
The registry's `<lang>_<script>_fishfood` corpus is the whole file, 7-15 GB per language
and 64 GB for the six together -- more than this repo's usual disk, and far more than a
chars/token measurement needs. Instead the slice is `CHUNKS` byte ranges at even offsets
across the *whole* file, each contributing an equal share of the character budget.

Spreading over the whole file rather than reading a prefix is not a detail: these files
are ordered by source and the first tenth or so is SEO keyword spam, which is far more
compressible than ordinary web text. The equal per-chunk budget is what keeps the spread:
a plain "read until the budget is full" loop stops inside the first few chunks and is a
prefix sample again.

Offsets are a function of the file size alone, so the slice is deterministic and
rebuildable, and each language's slice hash is recorded beside its numbers. `normalize_whitespace`
is applied, matching how the FineWeb training corpora are built -- without it the arms
would be compared on whitespace runs that no arm saw in training, which is exactly the
axis the boundary marker changes.

    uv run python paper_utils/boundary/downstream/eval_goldfish.py
    uv run python paper_utils/boundary/downstream/eval_goldfish.py --langs ko --force
"""

import collections
import concurrent.futures
import glob
import hashlib
import json
import math
import os

import cyclopts
import requests
from huggingface_hub import hf_hub_url

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.registry import SAMPLE_CACHE_DIRNAME, normalize_whitespace
from script_bpe.tokenizers.load import load_tokenizer

# Registers BoundaryScriptPretokenizer, without which loading any bnd_* tokenizer raises
# KeyError from the pretokenizer registry.
import paper_utils.boundary.downstream.boundary_tokenizer  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATED = os.path.join(os.path.dirname(HERE), "paper", "generated")
TOKENIZERS = os.path.join(HERE, "tokenizers")
SLICE_DIR = os.path.join(os.path.dirname(HERE), "eval_texts")
OUT = os.path.join(GENERATED, "eval_goldfish.json")

REPO_ID = "goldfish-models/fish-food"
# The hybrid paper's LANGUAGE_CONFIGS mapping (paper_utils/hybrid/generate_train_eval_compression_grid.py).
GOLDFISH_FILE = {
    "en": "eng_latn.txt",
    "de": "deu_latn.txt",
    "fi": "fin_latn.txt",
    "ru": "rus_cyrl.txt",
    "ar": "arb_arab.txt",
    "ko": "kor_hang.txt",
}
CHUNKS = 64              # byte ranges per language, at even offsets across the file
CHUNK_BYTES = 1 << 19    # 512 KB read per chunk: enough to fill the per-chunk character
                         # budget even for 3-bytes-per-character Hangul.
SLICE_CHARS = 8_000_000

app = cyclopts.App()


def build_slice(lang: str) -> list[str]:
    """The held-out slice for one language, cached under eval_texts/goldfish_{lang}.json."""
    path = os.path.join(SLICE_DIR, f"goldfish_{lang}.json")
    if os.path.exists(path):
        return json.load(open(path))

    url = hf_hub_url(REPO_ID, GOLDFISH_FILE[lang], repo_type="dataset")
    size = int(requests.head(url, allow_redirects=True, timeout=60).headers["content-length"])
    docs, chars, budget = [], 0, SLICE_CHARS // CHUNKS
    for i in range(CHUNKS):
        start = size * i // CHUNKS
        r = requests.get(
            url, headers={"Range": f"bytes={start}-{start + CHUNK_BYTES - 1}"}, timeout=120
        )
        r.raise_for_status()
        # First and last lines are cut mid-document by the byte range, and the first is
        # additionally cut mid-codepoint; drop both rather than feed a fragment to the
        # tokenizer.
        taken = 0
        for line in r.content.decode("utf-8", errors="ignore").split("\n")[1:-1]:
            line = normalize_whitespace(line)
            if line:
                docs.append(line)
                taken += len(line)
            if taken >= budget:     # per-chunk, so every offset contributes equally
                break
        chars += taken
    os.makedirs(SLICE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(docs, f)
    print(f"[slice] {lang}: {len(docs):,} docs, {chars:,} chars from {CHUNKS} chunk(s) of "
          f"{GOLDFISH_FILE[lang]} ({size / 1e9:.1f} GB)")
    return docs


def slice_hash(docs: list[str]) -> str:
    """Identifies the slice a number was measured on, so a rebuilt slice is detectable."""
    h = hashlib.sha256()
    for doc in docs:
        h.update(doc.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def train_chars(corpus: str) -> int | None:
    """Characters in the training corpus, from the sample cache's manifest.

    `selected_chars` is what the sampler actually kept, which is not the 5 GB it aimed for
    (Korean landed at 4.99 GB), so the nominal budget would be wrong in the fourth decimal
    of a characters-per-token figure. None when the corpus was built on another machine
    and its cache is not here; only the absolute baseline needs this, since characters
    cancel in every percentage against it.
    """
    root = os.path.join(PretokenizedCorpus.DEFAULT_BASE_PATH, SAMPLE_CACHE_DIRNAME)
    # The cache key is a hash of the sampling config, so glob it -- but `fineweb_de_5gb_*`
    # also matches `fineweb_de_5gb_quick_*`, hence the single-token suffix check.
    for path in sorted(glob.glob(os.path.join(root, f"{corpus}_*", "manifest.json"))):
        if "_" in os.path.basename(os.path.dirname(path)).removeprefix(f"{corpus}_"):
            continue
        return json.load(open(path))["selected_chars"]
    return None


def train_tokens(tokenizer) -> int:
    """Tokens the tokenizer's own training corpus takes, read from the saved model.

    Not measured here, and it does not need to be: both trainers already record it.
    MinGram writes `total_tokens` into its metadata, and BPE maintains a per-token
    `current_count` -- decremented for both constituents and set for the new token on
    every merge -- so at the end of training the counts sum to exactly the corpus token
    count under the final vocabulary. Re-deriving either would cost a pass over a 5 GB
    corpus per arm.
    """
    total = tokenizer.metadata.get("total_tokens")
    return total if total is not None else sum(t.current_count for t in tokenizer.tokens.values())


def renyi_bits(freqs: collections.Counter, alpha: float) -> float:
    """Renyi entropy of the token distribution, in bits.

    Same definition as `Tokenizer.corpus_performance` and so as the hybrid paper's table,
    reimplemented here only because that method takes a registered `PretokenizedCorpus`
    and this slice is a list of documents.
    """
    n = sum(freqs.values())
    ps = [c / n for c in freqs.values() if c]
    if alpha == 1.0:
        return -sum(p * math.log2(p) for p in ps)
    return math.log2(sum(p**alpha for p in ps)) / (1.0 - alpha)


def evaluate(args) -> tuple[str, dict]:
    key, path, lang, corpus = args
    tokenizer = load_tokenizer(path)
    docs = build_slice(lang)  # cached on disk by the parent before the pool starts
    chars = toks = fails = 0
    words = sum(len(doc.split()) for doc in docs)
    freqs = collections.Counter()
    for doc in docs:
        ids = tokenizer.encode(doc)
        chars += len(doc)
        toks += len(ids)
        freqs.update(int(i) for i in ids)
        # Against the normalized text, which is all a round trip can return: the
        # pretokenizer applies NFC, so e.g. an Arabic shadda/fatha sequence comes back
        # canonically reordered. Comparing against the raw document counted that as a
        # loss and made 3.3% of the Arabic slice look broken under every arm alike.
        if tokenizer.decode(ids) != tokenizer.pretokenizer.normalize(doc):
            fails += 1
    return key, {
        "train_corpus": corpus,
        "train_chars": train_chars(corpus),
        "train_tokens": train_tokens(tokenizer),
        "eval_corpus": f"goldfish:{GOLDFISH_FILE[lang]}",
        "eval_docs": len(docs),
        "eval_chars": chars,
        "eval_tokens": toks,
        "eval_chars_per_token": chars / toks,
        # Fertility. Whitespace words are not comparable across languages -- a Korean eojeol
        # and an Arabic clitic host are not the same unit -- but the word count is a property
        # of the slice, so within a language this is chars-per-word over chars-per-token and
        # every arm is measured against the same denominator.
        "eval_words": words,
        "eval_tokens_per_word": toks / words,
        # Renyi-3 efficiency (Zouhar et al. 2023; alpha and normalization as in Cognetta
        # et al. 2024), matching the hybrid paper's appendix table.
        "eval_nonzero_vocab": len(freqs),
        "eval_shannon_bits": renyi_bits(freqs, 1.0),
        "eval_renyi3_bits": renyi_bits(freqs, 3.0),
        "eval_renyi3_efficiency": renyi_bits(freqs, 3.0) / math.log2(len(freqs)),
        "roundtrip_failures": fails,
        "slice_hash": slice_hash(docs),
    }


@app.default
def main(
    tokenizers: str = TOKENIZERS,
    out: str = OUT,
    langs: str = ",".join(GOLDFISH_FILE),
    workers: int = 6,
    force: bool = False,
) -> None:
    """Evaluate every grid tokenizer on its language's Goldfish slice.

    Args:
        tokenizers: Directory of `<corpus>_<arm>_<trainer>_v<vocab>.json.gz` files.
        out: JSON to write, one entry per tokenizer key.
        langs: Comma-separated languages to cover.
        workers: Parallel tokenizers. Each holds one slice in memory, so this is the
            memory knob as much as the speed one.
        force: Re-measure keys that are already in `out`.
    """
    wanted = [x.strip() for x in langs.split(",") if x.strip()]
    results = json.load(open(out)) if os.path.exists(out) else {}

    jobs = []
    for name in sorted(os.listdir(tokenizers)):
        if not name.endswith(".json.gz"):
            continue
        key = name[: -len(".json.gz")]
        lang = key.split("_")[1]  # fineweb_<lang>_5gb[_quick]_<arm>_<trainer>_v<vocab>
        if lang not in wanted:
            continue
        if key in results and not force:
            continue
        corpus = key[: key.index(f"_{lang}_") + len(f"_{lang}_5gb")]
        if "_quick_" in key:
            corpus += "_quick"
        jobs.append((key, os.path.join(tokenizers, name), lang, corpus))

    if not jobs:
        print(f"[eval] nothing to do: {len(results)} key(s) already in {out}")
        return

    for lang in sorted({job[2] for job in jobs}):  # build once here, not per worker
        build_slice(lang)

    print(f"[eval] {len(jobs)} tokenizer(s) on {workers} worker(s)")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for key, info in pool.map(evaluate, jobs):
            results[key] = info
            print(f"  {key:<52} ch/tok={info['eval_chars_per_token']:.4f} "
                  f"rt_fail={info['roundtrip_failures']}")
            with open(out, "w") as f:  # written as they land: this is a long run
                json.dump(results, f, indent=2, sort_keys=True)

    print(f"[eval] {out}: {len(results)} key(s)")


if __name__ == "__main__":
    app()
