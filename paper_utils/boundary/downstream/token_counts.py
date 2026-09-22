#!/usr/bin/env python3
"""Count every vocabulary entry over a language's training shards.

One complete pass over the training text, validation shard excluded, so the counts are the
corpus frequencies a tokenizer would meet, not the exposure of any one run: the loader
repeats, permutes and crops, and reproducing that needs the loader rather than a count.

Written for the question of whether splitting a word's frequency across `word` and `' word'`
in the baseline explains what the trained embeddings look like, so it records what is needed
to line entries up across schemes: the count for every entry including the zeros, each
entry's surface form with the markers and case codes left visible, and the hashes of both
the tokenizer and every shard read.

script_bpe ids are sparse: a MinGram vocabulary of 34,685 entries runs to id 39,630. The
models never see those ids. `ScriptBPETokenizerAdapter` sorts them and remaps to a dense
space, `old_ids = sorted(tokenizer.tokens)`, so embedding row `d` is `ids[d]` here, and row
34,685 is the synthetic beginning-of-sequence token, which no text produces and which is
therefore absent from these counts. Every array below is in that same sorted order, so index
`d` is embedding row `d` throughout.

    python -m paper_utils.boundary.downstream.token_counts count \
        --tokenizer paper_utils/boundary/downstream/tokenizers/fineweb_ko_5gb_quick_plain_mingram_v34685.json.gz \
        --base-dir /capstor/scratch/cscs/$USER/marker_downstream/nanochat_base_ko \
        --out counts_ko_plain_mingram.json.gz

    python -m paper_utils.boundary.downstream.token_counts pair \
        --plain counts_ko_plain_mingram.json.gz --marked counts_ko_bnd_w_mingram.json.gz \
        --out pairs_ko_mingram.tsv
"""
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import cyclopts
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from paper_utils.boundary.vocab_duplicates import surface  # noqa: E402

app = cyclopts.App()
_TOKENIZER = None


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: str):
    """The tokenizer, by the class its file names, with the legacy classes registered."""
    from paper_utils.boundary.downstream.boundary_tokenizer import BPETokenizer, MinGramModel

    return (MinGramModel if "_mingram_" in os.path.basename(path) else BPETokenizer).load(path)


def special_texts(pretokenizer) -> dict[int, str]:
    """Marker and case-code atomics, mapped to how they should read.

    Duck-typed rather than `isinstance`, so an August tokenizer read through the legacy class
    keeps its markers visible instead of collapsing entries that differ only by one.
    """
    if not hasattr(pretokenizer, "marker_token_id"):
        return {}
    codes = {getattr(pretokenizer, "shift_token_id", None): pretokenizer.SHIFT_TEXT,
             getattr(pretokenizer, "caps_token_id", None): pretokenizer.CAPS_TEXT}
    return {pretokenizer.marker_token_id: pretokenizer.MARKER_TEXT,
            **{tid: text for tid, text in codes.items() if tid is not None}}


def _init(path):
    global _TOKENIZER
    _TOKENIZER = load(path)


def _count_batch(docs) -> Counter:
    counts = Counter()
    for doc in docs:
        counts.update(_TOKENIZER.encode(doc))
    return counts


def _batches(paths, size=200):
    for path in paths:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            column = pf.read_row_group(rg, columns=["text"]).column("text")
            batch = []
            for value in column:
                batch.append(value.as_py())
                if len(batch) == size:
                    yield batch
                    batch = []
            if batch:
                yield batch


@app.command
def count(tokenizer: Path, base_dir: Path, out: Path, workers: int = 48):
    """Count every entry of `tokenizer` over the training shards under `base_dir`.

    Args:
        tokenizer: the tokenizer file, as published.
        base_dir: the NANOCHAT_BASE holding base_data_climbmix/ and shard_provenance.json.
        out: gzipped JSON to write.
        workers: encoding processes. script_bpe encodes in pure Python, so this is the
            wall-clock knob; keep OMP_NUM_THREADS at 1 to stay inside the login node's
            per-user thread limit.
    """
    provenance = json.loads((base_dir / "shard_provenance.json").read_text())
    shards = [s for s in provenance["shards"] if s["role"] == "train"]
    if not shards:
        raise SystemExit(f"{base_dir}: provenance records no training shards")
    paths = [(base_dir / "base_data_climbmix" / s["shard"]).resolve() for s in shards]
    for path, shard in zip(paths, shards):
        if path.stat().st_size != shard["bytes"]:
            raise SystemExit(f"{path} is not the size the provenance records")

    tok = load(str(tokenizer))
    specials = special_texts(tok.pretokenizer)
    # Sorted, because that is the order the adapter gives the model: dense row d is ids[d].
    ids = sorted(tok.tokens)
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{tokenizer}: repeated ids in the vocabulary")
    forms = [surface(tok.pretokenizer, tok.tokens[i].atomic_tokens, specials) for i in ids]

    totals = Counter()
    with Pool(workers, initializer=_init, initargs=(str(tokenizer),)) as pool:
        for done, batch_counts in enumerate(pool.imap_unordered(_count_batch, _batches(paths), chunksize=1), 1):
            totals.update(batch_counts)
            if done % 2000 == 0:
                print(f"  {done * 200:,} documents, {sum(totals.values()):,} tokens", flush=True)
    counts = [totals.get(i, 0) for i in ids]
    unknown = set(totals) - set(ids)
    if unknown:
        raise SystemExit(f"{tokenizer}: encoding produced {len(unknown)} id(s) outside the vocabulary")

    record = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "language": provenance["language"],
        "tokenizer": os.path.basename(str(tokenizer)),
        "tokenizer_sha256": sha256(tokenizer),
        "vocab_size": len(forms),
        "id_space": "script_bpe ids, sparse; index d of every array below is embedding row d",
        "ids": ids,
        "bos_dense_row": len(forms),
        "source_dataset": provenance["source_dataset"],
        "shards": [{k: s[k] for k in ("shard", "source", "bytes", "sha256") if k in s} for s in shards],
        "documents": sum(pq.ParquetFile(p).metadata.num_rows for p in paths),
        "tokens_total": sum(counts),
        "zero_count_entries": sum(1 for c in counts if c == 0),
        "note": ("counts[d] is the corpus count of script_bpe id ids[d], which the model sees as "
                 "embedding row d; BOS is row vocab_size and is not counted"),
        "counts": counts,
        "surface": forms,
    }
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump(record, fh)
    print(f"{out}: {record['tokens_total']:,} tokens over {record['documents']:,} documents, "
          f"{record['zero_count_entries']} entry(s) with count 0")


@app.command
def pair(plain: Path, marked: Path, out: Path):
    """Line up each delimited entry of a marker vocabulary with the baseline's two forms.

    `<|>word<|>` against `word` and `' word'`, which is the comparison the embedding question
    asks for. Entries of either vocabulary with no counterpart are written too, with an empty
    count, so nothing is dropped silently.
    """
    def read(path):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    p, m = read(plain), read(marked)
    if p["language"] != m["language"]:
        raise SystemExit(f"{plain} is {p['language']}, {marked} is {m['language']}")
    plain_counts = {form: (d, p["ids"][d], c) for d, (form, c) in enumerate(zip(p["surface"], p["counts"]))}

    rows = [(
        "marked_row", "marked_id", "marked_form", "marked_count",
        "bare_row", "bare_id", "bare_count", "spaced_row", "spaced_id", "spaced_count",
    )]
    for i, (form, c) in enumerate(zip(m["surface"], m["counts"])):
        if not (form.startswith("<|>") and form.endswith("<|>")):
            continue
        word = form[3:-3]
        bare = plain_counts.get(word)
        spaced = plain_counts.get(" " + word)
        rows.append((
            i, m["ids"][i], form, c,
            *(bare if bare else ("", "", "")),
            *(spaced if spaced else ("", "", "")),
        ))
    with open(out, "w") as fh:
        for row in rows:
            fh.write("\t".join(str(x) for x in row) + "\n")
    print(f"{out}: {len(rows) - 1} delimited entry(s) of {marked.name} against {plain.name}")


if __name__ == "__main__":
    app()
