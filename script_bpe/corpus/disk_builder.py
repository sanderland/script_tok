"""Exact corpus counts with bounded Python memory and restartable SQLite staging."""

import json
from pathlib import Path
import sqlite3
import time

import pyarrow as pa
import pyarrow.parquet as pq

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.utils import mp_ctx, shutdown_pool

_PRETOKENIZER = None


def _init_worker(pretokenizer):
    global _PRETOKENIZER
    _PRETOKENIZER = pretokenizer


def _encode(texts):
    return PretokenizedCorpus.encode_texts(texts, _PRETOKENIZER, PretokenizedCorpus.DEFAULT_MAX_LENGTH)


def build_disk_corpus(name, batches, pretokenizer, base_dir, workers=2):
    corpus = PretokenizedCorpus(name, base_dir, pretokenizer, dummy=True)
    directory = Path(corpus.dir_path())
    metadata_path = Path(corpus.metadata_path())
    if metadata_path.exists():
        return PretokenizedCorpus(name, base_dir, pretokenizer)
    database = directory / "building.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-262144")  # 256 MiB, bounded even for large corpora.
    connection.execute("CREATE TABLE IF NOT EXISTS counts (chunk BLOB PRIMARY KEY, count INTEGER NOT NULL) WITHOUT ROWID")
    connection.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
    row = connection.execute("SELECT data FROM state WHERE id=1").fetchone()
    state = json.loads(row[0]) if row else {"batches": 0, "characters": 0, "version": corpus.VERSION,
        "max_length": corpus.DEFAULT_MAX_LENGTH, "pretokenizer_hash": pretokenizer.hash(),
        "docs": 0, "atomic_tokens": 0, "chunks": 0, "chunks_skipped": 0}
    started = time.monotonic()
    last_report = started
    pool = mp_ctx.Pool(workers, initializer=_init_worker, initargs=(pretokenizer,)) if workers > 1 else None
    _init_worker(pretokenizer)
    try:
        for index, texts in enumerate(batches):
            if index < state["batches"]:
                continue
            pieces = [texts[i::workers] for i in range(min(workers, len(texts)))]
            results = pool.map(_encode, pieces) if pool else [_encode(texts)]
            with connection:
                for counts, metadata in results:
                    connection.executemany("INSERT INTO counts VALUES (?, ?) ON CONFLICT(chunk) DO UPDATE SET count=count+excluded.count", sorted(counts.items()))
                    for key, value in metadata.items():
                        state[key] += value
                state["docs"] += len(texts)
                state["characters"] += sum(map(len, texts))
                state["batches"] = index + 1
                connection.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (json.dumps(state),))
            del results
            if time.monotonic() - last_report >= 30:
                print(f"Corpus {name}: {state['characters']:,} characters, {state['docs']:,} documents; counts checkpointed", flush=True)
                last_report = time.monotonic()
        if pool:
            shutdown_pool(pool)
            pool = None
        state["unique_chunks"] = connection.execute("SELECT COUNT(*) FROM counts").fetchone()[0]
        # Match the ordinary builder's sorted, round-robin partition assignment.
        schema = pa.schema([("chunk", pa.binary()), ("count", pa.int64())])
        writers = []
        try:
            for partition in range(corpus.DEFAULT_PARTITIONS):
                writers.append(pq.ParquetWriter(corpus.partition_path(partition), schema, compression="lz4"))
            cursor = connection.execute("SELECT chunk, count FROM counts ORDER BY chunk")
            offset = 0
            while rows := cursor.fetchmany(131072):
                for partition, writer in enumerate(writers):
                    selected = rows[(partition - offset) % len(writers)::len(writers)]
                    if selected:
                        writer.write_table(pa.table({"chunk": [r[0] for r in selected], "count": [r[1] for r in selected]}, schema=schema))
                offset += len(rows)
        finally:
            for writer in writers:
                writer.close()
        # Publish metadata last so an interrupted export is never a valid corpus.
        partial = metadata_path.with_suffix(".json.partial")
        partial.write_text(json.dumps(state, indent=2))
        partial.replace(metadata_path)
    finally:
        if pool:
            pool.terminate()
            pool.join()
        connection.close()
    database.unlink()  # Only this builder's disposable, now-exported counts database.
    print(f"Corpus complete: {state['characters']:,} characters, {state['unique_chunks']:,} unique chunks", flush=True)
    return PretokenizedCorpus(name, base_dir, pretokenizer)
