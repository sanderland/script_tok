import json
import os
import queue
from collections import Counter
from typing import Iterable, Iterator, Literal

import polars as pl

from script_bpe.pretokenize import Pretokenizer
from script_bpe.utils import PROJECT_ROOT, shutdown_pool, token_array, TokenSeq


# Upper bound on the characters of text sent to a worker in one task (a single longer document is its own task).
# Large enough that task pickling and result merging are negligible next to encoding, small enough that the
# idle tail at the end is short and a single huge batch (the `from_texts` path) still spreads over all workers.
ENCODE_TASK_MAX_CHARS = 4_000_000
# Tasks allowed in flight per worker. 2 keeps every worker fed while the main process reads the next batch;
# the texts of all in-flight tasks are held in memory (at most ~2 * num_workers * ENCODE_TASK_MAX_CHARS chars).
MAX_INFLIGHT_TASKS_PER_WORKER = 2

_worker_pretokenizer: Pretokenizer | None = None


def _init_encode_worker(pretokenizer: Pretokenizer) -> None:
    """Pool initializer: receive the pretokenizer once per worker instead of pickling it with every task."""
    global _worker_pretokenizer
    _worker_pretokenizer = pretokenizer


def _encode_texts_in_worker(texts: list[str], max_length: int) -> tuple[Counter[bytes], dict]:
    if _worker_pretokenizer is None:
        raise RuntimeError("Pool worker has no pretokenizer: _init_encode_worker was not run as the pool initializer")
    return PretokenizedCorpus.encode_texts(texts, _worker_pretokenizer, max_length)


def _split_by_chars(texts: list[str], max_chars: int) -> Iterator[list[str]]:
    """Yield consecutive slices of `texts` of at most `max_chars` characters (a longer document is its own slice)."""
    task: list[str] = []
    task_chars = 0
    for text in texts:
        if task and task_chars + len(text) > max_chars:
            yield task
            task, task_chars = [], 0
        task.append(text)
        task_chars += len(text)
    if task:
        yield task


class PretokenizedCorpus:
    VERSION = "v2"
    DEFAULT_MAX_LENGTH = 10_000_000  # max chunk length
    DEFAULT_PARTITIONS = 128  # number of partitions to split the corpus into
    DEFAULT_BASE_PATH = os.path.join(PROJECT_ROOT, "results/corpora")  # default path to save the corpus
    PARQUET_COMPRESSION: Literal["lz4", "uncompressed", "snappy", "gzip", "lzo", "brotli", "zstd"] = "lz4"

    def __init__(
        self,
        name: str,
        base_path: str,
        pretokenizer: Pretokenizer,
        dummy: bool = False,
    ):
        self.name = name
        self.base_path = base_path
        self.pretokenizer = pretokenizer
        if not dummy:
            with open(self.metadata_path(), "r") as f:
                self.metadata = json.load(f)
                assert pretokenizer.hash() == self.metadata["pretokenizer_hash"], "Pretokenizer hash mismatch"
            self.partitions = sorted([f for f in os.listdir(self.dir_path()) if f.endswith(".parquet")])

    def dir_path(self) -> str:
        dir_path = os.path.join(self.base_path, self.name, self.pretokenizer.hash())
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def partition_path(self, partition: int) -> str:
        return os.path.join(self.dir_path(), f"part_{partition:04d}.parquet")

    def metadata_path(self) -> str:
        return os.path.join(self.dir_path(), "metadata.json")

    @staticmethod
    def encode_texts(texts: list[str], pretokenizer: Pretokenizer, max_length: int) -> tuple[Counter[bytes], dict]:
        metadata: dict[str, int] = dict(atomic_tokens=0, chunks=0, chunks_skipped=0)
        chunk_counts: Counter[bytes] = Counter()
        for text in texts:
            for chunk in pretokenizer.pretokenize(text):
                if len(chunk) > max_length:
                    metadata["chunks_skipped"] += 1
                    continue
                chunk_counts[chunk.tobytes()] += 1
                metadata["atomic_tokens"] += len(chunk)
                metadata["chunks"] += 1
        return chunk_counts, metadata

    @classmethod
    def from_text_batches(
        cls,
        name: str,
        text_batches: Iterable[list[str]],
        pretokenizer: Pretokenizer,
        base_path=DEFAULT_BASE_PATH,
        num_partitions=DEFAULT_PARTITIONS,
        max_length=DEFAULT_MAX_LENGTH,
        num_workers: int = 1,
    ):
        corpus = cls(name, base_path, pretokenizer, dummy=True)  # for path
        if os.path.exists(corpus.metadata_path()):
            raise FileExistsError(
                f"Corpus {name} already exists at {corpus.metadata_path()}. Use a different name or delete the existing corpus."
            )

        if num_workers > 1:
            from script_bpe.utils import mp_ctx

            pool = mp_ctx.Pool(num_workers, initializer=_init_encode_worker, initargs=(pretokenizer,))
        else:
            import multiprocessing.dummy

            pool = multiprocessing.dummy.Pool(1, initializer=_init_encode_worker, initargs=(pretokenizer,))
        max_in_flight = MAX_INFLIGHT_TASKS_PER_WORKER * max(num_workers, 1)
        finished: queue.SimpleQueue = queue.SimpleQueue()  # results or exceptions, in completion order

        total_chunk_counts: Counter[bytes] = Counter()
        metadata: dict[str, int | str] = dict(
            version=cls.VERSION,
            max_length=max_length,
            pretokenizer_hash=pretokenizer.hash(),
            docs=0,
            atomic_tokens=0,
            chunks=0,
            chunks_skipped=0,
        )

        def merge_one_result() -> None:
            outcome = finished.get()
            if isinstance(outcome, BaseException):
                raise outcome
            part_chunk_counts, part_metadata = outcome
            # update(), not `+=`: Counter.__iadd__ rescans the whole accumulated counter after
            # every merge (_keep_positive), making the phase quadratic in the number of merges.
            # Counts here are always positive, so skipping the scan changes only the cost.
            total_chunk_counts.update(part_chunk_counts)
            for k, v in part_metadata.items():
                metadata[k] += v

        # Not `with pool:`. That exits via Pool.terminate(), whose join can block forever
        # when the forkserver leaves an exited worker unreaped, and it does so *after* the
        # loop below has already collected every result, so the corpus is lost at the last
        # step. shutdown_pool bounds the wait and escalates. See WORKER_JOIN_TIMEOUT_S.
        try:
            in_flight = 0
            for texts in text_batches:
                if not texts:
                    continue
                metadata["docs"] += len(texts)
                for task_texts in _split_by_chars(texts, ENCODE_TASK_MAX_CHARS):
                    while in_flight >= max_in_flight:
                        merge_one_result()
                        in_flight -= 1
                    pool.apply_async(
                        _encode_texts_in_worker,
                        (task_texts, max_length),
                        callback=finished.put,
                        error_callback=finished.put,
                    )
                    in_flight += 1
            while in_flight:
                merge_one_result()
                in_flight -= 1
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        shutdown_pool(pool)

        flattened_data: list[dict[str, int | bytes]] = [
            dict(
                chunk=chunk,
                count=total_chunk_counts[chunk],
            )
            for chunk in sorted(total_chunk_counts)
        ]
        metadata["unique_chunks"] = len(flattened_data)
        with open(corpus.metadata_path(), "w") as f:
            json.dump(metadata, f, indent=4)
        for p in range(num_partitions):  # save each partition to a separate parquet file
            pl.DataFrame(flattened_data[p::num_partitions]).write_parquet(
                corpus.partition_path(p), compression=cls.PARQUET_COMPRESSION
            )
        return cls(name, base_path, pretokenizer)

    @classmethod
    def from_texts(
        cls,
        name: str,
        texts: list[str],
        pretokenizer: Pretokenizer,
        base_path=DEFAULT_BASE_PATH,
        num_partitions=DEFAULT_PARTITIONS,
        max_length=DEFAULT_MAX_LENGTH,
        num_workers: int = 1,
    ):
        return cls.from_text_batches(
            name=name,
            text_batches=[texts],
            pretokenizer=pretokenizer,
            base_path=base_path,
            num_partitions=num_partitions,
            max_length=max_length,
            num_workers=num_workers,
        )

    def worker_iterate(self, worker_id: int, num_workers: int) -> Iterable[tuple[TokenSeq, int]]:
        for i, partition_file in enumerate(self.partitions):
            if i % num_workers == worker_id:  # could be smarter if not evenly divisible
                partition_path = os.path.join(self.dir_path(), partition_file)
                for bchunk, count in pl.read_parquet(partition_path).iter_rows():
                    chunk = token_array([])
                    chunk.frombytes(bchunk)
                    yield chunk, count

    def __iter__(self):  # single process iterate
        for chunk, count in self.worker_iterate(0, 1):
            yield chunk, count
