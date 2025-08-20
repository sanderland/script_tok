import json
import os
from collections import Counter
from typing import Iterable, Literal

import polars as pl

from script_bpe.pretokenize import Pretokenizer
from script_bpe.utils import PROJECT_ROOT, token_array, TokenSeq


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
        corpus = cls(name, base_path, pretokenizer, dummy=True)  # for path
        if os.path.exists(corpus.metadata_path()):
            raise FileExistsError(
                f"Corpus {name} already exists at {corpus.metadata_path()}. Use a different name or delete the existing corpus."
            )

        if num_workers > 1:
            from script_bpe.utils import mp_ctx

            pool = mp_ctx.Pool(num_workers)
        else:
            import multiprocessing.dummy

            pool = multiprocessing.dummy.Pool(1)

        total_chunk_counts: Counter[bytes] = Counter()
        metadata: dict[str, int | str] = dict(
            version=cls.VERSION,
            max_length=max_length,
            pretokenizer_hash=pretokenizer.hash(),
            docs=len(texts),
            atomic_tokens=0,
            chunks=0,
            chunks_skipped=0,
        )
        with pool:
            results = [
                pool.apply_async(cls.encode_texts, (texts[i::num_workers], pretokenizer, max_length))
                for i in range(num_workers)
            ]
            for result in results:
                part_chunk_counts, part_metadata = result.get()
                total_chunk_counts += part_chunk_counts
                for k, v in part_metadata.items():
                    metadata[k] += v

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
