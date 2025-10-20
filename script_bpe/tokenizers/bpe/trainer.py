from __future__ import annotations

import heapq
import itertools
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass, field

from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer, MergeRule, Token
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.utils import TokenSeq, token_array


class BPETrainerConfig(TrainerConfig):
    pass


class ChunkTokenization:
    COUNTER = itertools.count()
    __slots__ = ("curr_seq", "count", "id")

    def __init__(self, curr_seq: TokenSeq, count: int = 0):
        self.curr_seq = curr_seq  #  will mutate this!
        self.count = count
        self.id = next(self.COUNTER)

    def merge(self, from_ids: tuple[int, int], to_id: int):
        ids = self.curr_seq
        if len(ids) < 2:
            return {}, 0
        num_merges = 0
        delta_counts: dict[tuple[int, int], int] = defaultdict(int)
        from_i = 0
        to_i = -1  # last
        len_ids = len(ids)
        while from_i < len_ids - 1:
            if ids[from_i] == from_ids[0] and ids[from_i + 1] == from_ids[1]:
                if to_i >= 0:
                    delta_counts[(ids[to_i], ids[from_i])] -= 1
                    delta_counts[(ids[to_i], to_id)] += 1
                to_i += 1
                ids[to_i] = to_id
                num_merges += 1
                from_i += 2
                if from_i < len_ids:
                    delta_counts[(ids[from_i - 1], ids[from_i])] -= 1
                    delta_counts[(to_id, ids[from_i])] += 1
            else:
                to_i += 1
                ids[to_i] = ids[from_i]
                from_i += 1
        if from_i < len_ids:
            to_i += 1
            ids[to_i] = ids[from_i]
        del ids[to_i + 1 :]
        delta_counts[from_ids] -= num_merges
        return delta_counts, num_merges


@dataclass(slots=True)
class TokenPairCounts:
    chunk_id_to_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    count: int = 0

    def calculate_count(self, chunks: dict[int, ChunkTokenization]):
        self.count = sum(
            chunks[chunk_id].count * count_per_chunk for chunk_id, count_per_chunk in self.chunk_id_to_count.items()
        )
        return self.count


def worker_thread(
    worker_id: int,
    num_workers: int,
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    shared_chunks: dict[int, ChunkTokenization],
    shared_tokens: dict[int, Token],
    cmd_queue: queue.Queue,
    results_queue: queue.Queue,
    chunks_lock: threading.Lock,
    tokens_lock: threading.Lock,
):
    """
    Worker thread for BPE training with shared memory.
    Each worker owns a partition of chunks (determined by worker_iterate).
    Workers read shared_tokens (no lock needed for reads in free-threaded Python).
    """
    # Thread-local storage for this worker's data
    local_chunks = {}
    local_pair_counts: dict[tuple[int, int], TokenPairCounts] = defaultdict(TokenPairCounts)
    local_tokens = {k: Token(id=k, atomic_tokens=token_array([k])) for k in pretokenizer.atomic_tokens}
    
    # Load this worker's partition of the corpus
    for atomic_token_seq, count in corpus.worker_iterate(worker_id, num_workers):
        # Create a copy of the array for this worker
        chunk_seq = token_array(atomic_token_seq)
        chunk_info = ChunkTokenization(chunk_seq, count=count)
        local_chunks[chunk_info.id] = chunk_info
        for i in range(len(atomic_token_seq) - 1):
            pair = (atomic_token_seq[i], atomic_token_seq[i + 1])
            if pretokenizer.bpe_merge_allowed([pair[0]], [pair[1]]):
                local_pair_counts[pair].chunk_id_to_count[chunk_info.id] += 1
            local_tokens[atomic_token_seq[i]].original_count += count
        local_tokens[atomic_token_seq[-1]].original_count += count
    
    # Register chunks in shared memory
    with chunks_lock:
        shared_chunks.update(local_chunks)
    
    # Set current counts
    for t in local_tokens.values():
        t.current_count = t.original_count
    
    # Send initial pair counts
    results_queue.put(
        ("INITIAL_COUNTS", {pair: pc.calculate_count(local_chunks) for pair, pc in local_pair_counts.items()})
    )
    
    # Main merge loop
    while True:
        command = cmd_queue.get()
        if command == "GET_FINAL_STATE":
            results_queue.put(("FINAL_STATE", local_tokens))
            break
        
        merge_pair, new_token_id = command
        
        # Create new token in local token dict
        local_tokens[new_token_id] = Token(
            id=new_token_id, 
            atomic_tokens=local_tokens[merge_pair[0]].atomic_tokens + local_tokens[merge_pair[1]].atomic_tokens
        )
        
        local_delta_counts: dict[tuple[int, int], int] = defaultdict(int)
        local_merge_count = 0
        chunks_affected = list(local_pair_counts[merge_pair].chunk_id_to_count.keys())
        
        # Perform merges on this worker's chunks
        for chunk_id in chunks_affected:
            chunk = local_chunks[chunk_id]
            chunk_count = chunk.count
            delta_counts, chunk_merges = chunk.merge(from_ids=merge_pair, to_id=new_token_id)
            local_merge_count += chunk_merges * chunk_count
            
            for pair, dc in delta_counts.items():
                if dc == 0:
                    continue
                delta_count = dc * chunk_count
                pair_count = local_pair_counts.get(pair)
                if pair_count is None:
                    if not pretokenizer.bpe_merge_allowed(
                        local_tokens[pair[0]].atomic_tokens, 
                        local_tokens[pair[1]].atomic_tokens
                    ):
                        continue
                    local_pair_counts[pair] = pair_count = TokenPairCounts()
                local_delta_counts[pair] += delta_count
                pair_count.chunk_id_to_count[chunk_id] += dc
                if pair_count.chunk_id_to_count[chunk_id] == 0:
                    del pair_count.chunk_id_to_count[chunk_id]
        
        # Update local pair counts
        for pair, dc in local_delta_counts.items():
            local_pair_counts[pair].count += dc
        
        # Send deltas back to main thread
        results_queue.put(("DELTA_COUNTS", local_delta_counts, len(chunks_affected)))
        
        # Update local token counts
        for ti in merge_pair:
            local_tokens[ti].current_count -= local_merge_count
        local_tokens[new_token_id].current_count = local_tokens[new_token_id].original_count = local_merge_count


class BPETrainer(BaseTrainer):
    def __init__(self, pretokenizer: Pretokenizer, corpus: PretokenizedCorpus, config: BPETrainerConfig):
        super().__init__(pretokenizer, corpus, config)

    @staticmethod
    def init_tokens(pretokenizer: Pretokenizer):
        return {k: Token(id=k, atomic_tokens=token_array([k])) for k in pretokenizer.atomic_tokens}

    def train(self) -> BPETokenizer:
        cfg = self.config
        pretokenizer = self.pretokenizer
        corpus = self.corpus
        logger = self.logger

        logger.info(f"Training with {cfg.num_workers} workers (threading) on corpus {corpus.name}: {corpus.metadata}")

        # Shared data structures
        shared_chunks: dict[int, ChunkTokenization] = {}
        shared_tokens: dict[int, Token] = self.init_tokens(pretokenizer)
        chunks_lock = threading.Lock()
        tokens_lock = threading.Lock()
        
        # Communication queues
        cmd_queues = [queue.Queue() for _ in range(cfg.num_workers)]
        results_queue = queue.Queue()
        
        # Start worker threads
        workers = []
        for worker_id in range(cfg.num_workers):
            t = threading.Thread(
                target=worker_thread,
                args=(
                    worker_id, 
                    cfg.num_workers, 
                    pretokenizer, 
                    corpus, 
                    shared_chunks,
                    shared_tokens,
                    cmd_queues[worker_id], 
                    results_queue,
                    chunks_lock,
                    tokens_lock,
                ),
                daemon=True,
            )
            workers.append(t)
            t.start()

        # Collect initial pair counts from all workers
        overall_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        logger.info("Started workers, collecting initial pair counts")
        for _ in range(cfg.num_workers):
            msg_type, data = results_queue.get()
            assert msg_type == "INITIAL_COUNTS"
            for pair, count in data.items():
                overall_pair_counts[pair] += count

        tokens = self.init_tokens(pretokenizer)
        next_token_id = max(tokens.keys()) + 1
        merge_rules: list[MergeRule] = []

        pair_counts_heap = [(-count, p) for p, count in overall_pair_counts.items()]
        heapq.heapify(pair_counts_heap)
        logger.info(f"Initialized with {len(pair_counts_heap):,d} token pairs")

        # Main merge loop
        while len(merge_rules) < cfg.additional_vocab_size:
            if not pair_counts_heap:
                logger.warning("Heap empty - no more merges possible - overly small corpus?")
                break
            neg_max_count, most_common_pair = heapq.heappop(pair_counts_heap)
            if most_common_pair not in overall_pair_counts:
                continue
            if overall_pair_counts[most_common_pair] != -neg_max_count:
                heapq.heappush(pair_counts_heap, (-overall_pair_counts[most_common_pair], most_common_pair))
                continue

            (ta, tb) = most_common_pair
            tokens[next_token_id] = Token(
                id=next_token_id, atomic_tokens=tokens[ta].atomic_tokens + tokens[tb].atomic_tokens
            )
            merge_rules.append(MergeRule(tokens_from=(ta, tb), token_to=next_token_id))
            
            # Send merge command to all workers
            for q in cmd_queues:
                q.put((most_common_pair, next_token_id))

            # Collect deltas from all workers
            aggregated_deltas: dict[tuple[int, int], int] = defaultdict(int)
            n_chunks_affected = {}
            for worker_id in range(cfg.num_workers):
                msg_type, worker_deltas, n_chunks_affected[worker_id] = results_queue.get()
                assert msg_type == "DELTA_COUNTS"
                for pair, delta_count in worker_deltas.items():
                    aggregated_deltas[pair] += delta_count

            # Update overall pair counts
            for pair, delta_count in aggregated_deltas.items():
                if delta_count == 0:
                    continue
                overall_pair_counts[pair] += delta_count
                if overall_pair_counts[pair] == 0:
                    del overall_pair_counts[pair]
                elif delta_count > 0:
                    heapq.heappush(pair_counts_heap, (-overall_pair_counts[pair], pair))

            logger.info(
                f"Max count {-neg_max_count:10,d} from {len(overall_pair_counts):8,d} candidate pairs -> Created {tokens[next_token_id].pretty_repr(pretokenizer):<40} from "
                f"merging {tokens[ta].pretty_repr(pretokenizer):<30} & {tokens[tb].pretty_repr(pretokenizer):<30} \tTotal {sum(n_chunks_affected.values()):10,d} chunks affected ({min(n_chunks_affected.values()):5,d} - {max(n_chunks_affected.values()):5,d}/worker). Heap size {len(pair_counts_heap):,d}"
            )
            next_token_id += 1

        # Request final state from workers and aggregate token counts
        logger.debug("Calculating final token counts and terminating workers")
        for q in cmd_queues:
            q.put("GET_FINAL_STATE")
        for _ in range(cfg.num_workers):
            msg_type, worker_tokens = results_queue.get()
            assert msg_type == "FINAL_STATE"
            for token_id, token in worker_tokens.items():
                tokens[token_id].original_count += token.original_count
                tokens[token_id].current_count += token.current_count

        # Wait for all workers to finish
        for t in workers:
            t.join()

        logger.info(f"Done! {len(merge_rules)} merge rules created, tokenizer has {len(tokens)} total tokens")
        return BPETokenizer(
            pretokenizer=pretokenizer,
            merge_rules=merge_rules,
            metadata=dict(
                settings=dict(num_workers=cfg.num_workers, threading=True),
                corpus=corpus.metadata,
            ),
            tokens=tokens,
        )
