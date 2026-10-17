from __future__ import annotations

import heapq
import itertools
import math
import multiprocessing
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from pydantic import model_validator

from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer, MergeRule, Token
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.utils import TokenSeq, join_workers, mp_ctx, token_array


class BPETrainerConfig(TrainerConfig):
    # "unibpe" ranks merges by the drop in Unigram corpus loss instead of by pair count (Aleph Alpha, Kolibri
    # tech report, Sec. 2.1.2). The tokenizer is still an ordinary merge list.
    merge_score: Literal["count", "unibpe"] = "count"
    # Number of leading merges scored by UniBPE before falling back to plain pair-count BPE. None means all.
    # The report uses 119,744 UniBPE merges followed by 7,900 BPE merges, because pure UniBPE compresses worse.
    unibpe_merges: int | None = None

    @model_validator(mode="after")
    def _unibpe_merges_needs_unibpe(self):
        if self.unibpe_merges is not None and self.merge_score != "unibpe":
            raise ValueError("unibpe_merges requires merge_score='unibpe'")
        return self


def _h(x: int | float) -> float:
    return x * math.log(x) if x > 0 else 0.0


class UniBPEScorer:
    """Ranks candidate merges by the change in Unigram negative log-likelihood (the most negative first).

    With token counts c_t and N = sum c_t, the loss is h(N) - sum_t h(c_t) for h(x) = x log x. Merging (a, b)
    c_ab times changes it by
        h(c_a) - h(c_a - c_ab) + h(c_b) - h(c_b - c_ab) - h(c_ab) + h(N - c_ab) - h(N).
    Scores are kept in a lazily re-validated heap: a popped entry is rescored and pushed back if stale. This is
    exact because a falling N or a fallen pair count can only make a stale score look better than it is, and the
    pairs whose parent counts fell (those touching a or b) are rescored eagerly after every merge.
    """

    def __init__(self, pair_counts: dict[tuple[int, int], int], token_counts: dict[int, int]):
        self.pair_counts = pair_counts
        self.token_counts = token_counts
        self.n = sum(token_counts.values())
        self.pairs_by_token: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for pair in pair_counts:
            self._index(pair)
        self.heap = [(self.delta_loss(pair), pair) for pair in pair_counts]
        heapq.heapify(self.heap)

    def _index(self, pair: tuple[int, int]):
        self.pairs_by_token[pair[0]].add(pair)
        self.pairs_by_token[pair[1]].add(pair)

    def delta_loss(self, pair: tuple[int, int]) -> float:
        a, b = pair
        ca = self.token_counts[a]
        if a == b:
            # Overlapping runs make the pair count exceed the merge count, which is at most c_a // 2.
            c = min(self.pair_counts[pair], ca // 2)
            return _h(ca) - _h(ca - 2 * c) - _h(c) + _h(self.n - c) - _h(self.n)
        c = self.pair_counts[pair]
        cb = self.token_counts[b]
        return _h(ca) - _h(ca - c) + _h(cb) - _h(cb - c) - _h(c) + _h(self.n - c) - _h(self.n)

    def pop(self) -> tuple[int, int] | None:
        while self.heap:
            score, pair = heapq.heappop(self.heap)
            if pair not in self.pair_counts:
                continue
            fresh = self.delta_loss(pair)
            if fresh != score:
                heapq.heappush(self.heap, (fresh, pair))
                continue
            return pair
        return None

    def apply_merge(
        self,
        pair: tuple[int, int],
        new_token_id: int,
        num_merges: int,
        pair_deltas: dict[tuple[int, int], int],
    ):
        """Update token counts and heap after the merge; `self.pair_counts` must already include `pair_deltas`."""
        for t in pair:
            self.token_counts[t] -= num_merges
        self.token_counts[new_token_id] = num_merges
        self.n -= num_merges
        rescore = set(self.pairs_by_token[pair[0]]) | set(self.pairs_by_token[pair[1]])
        for p, dc in pair_deltas.items():
            if p in self.pair_counts:
                self._index(p)
                if dc > 0:
                    rescore.add(p)
            else:
                self.pairs_by_token[p[0]].discard(p)
                self.pairs_by_token[p[1]].discard(p)
        for p in rescore:
            if p in self.pair_counts:
                heapq.heappush(self.heap, (self.delta_loss(p), p))


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


def worker_process(
    worker_id: int,
    num_workers: int,
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    cmd_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
):
    tokens = BPETrainer.init_tokens(pretokenizer)

    local_chunks = {}
    local_pair_counts: dict[tuple[int, int], TokenPairCounts] = defaultdict(TokenPairCounts)
    for atomic_token_seq, count in corpus.worker_iterate(worker_id, num_workers):
        chunk_info = ChunkTokenization(atomic_token_seq, count=count)
        local_chunks[chunk_info.id] = chunk_info
        for i in range(len(atomic_token_seq) - 1):
            pair = (atomic_token_seq[i], atomic_token_seq[i + 1])
            if pretokenizer.bpe_merge_allowed([pair[0]], [pair[1]]):
                local_pair_counts[pair].chunk_id_to_count[chunk_info.id] += 1
            tokens[atomic_token_seq[i]].original_count += count
        tokens[atomic_token_seq[-1]].original_count += count
    del corpus

    for t in tokens.values():
        t.current_count = t.original_count
    results_queue.put(
        (
            "INITIAL_COUNTS",
            {pair: pc.calculate_count(local_chunks) for pair, pc in local_pair_counts.items()},
            {t.id: t.original_count for t in tokens.values()},
        )
    )

    while True:
        command = cmd_queue.get()
        if command == "GET_FINAL_STATE":
            results_queue.put(("FINAL_STATE", tokens))
            break
        merge_pair, new_token_id = command
        tokens[new_token_id] = Token(
            id=new_token_id, atomic_tokens=tokens[merge_pair[0]].atomic_tokens + tokens[merge_pair[1]].atomic_tokens
        )
        local_delta_counts: dict[tuple[int, int], int] = defaultdict(int)
        local_merge_count = 0
        chunks_affected = list(local_pair_counts[merge_pair].chunk_id_to_count.keys())
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
                    if not pretokenizer.bpe_merge_allowed(tokens[pair[0]].atomic_tokens, tokens[pair[1]].atomic_tokens):
                        continue
                    local_pair_counts[pair] = pair_count = TokenPairCounts()
                local_delta_counts[pair] += delta_count
                pair_count.chunk_id_to_count[chunk_id] += dc
                if pair_count.chunk_id_to_count[chunk_id] == 0:
                    del pair_count.chunk_id_to_count[chunk_id]
        for pair, dc in local_delta_counts.items():
            local_pair_counts[pair].count += dc
        results_queue.put(("DELTA_COUNTS", local_delta_counts, len(chunks_affected), local_merge_count))
        for ti in merge_pair:
            tokens[ti].current_count -= local_merge_count
        tokens[new_token_id].current_count = tokens[new_token_id].original_count = local_merge_count


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

        logger.info(f"Training with {cfg.num_workers} workers on corpus {corpus.name}: {corpus.metadata}")

        cmd_queues = [mp_ctx.Queue() for _ in range(cfg.num_workers)]
        results_queue = mp_ctx.Queue()
        workers = []
        for worker_id in range(cfg.num_workers):
            p = mp_ctx.Process(
                target=worker_process,
                args=(worker_id, cfg.num_workers, pretokenizer, corpus, cmd_queues[worker_id], results_queue),
                daemon=True,
            )
            workers.append(p)
            p.start()

        overall_pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        overall_token_counts: dict[int, int] = defaultdict(int)
        logger.info("Started workers, collecting initial pair counts")
        for _ in range(cfg.num_workers):
            msg_type, data, worker_token_counts = results_queue.get()
            assert msg_type == "INITIAL_COUNTS"
            for pair, count in data.items():
                overall_pair_counts[pair] += count
            for token_id, count in worker_token_counts.items():
                overall_token_counts[token_id] += count

        tokens = self.init_tokens(pretokenizer)
        next_token_id = max(tokens.keys()) + 1
        merge_rules: list[MergeRule] = []

        unibpe_budget = cfg.additional_vocab_size if cfg.unibpe_merges is None else cfg.unibpe_merges
        unibpe = (
            UniBPEScorer(overall_pair_counts, overall_token_counts)
            if cfg.merge_score == "unibpe" and unibpe_budget > 0
            else None
        )
        pair_counts_heap: list[tuple[int, tuple[int, int]]] = []
        if unibpe is None:
            pair_counts_heap = [(-count, p) for p, count in overall_pair_counts.items()]
            heapq.heapify(pair_counts_heap)
        logger.info(f"Initialized with {len(overall_pair_counts):,d} token pairs")

        while len(merge_rules) < cfg.additional_vocab_size:
            if unibpe is not None and len(merge_rules) >= unibpe_budget:
                logger.info(f"Switching to count-ranked BPE after {len(merge_rules):,d} UniBPE merges")
                unibpe = None
                pair_counts_heap = [(-count, p) for p, count in overall_pair_counts.items()]
                heapq.heapify(pair_counts_heap)
            if unibpe is not None:
                most_common_pair = unibpe.pop()
                if most_common_pair is None:
                    logger.warning("Heap empty - no more merges possible - overly small corpus?")
                    break
            else:
                if not pair_counts_heap:
                    logger.warning("Heap empty - no more merges possible - overly small corpus?")
                    break
                neg_max_count, most_common_pair = heapq.heappop(pair_counts_heap)
                if most_common_pair not in overall_pair_counts:
                    continue
                if overall_pair_counts[most_common_pair] != -neg_max_count:
                    heapq.heappush(pair_counts_heap, (-overall_pair_counts[most_common_pair], most_common_pair))
                    continue
            max_count = overall_pair_counts[most_common_pair]

            (ta, tb) = most_common_pair
            tokens[next_token_id] = Token(
                id=next_token_id, atomic_tokens=tokens[ta].atomic_tokens + tokens[tb].atomic_tokens
            )
            merge_rules.append(MergeRule(tokens_from=(ta, tb), token_to=next_token_id))
            for q in cmd_queues:
                q.put((most_common_pair, next_token_id))

            aggregated_deltas: dict[tuple[int, int], int] = defaultdict(int)
            n_chunks_affected = {}
            num_merges = 0
            for worker_id in range(cfg.num_workers):
                msg_type, worker_deltas, n_chunks_affected[worker_id], worker_merges = results_queue.get()
                assert msg_type == "DELTA_COUNTS"
                num_merges += worker_merges
                for pair, delta_count in worker_deltas.items():
                    aggregated_deltas[pair] += delta_count

            for pair, delta_count in aggregated_deltas.items():
                if delta_count == 0:
                    continue
                overall_pair_counts[pair] += delta_count
                if overall_pair_counts[pair] == 0:
                    del overall_pair_counts[pair]
                elif delta_count > 0 and unibpe is None:
                    heapq.heappush(pair_counts_heap, (-overall_pair_counts[pair], pair))
            if unibpe is not None:
                unibpe.apply_merge(most_common_pair, next_token_id, num_merges, aggregated_deltas)

            logger.info(
                f"Max count {max_count:10,d} from {len(overall_pair_counts):8,d} candidate pairs -> Created {tokens[next_token_id].pretty_repr(pretokenizer):<40} from "
                f"merging {tokens[ta].pretty_repr(pretokenizer):<30} & {tokens[tb].pretty_repr(pretokenizer):<30} \tTotal {sum(n_chunks_affected.values()):10,d} chunks affected ({min(n_chunks_affected.values()):5,d} - {max(n_chunks_affected.values()):5,d}/worker). Heap size {len(unibpe.heap if unibpe is not None else pair_counts_heap):,d}"
            )
            next_token_id += 1

        logger.debug("Calculating final token counts and terminating workers")
        for q in cmd_queues:
            q.put("GET_FINAL_STATE")
        for _ in range(cfg.num_workers):
            msg_type, worker_tokens = results_queue.get()
            assert msg_type == "FINAL_STATE"
            for token_id, token in worker_tokens.items():
                tokens[token_id].original_count += token.original_count
                tokens[token_id].current_count += token.current_count

        # Every worker's FINAL_STATE is already folded into `tokens` above, so a worker
        # that will not exit costs nothing to kill. A plain join() here can block forever
        # when the forkserver leaves an exited worker unreaped. See WORKER_JOIN_TIMEOUT_S.
        stuck = join_workers(workers, logger=logger)
        if stuck:
            logger.warning(f"{stuck} of {cfg.num_workers} workers did not exit; results already collected")

        logger.info(f"Done! {len(merge_rules)} merge rules created, tokenizer has {len(tokens)} total tokens")
        return BPETokenizer(
            pretokenizer=pretokenizer,
            merge_rules=merge_rules,
            metadata=dict(
                config=cfg.model_dump(),
                corpus=corpus.metadata,
            ),
            tokens=tokens,
        )
