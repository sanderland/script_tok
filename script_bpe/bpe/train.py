import heapq
import itertools
import multiprocessing
from collections import defaultdict
from dataclasses import dataclass, field

from script_bpe.bpe.tokenizer import BPETokenizer, MergeRule, Token
from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.utils import TokenSeq, create_logger, mp_ctx, token_array


# --- Data Structures ---
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
        delta_counts = defaultdict(int)
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
        del ids[to_i + 1 :]  # ~ self.curr_seq = ids[:to_i+1]
        delta_counts[from_ids] -= num_merges
        return delta_counts, num_merges


@dataclass(slots=True)
class TokenPairCounts:
    """Stores counts for a pair, potentially specific to one worker's chunks."""

    chunk_id_to_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    count: int = 0

    def calculate_count(self, chunks: dict[int, ChunkTokenization]):
        self.count = sum(
            chunks[chunk_id].count * count_per_chunk for chunk_id, count_per_chunk in self.chunk_id_to_count.items()
        )
        return self.count


# --- Worker Process Function ---


def worker_process(
    worker_id: int,
    num_workers: int,
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    cmd_queue: multiprocessing.Queue,
    results_queue: multiprocessing.Queue,
):
    """Worker process that selects and processes its own chunks based on ID."""
    tokens = init_tokens(pretokenizer)

    local_chunks = {}
    local_pair_counts = defaultdict(TokenPairCounts)
    for base_token_seq, count in corpus.worker_iterate(worker_id, num_workers):  # lazy loading
        chunk_info = ChunkTokenization(base_token_seq, count=count)
        local_chunks[chunk_info.id] = chunk_info
        for i in range(len(base_token_seq) - 1):  # initial pair counts
            pair = (base_token_seq[i], base_token_seq[i + 1])
            if pretokenizer.bpe_merge_allowed([pair[0]], [pair[1]]):
                local_pair_counts[pair].chunk_id_to_count[chunk_info.id] += 1
            tokens[base_token_seq[i]].original_count += count
        tokens[base_token_seq[-1]].original_count += count
    del corpus  # no longer needed

    for t in tokens.values():
        t.current_count = t.original_count
    # Send initial pair counts to main process
    results_queue.put(
        (
            "INITIAL_COUNTS",
            {pair: pc.calculate_count(local_chunks) for pair, pc in local_pair_counts.items()},
        )
    )

    # Main worker loop
    while True:
        command = cmd_queue.get()

        if command == "GET_FINAL_STATE":
            results_queue.put(("FINAL_STATE", tokens))
            break
        merge_pair, new_token_id = command
        tokens[new_token_id] = Token(
            id=new_token_id, base_tokens=tokens[merge_pair[0]].base_tokens + tokens[merge_pair[1]].base_tokens
        )
        local_delta_counts = defaultdict(int)
        local_merge_count = 0
        chunks_affected = list(local_pair_counts[merge_pair].chunk_id_to_count.keys())
        for chunk_id in chunks_affected:
            chunk = local_chunks[chunk_id]
            chunk_count = chunk.count
            delta_counts, chunk_merges = chunk.merge(
                from_ids=merge_pair,
                to_id=new_token_id,
            )
            local_merge_count += chunk_merges * chunk_count

            # Update local count structures
            for pair, dc in delta_counts.items():
                if dc == 0:
                    continue
                delta_count = dc * chunk_count

                # Update pair counts
                pair_count = local_pair_counts.get(pair)
                if pair_count is None:
                    if not pretokenizer.bpe_merge_allowed(tokens[pair[0]].base_tokens, tokens[pair[1]].base_tokens):
                        continue
                    local_pair_counts[pair] = pair_count = TokenPairCounts()

                local_delta_counts[pair] += delta_count
                pair_count.chunk_id_to_count[chunk_id] += dc
                if pair_count.chunk_id_to_count[chunk_id] == 0:
                    del pair_count.chunk_id_to_count[chunk_id]

        for pair, dc in local_delta_counts.items():
            local_pair_counts[pair].count += dc
        results_queue.put(("DELTA_COUNTS", local_delta_counts, len(chunks_affected)))

        for ti in merge_pair:
            tokens[ti].current_count -= local_merge_count
        tokens[new_token_id].current_count = tokens[new_token_id].original_count = local_merge_count


# --- Helper Functions ---
def init_tokens(pretokenizer: Pretokenizer):
    return {k: Token(id=k, base_tokens=token_array([k])) for k in pretokenizer.base_tokens}


# --- Main Training Function ---
def train_bpe(
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    additional_vocab_size: int,
    num_workers: int = 4,
    verbose: bool = True,
):
    logger = create_logger("train_bpe", verbose)
    logger.info(f"Training with {num_workers} workers on corpus {corpus.name}: {corpus.metadata}")

    # --- Setup Workers and Communication ---
    cmd_queues = [mp_ctx.Queue() for _ in range(num_workers)]
    results_queue = mp_ctx.Queue()
    workers = []
    # Start worker processes
    for worker_id in range(num_workers):
        p = mp_ctx.Process(
            target=worker_process,
            args=(worker_id, num_workers, pretokenizer, corpus, cmd_queues[worker_id], results_queue),
            daemon=True,
        )
        workers.append(p)
        p.start()

    overall_pair_counts = defaultdict(int)
    # --- Collect Initial Pair Counts from Workers ---
    logger.info("Started workers, collecting initial pair counts")
    for worker_id in range(num_workers):
        msg_type, data = results_queue.get()
        assert msg_type == "INITIAL_COUNTS"
        for pair, count in data.items():
            overall_pair_counts[pair] += count

    # Initialize base tokens (we'll add counts later)
    tokens = init_tokens(pretokenizer)
    next_token_id = max(tokens.keys()) + 1
    merge_rules = []

    # --- Initialize Heap ---
    pair_counts_heap = [(-count, p) for p, count in overall_pair_counts.items()]
    heapq.heapify(pair_counts_heap)
    logger.info(f"Initialized with {len(pair_counts_heap):,d} token pairs")

    # --- Main Training Loop ---
    while len(merge_rules) < additional_vocab_size:
        if not pair_counts_heap:
            logger.warning("Heap empty - no more merges possible - overly small corpus?")
            break
        # --- Get Next Merge Pair ---
        neg_max_count, most_common_pair = heapq.heappop(pair_counts_heap)
        if most_common_pair not in overall_pair_counts:  # pairs eliminated as side effect of merges
            continue
        if overall_pair_counts[most_common_pair] != -neg_max_count:  # stale entry, reinsert
            heapq.heappush(pair_counts_heap, (-overall_pair_counts[most_common_pair], most_common_pair))
            continue

        (ta, tb) = most_common_pair

        # --- Create New Token and Broadcast Merge ---
        tokens[next_token_id] = Token(id=next_token_id, base_tokens=tokens[ta].base_tokens + tokens[tb].base_tokens)
        merge_rules.append(MergeRule(tokens_from=(ta, tb), token_to=next_token_id))
        for q in cmd_queues:
            q.put((most_common_pair, next_token_id))

        # --- Collect and Aggregate Results ---
        aggregated_deltas = defaultdict(int)
        n_chunks_affected = {}
        for worker_id in range(num_workers):
            msg_type, worker_deltas, n_chunks_affected[worker_id] = results_queue.get()
            assert msg_type == "DELTA_COUNTS"
            for pair, delta_count in worker_deltas.items():
                aggregated_deltas[pair] += delta_count

        # --- Update Global Counts and Heap ---
        for pair, delta_count in aggregated_deltas.items():
            if delta_count == 0:
                continue
            overall_pair_counts[pair] += delta_count

            if overall_pair_counts[pair] == 0:
                del overall_pair_counts[pair]
            elif delta_count > 0:  # < 0 gets handled by stale entries logic
                heapq.heappush(pair_counts_heap, (-overall_pair_counts[pair], pair))

        logger.info(
            f"Max count {-neg_max_count:10,d} from {len(overall_pair_counts):8,d} candidate pairs -> Created {tokens[next_token_id].pretty_repr(pretokenizer):<40} from "
            f"merging {tokens[ta].pretty_repr(pretokenizer):<30} & {tokens[tb].pretty_repr(pretokenizer):<30} \tTotal {sum(n_chunks_affected.values()):10,d} chunks affected ({min(n_chunks_affected.values()):5,d} - {max(n_chunks_affected.values()):5,d}/worker). Heap size {len(pair_counts_heap):,d}"
        )
        next_token_id += 1

    # --- Calculate Final Token Counts: sum up from workers ---
    logger.debug("Calculating final token counts and terminating workers")
    for q in cmd_queues:
        q.put("GET_FINAL_STATE")
    for _ in range(num_workers):
        msg_type, worker_tokens = results_queue.get()
        assert msg_type == "FINAL_STATE"
        for token_id, token in worker_tokens.items():
            tokens[token_id].original_count += token.original_count
            tokens[token_id].current_count += token.current_count

    # --- Terminate Workers ---
    for p in workers:
        p.join()

    # --- Build and Return Tokenizer ---
    logger.info(f"Done! {len(merge_rules)} merge rules created, tokenizer has {len(tokens)} total tokens")
    return BPETokenizer(
        pretokenizer=pretokenizer,
        merge_rules=merge_rules,
        metadata=dict(
            settings=dict(num_workers=num_workers),
            corpus=corpus.metadata,
            tokens=[t.report_dict(pretokenizer) for t in tokens.values()],
        ),
    )
