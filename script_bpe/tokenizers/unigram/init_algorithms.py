from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal
from script_bpe.pretokenize import Pretokenizer
from script_bpe.corpus import PretokenizedCorpus

# Type definitions
FlatCorpusT = list[
    tuple[tuple[int, ...], int]
]  # List of tokenized words (each word is tuple of token IDs) with frequency
CorpusSuffixArrayT = list[tuple[int, int]]  # (word_index, position_in_word)


STRATEGY = Literal["long", "intermediate", "fallback"]


def _is_single_pretoken(pretokenizer: Pretokenizer, span: tuple[int, ...]) -> bool:
    return len(pretokenizer.pretokenize(pretokenizer.decode(span))) == 1


def compute_substring_frequencies_simple(
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    max_token_length: int,
) -> Counter:
    """Simple local enumeration over each pretokenized sequence.

    Returns Counter of tuple[atomic_token,...] -> frequency.
    """
    freq: Counter[tuple[int, ...]] = Counter()
    for atomic_token_seq, count in corpus:
        seq_len = len(atomic_token_seq)
        for start in range(seq_len):
            for end in range(start + 1, min(seq_len + 1, start + max_token_length + 1)):
                span = tuple(atomic_token_seq[start:end])
                if pretokenizer.token_allowed(span):
                    if _is_single_pretoken(pretokenizer, span):
                        freq[span] += count
    return freq


def _suffix_array_corpus(corpus: FlatCorpusT, max_token_length: int) -> CorpusSuffixArrayT:
    """Build grouped suffix arrays for a flat corpus.

    For each word (sequence of token IDs) and each position in that word, a
    suffix is created and grouped by the suffix's first token. Within each
    group, suffixes are sorted lexicographically by the remaining token
    sequence.

    Args:
        corpus: List of (word_token_ids, frequency). Frequencies are not used
            for constructing suffix arrays, only the token sequences are.
        max_token_length: Maximum pattern length we care about. Suffixes are
            sorted by their first `max_token_length` tokens to ensure correct
            clustering of patterns up to this length.

    Yields:
        tuple[int, list[tuple[int, int]]]: Pairs of (initial_token_id,
        suffix_array). Each suffix_array is a list of (word_index,
        position_in_word) entries sorted by the suffix text.
    """

    # Create all suffixes: for each word, create suffix starting at each position
    suffix_arrays = defaultdict(list)
    print(f"Building suffix arrays for {len(corpus):,} corpus entries")

    for word_idx, (word_tokens, _) in enumerate(corpus):
        for pos_in_word in range(len(word_tokens)):
            suffix_arrays[word_tokens[pos_in_word]].append((word_idx, pos_in_word))

    # Sort suffixes by lexicographic order of the token sequence
    # Strategy: Use limited-length prefix keys to balance memory and speed.
    # We materialize only the first max_token_length tokens of each suffix as a tuple.
    # This ensures that all suffixes sharing a prefix of length max_token_length
    # are contiguous, which is sufficient for finding patterns up to that length.

    for idx, (token_id, suffix_array) in enumerate(suffix_arrays.items()):
        # Create sort keys: (prefix_tuple)
        def make_sort_key(entry: tuple[int, int]) -> tuple[int, ...]:
            word_idx, pos = entry
            tokens = corpus[word_idx][0]
            # We only need to sort by the first max_token_length tokens
            # to correctly cluster patterns of that length.
            return tokens[pos : pos + max_token_length]

        suffix_array.sort(key=make_sort_key)
        yield token_id, suffix_array


def extract_frequent_patterns_lcp(
    pretokenizer: Pretokenizer,
    corpus: FlatCorpusT,
    suffix_array: CorpusSuffixArrayT,
    max_token_length: int,
    strategy: STRATEGY = "long",
) -> Counter[tuple[int, ...]]:
    """Extract repeated substrings using an LCP stack over a grouped suffix array.

    Processes a single group of suffixes that all start with the same token and
    returns total frequencies for repeated substrings up to ``max_token_length``.
    Frequencies are computed by summing the corpus word frequencies across the
    ranks in each LCP interval. If ``strategy`` is "intermediate" or "fallback", counts for
    all strict prefixes of each discovered pattern are also included starting at
    the appropriate LCP height; otherwise only the full pattern length is
    counted ("long").
    The "fallback" strategy iterates from longest to shortest valid prefix and stops.

    Args:
        corpus: Flat corpus of (word_token_ids, frequency).
        suffix_array: Sorted list of (word_index, position_in_word) for one
            initial token.
        max_token_length: Maximum allowed pattern length in tokens.
        strategy: Whether to also count intermediate prefixes of
            each pattern.

    Returns:
        Counter mapping token-id tuples to integer frequencies.
    """
    N = len(suffix_array)

    def _common_prefix_length(rank_a: int, rank_b: int) -> int:
        word_idx_a, pos_a = suffix_array[rank_a]
        word_a = corpus[word_idx_a][0]
        word_idx_b, pos_b = suffix_array[rank_b]
        word_b = corpus[word_idx_b][0]

        d = 0
        max_d = min(len(word_a) - pos_a, len(word_b) - pos_b)
        while d < max_d and word_a[pos_a + d] == word_b[pos_b + d]:
            d += 1

        return d

    patterns = defaultdict(int)
    stack = []  # stack of (height, start_rank).

    # We iterate N times (from rank 1 to N, where N is the virtual sentinel)
    # This loop processes the LCP height *between* suffix[i-1] and suffix[i].
    for rank in range(1, N + 1):
        # Calculate LCP height between SA[rank-1] and SA[rank]
        if rank == N:
            height = 0
        else:
            height = _common_prefix_length(rank - 1, rank)

        # Initialize the start rank for the interval we might push/extend
        start_rank = rank - 1
        # 2. Popping Phase: Close intervals whose height is greater than the current LCP height (h)
        while stack and stack[-1][0] > height:
            pattern_length, interval_start_rank = stack.pop()

            word_idx, pos = suffix_array[interval_start_rank]
            pattern_tokens = corpus[word_idx][0][pos : pos + pattern_length]

            if pattern_length <= max_token_length:
                # Calculate total frequency by summing frequencies of words within the interval ranks
                frequency = sum(
                    corpus[suffix_array[suffix_rank][0]][1] for suffix_rank in range(interval_start_rank, rank)
                )
                if frequency > 1:  # Save the found pattern
                    if strategy in ("intermediate", "fallback"):  # Major change from sentencepiece
                        intermediate_start_rank = stack[-1][0] + 1 if stack else height + 1  # previous height + 1
                    else:
                        intermediate_start_rank = pattern_length

                    for e in reversed(range(intermediate_start_rank, pattern_length + 1)):
                        sub_pattern = pattern_tokens[:e]
                        if pretokenizer.token_allowed(sub_pattern):
                            if _is_single_pretoken(pretokenizer, sub_pattern):  # todo: remove
                                patterns[sub_pattern] += frequency
                                if strategy == "fallback":
                                    break

            # Crucial Step: The start rank of the popped interval becomes the new candidate start rank,
            # because this interval is merging with the current or subsequent interval defined by height 'h'.
            start_rank = interval_start_rank

        # 3. Pushing Phase: Extend or push a new interval
        # If the stack is empty, or the current height 'h' is higher than the stack top,
        # we have found a new or extended maximal repeating pattern.
        if not stack or stack[-1][0] < height:
            stack.append((height, start_rank))

    return patterns


def compute_substring_frequencies_corpus(
    pretokenizer: Pretokenizer,
    corpus: FlatCorpusT,
    max_token_length: int,
    strategy: STRATEGY = "long",
) -> Counter[tuple[int, ...]]:
    """Compute substring frequencies for the entire corpus.

    Builds grouped suffix arrays by first token and merges counts from
    ``extract_frequent_patterns_lcp`` for each group.

    Args:
        corpus: Flat corpus of (word_token_ids, frequency).
        max_token_length: Maximum pattern length to count (in tokens).
        strategy: Pattern counting strategy.

    Returns:
        Counter mapping token-id tuples to integer frequencies.
    """
    assert strategy in ("long", "intermediate", "fallback")
    init_patterns = Counter()

    # Step 1: Build suffix arrays grouped by first token
    processed_count = 0
    for _tid, suffix_array in _suffix_array_corpus(corpus, max_token_length):
        patterns_i = extract_frequent_patterns_lcp(pretokenizer, corpus, suffix_array, max_token_length, strategy)
        init_patterns.update(patterns_i)
        processed_count += 1

    # Add all corpus items with frequency >= 2 as candidates
    for word_tokens, freq in corpus:
        if freq >= 2 and 1 < len(word_tokens) <= max_token_length:
            if word_tokens not in init_patterns:
                init_patterns[word_tokens] = freq
            else:  # this plus others
                decoded_word = pretokenizer.decode(word_tokens)
                assert init_patterns[word_tokens] > freq, (
                    f"{init_patterns[word_tokens]} > {freq} for word_tokens={word_tokens} decoded_word={decoded_word}"
                )

    # Add single characters, which are not always counted by the LCP algorithm
    single_token_count = Counter()
    for word_tokens, freq in corpus:
        for token in word_tokens:
            single_token_count[token] += freq
    for token, freq in single_token_count.items():
        init_patterns[(token,)] = freq  # overwrite if already present

    return init_patterns
