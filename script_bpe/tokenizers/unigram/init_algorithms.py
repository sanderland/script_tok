from __future__ import annotations

from collections import Counter
from typing import Iterable

from script_bpe.pretokenize import Pretokenizer
from script_bpe.corpus import PretokenizedCorpus

def _suffix_array(token_ids: list[int]) -> list[int]:
    """Suffix array over a list of integers using a doubling algorithm.

    Returns an array sa such that seq[sa[i]:] is the i-th suffix in lexicographic order.
    """
    n = len(token_ids)
    if n == 0:
        return []
    # Initial ranking by value (shifted to be non-negative)
    unique_vals = {v for v in token_ids}
    min_v = min(unique_vals)
    rank: list[int] = [v - min_v + 1 for v in token_ids]  # ensure >= 1
    sa = list(range(n))
    k = 1
    tmp: list[int] = [0] * n
    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else 0))
        tmp[sa[0]] = 1
        for i in range(1, n):
            prev, cur = sa[i - 1], sa[i]
            r1_prev = rank[prev]
            r1_cur = rank[cur]
            r2_prev = rank[prev + k] if prev + k < n else 0
            r2_cur = rank[cur + k] if cur + k < n else 0
            tmp[cur] = tmp[prev] + (1 if (r1_prev != r1_cur or r2_prev != r2_cur) else 0)
        rank, tmp = tmp, rank
        if rank[sa[-1]] == n:
            break
        k <<= 1
    return sa


def _kasai_lcp(token_ids: list[int], suffix_array: list[int]) -> list[int]:
    """Kasai LCP for integer sequences.

    lcp[i] = LCP(suffix at sa[i], suffix at sa[i-1]), lcp[0] = 0.
    """
    n = len(token_ids)
    lcp_array: list[int] = [0] * n
    if n == 0:
        return lcp_array
    rank_by_pos = [0] * n
    for rank, pos in enumerate(suffix_array):
        rank_by_pos[pos] = rank
    h = 0
    for i in range(n):
        r = rank_by_pos[i]
        if r == 0:
            h = 0
            continue
        j = suffix_array[r - 1]
        while i + h < n and j + h < n and token_ids[i + h] == token_ids[j + h]:
            h += 1
        lcp_array[r] = h
        if h:
            h -= 1
    return lcp_array


def _emit_from_lcp(
    concatenated_token_ids: list[int],
    suffix_array: list[int],
    lcp_array: list[int],
    id_to_token: list[int],
    max_token_length: int,
    *,
    delimiter_id: int,
    repair_with: Pretokenizer | None = None,
) -> Counter:
    """Enumerate repeated substrings (length>1, freq>1) from LCP intervals.

    Returns Counter mapping tuple[token,...] -> frequency.
    If repair_with is provided, invalid candidates are shrunk to longest valid prefix
    per pretokenizer.token_allowed.
    """
    n = len(concatenated_token_ids)
    out: Counter[tuple[int, ...]] = Counter()
    stack: list[dict[str, int]] = []
    # scan
    for i in range(1, n):
        height = lcp_array[i]
        start_pos = i - 1
        while stack and stack[-1]["height"] > height:
            top = stack.pop()
            freq = i - top["start_pos"]
            length = top["height"]
            if freq > 1 and length > 1:
                offset = suffix_array[top["start_pos"]]
                if offset + length <= n:
                    span = concatenated_token_ids[offset : offset + length]
                    if delimiter_id in span:
                        k = span.index(delimiter_id)
                        if 1 < k <= max_token_length:
                            candidate: tuple[int, ...] = tuple(id_to_token[t] for t in span[:k])
                            if repair_with is not None and not repair_with.token_allowed(candidate):
                                # shrink to longest valid prefix
                                while len(candidate) > 1 and not repair_with.token_allowed(candidate):
                                    candidate = candidate[:-1]
                            if len(candidate) > 1 and (repair_with is None or repair_with.token_allowed(candidate)):
                                out[candidate] += freq
                        start_pos = top["start_pos"]
                        continue
                    if length <= max_token_length:
                        candidate: tuple[int, ...] = tuple(id_to_token[t] for t in span)
                        if repair_with is not None and not repair_with.token_allowed(candidate):
                            # shrink to longest valid prefix
                            while len(candidate) > 1 and not repair_with.token_allowed(candidate):
                                candidate = candidate[:-1]
                        if len(candidate) > 1 and (repair_with is None or repair_with.token_allowed(candidate)):
                            out[candidate] += freq
            start_pos = top["start_pos"]
        if not stack or stack[-1]["height"] < height:
            stack.append({"height": height, "start_pos": start_pos})

    # flush
    i = n
    height = 0
    start_pos = n - 1
    while stack and stack[-1]["height"] > height:
        top = stack.pop()
        freq = i - top["start_pos"]
        length = top["height"]
        if freq > 1 and length > 1:
            offset = suffix_array[top["start_pos"]]
            if offset + length <= n:
                span = concatenated_token_ids[offset : offset + length]
                if delimiter_id in span:
                    k = span.index(delimiter_id)
                    if 1 < k <= max_token_length:
                        candidate = tuple(id_to_token[t] for t in span[:k])
                        if repair_with is not None and not repair_with.token_allowed(candidate):
                            while len(candidate) > 1 and not repair_with.token_allowed(candidate):
                                candidate = candidate[:-1]
                        if len(candidate) > 1 and (repair_with is None or repair_with.token_allowed(candidate)):
                            out[candidate] += freq
                elif length <= max_token_length:
                    candidate = tuple(id_to_token[t] for t in span)
                    if repair_with is not None and not repair_with.token_allowed(candidate):
                        while len(candidate) > 1 and not repair_with.token_allowed(candidate):
                            candidate = candidate[:-1]
                    if len(candidate) > 1 and (repair_with is None or repair_with.token_allowed(candidate)):
                        out[candidate] += freq
        start_pos = top["start_pos"]
    return out


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
                    freq[span] += count
    return freq


def compute_substring_frequencies_spm(
    pretokenizer: Pretokenizer,
    corpus: PretokenizedCorpus,
    max_token_length: int,
    *,
    repair: bool = False,
) -> Counter:
    """SPM-like global enumeration using a suffix array over concatenated token IDs.

    - Build integer IDs for atomic tokens. Use 0 as the sentence delimiter.
    - Concatenate all sequences with delimiter (repeated by frequency).
    - Build suffix array + LCP.
    - Extract repeated substrings (freq>1, length>1), optionally repairing invalid
      candidates to their longest valid prefix using the pretokenizer.
    """
    # Build token-id mapping.
    # Ensure stable order: use enumeration over pretokenizer.atomic_tokens
    token_to_id: dict[int, int] = {}
    id_to_token: list[int] = [0]  # id 0 is reserved for delimiter
    next_id = 1
    for t in pretokenizer.atomic_tokens:
        token_to_id[t] = next_id
        id_to_token.append(t)
        next_id += 1

    # Build concatenated id sequence with delimiter 0.
    concatenated_ids: list[int] = []
    for atomic_token_seq, count in corpus:
        if not atomic_token_seq:
            continue
        seq_ids = [token_to_id[t] for t in atomic_token_seq]
        for _ in range(count):
            if concatenated_ids:
                concatenated_ids.append(0)
            concatenated_ids.extend(seq_ids)

    if not concatenated_ids:
        return Counter()

    suffix_array = _suffix_array(concatenated_ids)
    lcp_array = _kasai_lcp(concatenated_ids, suffix_array)
    return _emit_from_lcp(
        concatenated_ids,
        suffix_array,
        lcp_array,
        id_to_token,
        max_token_length,
        delimiter_id=0,
        repair_with=pretokenizer if repair else None,
    )


