from collections import Counter

import pytest

from script_bpe.tokenizers.bpe.trainer import ChunkTokenization
from script_bpe.utils import token_array


def count_pairs(sequence):
    return Counter(zip(sequence, sequence[1:]))


def delta_counts(original_counts, result_counts):
    return {k: result_counts.get(k, 0) - original_counts.get(k, 0) for k in set(original_counts) | set(result_counts)}


@pytest.mark.parametrize(
    "curr_seq, pair, new_token, expected_seq",
    [
        ([1, 2, 3, 4], (2, 3), 5, [1, 5, 4]),
        ([1, 2, 3, 4], (3, 5), 6, [1, 2, 3, 4]),
        ([1, 2, 3, 2, 3, 4], (2, 3), 5, [1, 5, 5, 4]),
        ([], (1, 2), 3, []),
        ([1, 2, 1, 2, 1, 2], (1, 2), 3, [3, 3, 3]),
        ([1, 1, 1], (1, 1), 2, [2, 1]),
        ([1, 1, 1, 1], (1, 1), 2, [2, 2]),
        ([1, 1, 1, 1, 1, 1], (1, 1), 2, [2, 2, 2]),
        ([3, 1, 1, 1, 1, 1, 1, 4], (1, 1), 2, [3, 2, 2, 2, 4]),
        ([3, 1, 1, 1, 1, 1, 1, 1, 1, 4], (1, 1), 2, [3, 2, 2, 2, 2, 4]),
    ],
)
def test_chunk_merge(curr_seq, pair, new_token, expected_seq):
    chunk = ChunkTokenization(curr_seq=token_array(curr_seq))
    pair_delta_from_merge, merge_count = chunk.merge(pair, new_token)
    pair_delta_manual = delta_counts(count_pairs(curr_seq), count_pairs(expected_seq))

    assert chunk.curr_seq.tolist() == expected_seq
    delta_from_merge_nz = {k: v for k, v in pair_delta_from_merge.items() if v != 0}
    # Compare against expected pair delta computed from before/after sequences
    delta_manual_nz = {k: v for k, v in pair_delta_manual.items() if v != 0}
    assert delta_from_merge_nz == delta_manual_nz, f"Pair delta mismatch: {delta_from_merge_nz} != {delta_manual_nz}"

    token_delta_manual = {k: v for k, v in delta_counts(Counter(curr_seq), Counter(expected_seq)).items()}
    if pair[0] == pair[1]:
        assert token_delta_manual.get(pair[0], 0) == -merge_count * 2
    else:
        assert token_delta_manual.get(pair[0], 0) == -merge_count
        assert token_delta_manual.get(pair[1], 0) == -merge_count
