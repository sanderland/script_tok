import math
import pytest

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.unigram.model import Lattice, Trie, UnigramModel, UnigramToken
from script_bpe.utils import token_array


@pytest.fixture
def bytes_pre():
    return get_pretokenizer("bytes_gpt4_cb")


@pytest.fixture
def zero_id(bytes_pre):
    return bytes_pre.encode_text("0")[0].atomic_token_ids[0]


@pytest.fixture
def one_id(bytes_pre):
    return bytes_pre.encode_text("1")[0].atomic_token_ids[0]


@pytest.fixture
def tiny_tokens(bytes_pre, zero_id, one_id):
    t0 = UnigramToken(id=0, atomic_tokens=token_array([zero_id]), log_prob=math.log(0.4), required=True)
    t1 = UnigramToken(id=1, atomic_tokens=token_array([zero_id, zero_id]), log_prob=math.log(0.5))
    t2 = UnigramToken(id=2, atomic_tokens=token_array([one_id]), log_prob=math.log(0.1), required=True)
    return bytes_pre, [t0, t1, t2]


@pytest.fixture
def tiny_model(tiny_tokens):
    pre, tokens = tiny_tokens
    return UnigramModel(pre, tokens)


@pytest.fixture
def seq_00(zero_id):
    return token_array([zero_id, zero_id])


@pytest.fixture
def seq_001(zero_id, one_id):
    return token_array([zero_id, zero_id, one_id])


def test_trie_prefix_search(tiny_tokens, zero_id, bytes_pre):
    _, tokens = tiny_tokens
    trie = Trie(tokens)

    # For "00" we should see both the single "0" and the double "00"
    matches = trie.find_prefixes(token_array([zero_id, zero_id]))
    found_ids = {t.id for t in matches}
    assert 0 in found_ids  # '0'
    assert 1 in found_ids  # '00'

    # No matches for a token not in trie (use a third byte '2')
    two_id = bytes_pre.encode_text("2")[0].atomic_token_ids[0]
    assert trie.find_prefixes(token_array([two_id])) == []


def test_lattice_viterbi_and_all_paths(tiny_model, seq_001):
    lattice = tiny_model.make_lattice(seq_001)

    # Viterbi should choose ["00", "1"]
    path, score = lattice.viterbi()
    assert [t.id for t in path] == [1, 2]

    # All paths should include ["0", "0", "1"] as well
    all_paths = list(lattice.all_paths())
    assert any([t.id for t in p] == [0, 0, 2] for p, _ in all_paths)

    # log-sum-exp of path probs equals z
    z, _ = lattice.calc_marginal()
    total_path_prob = sum(math.exp(lp) for _, lp in all_paths)
    assert math.isclose(total_path_prob, math.exp(z), rel_tol=1e-9)


def test_lattice_viterbi_no_single_token(tiny_model, seq_00):
    lattice = tiny_model.make_lattice(seq_00)

    # Disallow single-token path covering the whole sequence; expect ["0","0"]
    path, _ = lattice.viterbi(allow_single_token=False)
    assert [t.id for t in path] == [0, 0]


def test_calc_marginal_bounds(tiny_model, seq_001):
    lattice = tiny_model.make_lattice(seq_001)
    z, probs = lattice.calc_marginal()

    # Each token's expected count contribution is between 0 and 1 here
    for v in probs.values():
        assert 0.0 <= v <= 1.0

    # Expected total tokens in segmentation lies between 2 and 3 for "001"
    total_expected_tokens = sum(probs.values())
    assert 2.0 <= total_expected_tokens <= 3.0


def test_unigram_model_encode_prefers_longer_token(tiny_model):
    # Sanity: path should be ["00", "1"] for "001"
    ids = tiny_model.encode("001")
    assert ids == [1, 2]


