import math

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.mingram.model import ATOMIC_FALLBACK_LOG_PROB_FLOOR, MinGramModel
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array


def _make_single_char_model() -> tuple[MinGramModel, int]:
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    a_id = pretokenizer.encode_text("a")[0].atomic_token_ids[0]
    tokens = [
        UnigramToken(id=0, atomic_tokens=token_array([a_id]), log_prob=math.log(0.4), required=True),
        UnigramToken(id=1, atomic_tokens=token_array([a_id, a_id]), log_prob=math.log(0.4)),
        UnigramToken(id=2, atomic_tokens=token_array([a_id, a_id, a_id]), log_prob=math.log(0.2)),
    ]
    return MinGramModel(pretokenizer, tokens), a_id


def test_encode_api_and_decode_roundtrip():
    model, _ = _make_single_char_model()
    text = "aaaaa"

    ids = model.encode(text)
    tokens = model.encode(text, return_tokens=True)

    assert isinstance(ids, list)
    assert ids == [token.id for token in tokens]

    decoded = model.decode(ids)
    orig_atomic = sum(model.pretokenizer.pretokenize(text), token_array([]))
    dec_atomic = sum(model.pretokenizer.pretokenize(decoded), token_array([]))
    assert dec_atomic.tolist() == orig_atomic.tolist()


def test_encode_single_chunk_tie_uses_dfs_lifo_longest_first():
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    a_id = pretokenizer.encode_text("a")[0].atomic_token_ids[0]
    tokens = [
        UnigramToken(id=0, atomic_tokens=token_array([a_id]), log_prob=0.0, required=True),
        UnigramToken(id=1, atomic_tokens=token_array([a_id, a_id]), log_prob=0.0),
    ]
    model = MinGramModel(pretokenizer, tokens)

    # For "aaa", minimal token-count paths are [aa, a] and [a, aa] with equal logprob.
    # DFS/LIFO traversal should settle the first as designed.
    chosen = model.encode("aaa")
    assert chosen == [1, 0]


def test_encode_allows_required_atomic_fallback_with_negative_infinity_scores():
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    a_id = pretokenizer.encode_text("a")[0].atomic_token_ids[0]
    b_id = pretokenizer.encode_text("b")[0].atomic_token_ids[0]
    tokens = [
        UnigramToken(id=0, atomic_tokens=token_array([a_id]), log_prob=float("-inf"), required=True),
        UnigramToken(id=1, atomic_tokens=token_array([b_id]), log_prob=float("-inf"), required=True),
    ]
    model = MinGramModel(pretokenizer, tokens)

    assert model.encode("ab") == [0, 1]


def test_load_restores_atomic_fallback_metadata(tmp_path):
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    a_id = pretokenizer.encode_text("a")[0].atomic_token_ids[0]
    b_id = pretokenizer.encode_text("b")[0].atomic_token_ids[0]
    tokens = [
        UnigramToken(id=0, atomic_tokens=token_array([a_id]), log_prob=float("-inf"), required=True),
        UnigramToken(id=1, atomic_tokens=token_array([b_id]), log_prob=float("-inf"), required=True),
        UnigramToken(id=2, atomic_tokens=token_array([a_id, b_id]), log_prob=-1.5),
    ]
    model = MinGramModel(pretokenizer, tokens)
    path = tmp_path / "mingram.json.gz"
    model.save(str(path))

    loaded = MinGramModel.load(str(path))

    assert loaded.tokens[0].log_prob == ATOMIC_FALLBACK_LOG_PROB_FLOOR
    assert loaded.tokens[1].log_prob == ATOMIC_FALLBACK_LOG_PROB_FLOOR
    assert loaded.tokens[2].log_prob == -1.5
