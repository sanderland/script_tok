from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.pathpiece.model import PathPieceModel
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array


def _aaa_model() -> tuple[PathPieceModel, int]:
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    a_id = pretokenizer.encode_text("a")[0].atomic_token_ids[0]
    tokens = [
        UnigramToken(id=0, atomic_tokens=token_array([a_id]), required=True),
        UnigramToken(id=1, atomic_tokens=token_array([a_id, a_id])),
        UnigramToken(id=2, atomic_tokens=token_array([a_id, a_id, a_id])),
    ]
    return PathPieceModel(pretokenizer, tokens), a_id


def test_min_token_segmentation_picks_widest():
    model, _ = _aaa_model()
    # 'aaaaa' (5 a's) has shortest paths of length 2: [3,2] or [2,3]; longest-token
    # tiebreak should pick the variant whose *last* token is widest, i.e. [2,3].
    ids = model.encode("aaaaa")
    assert ids == [1, 2]


def test_longest_token_tiebreak_on_three_chars():
    model, _ = _aaa_model()
    # 'aaa' has competing minimal paths: [3] (one token) and [aa, a]/[a, aa] (two tokens).
    # One token wins.
    assert model.encode("aaa") == [2]


def test_two_char_picks_pair():
    model, _ = _aaa_model()
    assert model.encode("aa") == [1]


def test_falls_back_to_atomic_only():
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    a_id = pretokenizer.encode_text("a")[0].atomic_token_ids[0]
    b_id = pretokenizer.encode_text("b")[0].atomic_token_ids[0]
    tokens = [
        UnigramToken(id=0, atomic_tokens=token_array([a_id]), required=True),
        UnigramToken(id=1, atomic_tokens=token_array([b_id]), required=True),
    ]
    model = PathPieceModel(pretokenizer, tokens)
    assert model.encode("ab") == [0, 1]


def test_round_trip_save_load(tmp_path):
    model, _ = _aaa_model()
    path = tmp_path / "pp.json.gz"
    model.save(str(path))
    loaded = PathPieceModel.load(str(path))
    assert loaded.encode("aaaaa") == model.encode("aaaaa")
    # required flag survives the round trip
    assert loaded.tokens[0].required is True
    assert loaded.tokens[1].required is False


def test_decode_round_trip():
    model, _ = _aaa_model()
    text = "aaaaa"
    ids = model.encode(text)
    decoded = model.decode(ids)
    assert decoded == text
