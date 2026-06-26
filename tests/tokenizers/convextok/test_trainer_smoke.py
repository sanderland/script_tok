from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.convextok.model import ConvexTokModel
from script_bpe.tokenizers.convextok.trainer import ConvexTokTrainer, ConvexTokTrainerConfig
from script_bpe.utils import token_array


def _make_corpus(tmp_path, name):
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    text = ("the quick brown fox jumps over the lazy dog\n" * 20) + ("hello world hello world\n" * 12)
    corpus = PretokenizedCorpus.from_texts(name, texts=[text], pretokenizer=pretokenizer, base_path=str(tmp_path))
    return pretokenizer, corpus, text


def _roundtrips(pretokenizer, model, text):
    ids = model.encode(text)
    decoded = model.decode(ids)
    orig = sum(pretokenizer.pretokenize(text), token_array([]))
    got = sum(pretokenizer.pretokenize(decoded), token_array([]))
    return got.tolist() == orig.tolist()


def test_convextok_lp_smoke(tmp_path):
    pretokenizer, corpus, text = _make_corpus(tmp_path, "convextok_lp_smoke")
    cfg = ConvexTokTrainerConfig(
        additional_vocab_size=20, cmin=2, max_pretokens=200, max_token_width=12, rounding="det", verbose=False
    )
    model = ConvexTokTrainer(pretokenizer, corpus, cfg).train()

    assert isinstance(model, ConvexTokModel)
    expected = len(pretokenizer.atomic_tokens) + 20
    assert len(model.tokens) == expected, f"got {len(model.tokens)} != {expected}"
    for atomic_id in pretokenizer.atomic_tokens:
        assert atomic_id in model.tokens
        assert model.tokens[atomic_id].required is True
    assert _roundtrips(pretokenizer, model, text)
    assert model.metadata["tokenizer_variant"] == "convextok"
    assert model.metadata["lp_status"].startswith("highs")
    # a frequent multi-atomic-token substring should be selected
    assert any(len(t.atomic_tokens) > 1 for t in model.tokens.values())
    # LP objective is a valid lower bound: actual count >= bound (ratio >= 1)
    assert model.metadata["rounded_corpus_token_count"] >= model.metadata["lp_objective_lower_bound"] - 1e-6
    assert model.metadata["optimality_ratio"] >= 1.0 - 1e-6


def test_convextok_rounding_variants(tmp_path):
    pretokenizer, corpus, text = _make_corpus(tmp_path, "convextok_rounding")
    n_atomic = len(pretokenizer.atomic_tokens)
    budget = 15
    # det/bias pad up to the full budget; prob/int keep only positive-mass types
    # and may select fewer (the LP optimum can use < budget tokens). All must
    # round-trip losslessly regardless.
    for rounding in ("det", "bias", "prob", "int"):
        cfg = ConvexTokTrainerConfig(
            additional_vocab_size=budget, cmin=2, max_pretokens=200, max_token_width=12,
            rounding=rounding, verbose=False,
        )
        model = ConvexTokTrainer(pretokenizer, corpus, cfg).train()
        n_added = len(model.tokens) - n_atomic
        if rounding in ("det", "bias"):
            assert n_added == budget, rounding
        else:
            assert 0 < n_added <= budget, rounding
        assert _roundtrips(pretokenizer, model, text), rounding


def test_convextok_save_load(tmp_path):
    pretokenizer, corpus, text = _make_corpus(tmp_path, "convextok_saveload")
    cfg = ConvexTokTrainerConfig(
        additional_vocab_size=20, cmin=2, max_pretokens=200, max_token_width=12, verbose=False
    )
    model = ConvexTokTrainer(pretokenizer, corpus, cfg).train()
    path = tmp_path / "convextok.json.gz"
    model.save(str(path))
    loaded = ConvexTokModel.load(str(path))
    assert loaded.VERSION == "seconvextok-v1"
    assert model.encode(text) == loaded.encode(text)
