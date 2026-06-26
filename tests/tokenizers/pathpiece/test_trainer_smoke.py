from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.pathpiece.model import PathPieceModel
from script_bpe.tokenizers.pathpiece.trainer import PathPieceTrainer, PathPieceTrainerConfig
from script_bpe.utils import token_array


def _make_corpus(tmp_path, name: str):
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    text = ("the quick brown fox jumps over the lazy dog\n" * 12) + ("hello world hello world\n" * 8)
    corpus = PretokenizedCorpus.from_texts(
        name, texts=[text], pretokenizer=pretokenizer, base_path=str(tmp_path)
    )
    return pretokenizer, corpus, text


def test_ngram_init_smoke(tmp_path):
    pretokenizer, corpus, text = _make_corpus(tmp_path, "pp_ngram_smoke")
    config = PathPieceTrainerConfig(
        additional_vocab_size=20,
        init="ngram",
        # bytes pretokenizer has 256 atomic tokens; leave room for ~100 n-gram candidates
        init_vocab_size=356,
        max_token_width=8,
        prune_batch_fraction=0.25,
        verbose=False,
    )
    model = PathPieceTrainer(pretokenizer, corpus, config).train()

    assert isinstance(model, PathPieceModel)
    expected_size = len(pretokenizer.atomic_tokens) + 20
    assert len(model.tokens) == expected_size
    # all atomic tokens are required and present
    for atomic_id in pretokenizer.atomic_tokens:
        assert atomic_id in model.tokens
        assert model.tokens[atomic_id].required is True

    ids = model.encode(text)
    decoded = model.decode(ids)
    orig = sum(pretokenizer.pretokenize(text), token_array([]))
    got = sum(pretokenizer.pretokenize(decoded), token_array([]))
    assert got.tolist() == orig.tolist()

    assert model.metadata["tokenizer_variant"] == "pathpiece"
    assert model.metadata["init"] == "ngram"
    assert model.metadata["final_vocab_size"] == expected_size


def test_bpe_init_smoke(tmp_path):
    pretokenizer, corpus, text = _make_corpus(tmp_path, "pp_bpe_smoke")
    config = PathPieceTrainerConfig(
        additional_vocab_size=20,
        init="bpe",
        init_vocab_size=320,
        max_token_width=8,
        prune_batch_fraction=0.25,
        verbose=False,
        num_workers=1,
    )
    model = PathPieceTrainer(pretokenizer, corpus, config).train()

    expected_size = len(pretokenizer.atomic_tokens) + 20
    assert len(model.tokens) == expected_size
    ids = model.encode(text)
    decoded = model.decode(ids)
    orig = sum(pretokenizer.pretokenize(text), token_array([]))
    got = sum(pretokenizer.pretokenize(decoded), token_array([]))
    assert got.tolist() == orig.tolist()
    assert model.metadata["init"] == "bpe"


def test_history_is_monotone(tmp_path):
    pretokenizer, corpus, _ = _make_corpus(tmp_path, "pp_history")
    config = PathPieceTrainerConfig(
        additional_vocab_size=10,
        init="ngram",
        init_vocab_size=330,
        max_token_width=6,
        prune_batch_fraction=0.2,
        verbose=False,
    )
    model = PathPieceTrainer(pretokenizer, corpus, config).train()

    sizes = [h["vocab_before"] for h in model.metadata["history"]]
    assert sizes == sorted(sizes, reverse=True), f"vocab size should be monotonically non-increasing: {sizes}"
