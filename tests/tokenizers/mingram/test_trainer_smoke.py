from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig
from script_bpe.utils import token_array


def test_mingram_trainer_smoke(tmp_path):
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    text = ("00010001 0011000\n" * 4) + ("111000 010101\n" * 4)
    corpus = PretokenizedCorpus.from_texts(
        "test_mingram_smoke",
        texts=[text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
    )
    config = MinGramTrainerConfig(
        additional_vocab_size=8,
        overshoot_factor=1.1,
        verbose=False,
    )
    model = MinGramTrainer(pretokenizer, corpus, config).train()

    assert isinstance(model, MinGramModel)
    assert len(pretokenizer.atomic_tokens) <= len(model.tokens) <= len(pretokenizer.atomic_tokens) + 8
    assert "tokens/pretoken" in model.metadata
    assert "objective" in model.metadata

    ids = model.encode(text)
    decoded = model.decode(ids)
    orig_atomic = sum(pretokenizer.pretokenize(text), token_array([]))
    dec_atomic = sum(pretokenizer.pretokenize(decoded), token_array([]))
    assert dec_atomic.tolist() == orig_atomic.tolist()


def test_mingram_pp_pruning_smoke(tmp_path):
    pretokenizer = get_pretokenizer("bytes_gpt4_cb")
    text = ("abracadabra abracadabra\n" * 6) + ("banana bandana banana\n" * 6)
    corpus = PretokenizedCorpus.from_texts(
        "test_mingram_pp_smoke",
        texts=[text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
    )
    config = MinGramTrainerConfig(
        additional_vocab_size=10,
        overshoot_factor=2.0,
        pruning_shrinking_factor=0.5,
        prune_criterion="mi",
        verbose=False,
    )
    model = MinGramTrainer(pretokenizer, corpus, config).train()

    assert isinstance(model, MinGramModel)
    assert model.metadata["config"]["prune_criterion"] == "mi"
    assert len(pretokenizer.atomic_tokens) <= len(model.tokens) <= len(pretokenizer.atomic_tokens) + 10

    ids = model.encode(text)
    decoded = model.decode(ids)
    orig_atomic = sum(pretokenizer.pretokenize(text), token_array([]))
    dec_atomic = sum(pretokenizer.pretokenize(decoded), token_array([]))
    assert dec_atomic.tolist() == orig_atomic.tolist()
