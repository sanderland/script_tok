import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.bpe.stats import compression_curve
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig


def _zeros_text():
    return "\n".join([(" " + "0" * i) * j for i in range(6) for j in range(3)])


@pytest.mark.parametrize("pretokenizer_name", ["bytes_gpt4", "scriptenc_cb"])
def test_bpe_io_stats_and_report(tmp_path, pretokenizer_name):
    text = _zeros_text()
    pre = get_pretokenizer(pretokenizer_name)
    corpus = PretokenizedCorpus.from_texts(
        name=f"bpe_misc_{pretokenizer_name}", texts=[text], pretokenizer=pre, base_path=str(tmp_path)
    )

    trainer = BPETrainer(pre, corpus, BPETrainerConfig(additional_vocab_size=6, num_workers=1, verbose=False))
    tok = trainer.train()

    # save/load roundtrip
    path = str(tmp_path / "tok.json")
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode(text) == tok.encode(text)

    # stats shape and report
    stats = loaded.stats()
    for k in [
        "num_merge_rules",
        "num_tokens",
        "num_undecodeable",
        "last_merge_count",
        "longest_tokens_atomic_tokens",
        "longest_tokens_chars",
        "avg_token_length_bt",
    ]:
        assert k in stats
    assert isinstance(loaded.report(), str) and len(loaded.report()) > 0


def test_compression_curve_monotone(tmp_path):
    pre = get_pretokenizer("bytes_gpt4")
    text = _zeros_text()
    corpus = PretokenizedCorpus.from_texts("bpe_curve", [text], pre, base_path=str(tmp_path))
    tok = BPETrainer(pre, corpus, BPETrainerConfig(additional_vocab_size=5, num_workers=1, verbose=False)).train()

    curve = compression_curve(tok)
    assert len(curve) == len(tok.merge_rules) + 1
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))


