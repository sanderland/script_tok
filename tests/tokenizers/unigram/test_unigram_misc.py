import math
import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.unigram import UnigramModel
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig


def _tiny_text():
    return "hello 0011 world"


@pytest.mark.parametrize("pretokenizer_name", ["bytes_gpt4_cb", "scriptenc_cb"])
def test_unigram_stats_and_report(tmp_path, pretokenizer_name):
    pre = get_pretokenizer(pretokenizer_name)
    corpus = PretokenizedCorpus.from_texts("uni_misc", [_tiny_text()], pre, base_path=str(tmp_path))
    model = UnigramTrainer(pre, corpus, UnigramTrainerConfig(additional_vocab_size=8, num_workers=1, verbose=False)).train()

    stats = model.stats()
    for k in [
        "num_tokens",
        "num_atomic_tokens",
        "num_multi_tokens",
        "num_undecodable",
        "avg_token_length_bt",
        "avg_char_length",
        "longest_tokens_by_atomic",
    ]:
        assert k in stats

    rep = model.report()
    assert isinstance(rep, str) and len(rep) > 0 and rep.startswith("# Unigram Tokenizer Report")

    # save/load roundtrip
    path = str(tmp_path / "uni.json")
    model.save(path)
    loaded = UnigramModel.load(path)
    assert loaded.encode(_tiny_text()) == model.encode(_tiny_text())


