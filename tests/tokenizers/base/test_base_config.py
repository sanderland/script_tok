from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.corpus import PretokenizedCorpus


def test_trainer_config_and_logger(tmp_path):
    cfg = TrainerConfig(additional_vocab_size=1, num_workers=1, verbose=False)
    pre = get_pretokenizer("bytes_gpt4_cb")
    corpus = PretokenizedCorpus.from_texts("base_cfg", ["hi"], pretokenizer=pre, base_path=str(tmp_path))

    # instantiate minimal BaseTrainer via subclassing
    class Dummy(BaseTrainer):
        pass

    d = Dummy(pre, corpus, cfg)
    # logger exists and has name
    assert d.logger.name == "Dummy"


