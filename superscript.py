
# --- Configuration ---
pretokenizer_name = "scriptenc_cb"
corpus_name = "eng_latn_300mb"
retrain = False
n_cpus = 4

trainer_config_kwargs = {
    "additional_vocab_size": 100_000,
    "init_vocab_algo": "corpus_repair",
    "initial_vocab_factor": 10,
}
# --- End Configuration ---

from script_bpe.train import train_tokenizer
from script_bpe.utils import create_logger

logger = create_logger("superscript", verbose=True)

logger.info(f"Pretokenizer: {pretokenizer_name}, Corpus: {corpus_name}")
logger.info(f"Trainer Config: {trainer_config_kwargs}")

model = train_tokenizer(
    pretokenizer_name=pretokenizer_name,
    model_name="unigram",
    corpus_name=corpus_name,
    additional_vocab_size=trainer_config_kwargs["additional_vocab_size"],
    n_cpus=n_cpus,
    retrain=retrain,
    report=False,
    trainer_config_kwargs=trainer_config_kwargs,
)

if model:
    logger.info("Run complete.")
    logger.info(f"Final model stats: {model.metadata}")
else:
    logger.error("Could not train or load a model.")


