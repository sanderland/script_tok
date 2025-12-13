"""UniformGram tokenizer module.

UniformGram is a tokenizer similar to Unigram but with uniform probabilities.
Instead of using EM to learn token probabilities, it uses iterative Viterbi
counting and pruning to select the most frequently used tokens.

With uniform probabilities (all log_prob = 0.0), Viterbi decoding naturally
selects the segmentation with the fewest tokens, effectively greedily choosing
the longest available tokens.
"""

from script_bpe.tokenizers.uniformgram.model import UniformGramModel, UniformGramToken
from script_bpe.tokenizers.uniformgram.trainer import UniformGramTrainer, UniformGramTrainerConfig

__all__ = [
    "UniformGramModel",
    "UniformGramToken",
    "UniformGramTrainer",
    "UniformGramTrainerConfig",
]
