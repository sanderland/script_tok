"""N-gram language-model evaluation of tokenizers.

A CPU-only, minutes-not-GPU-hours stand-in for pretraining a model to compare tokenizers:
fit an interpolated modified Kneser-Ney model over each tokenizer's own token stream and
report held-out bits per true UTF-8 byte. See `evaluate.py` for what the number means and
where it stops being trustworthy.
"""

from script_bpe.ngram.counts import OrderTable, build_stream, gram_ids
from script_bpe.ngram.evaluate import NgramResult, VocabGeometry, evaluate_ngram_bpb
from script_bpe.ngram.kn import KneserNeyLM, estimate_discounts
from script_bpe.ngram.text import iter_documents, take_split

__all__ = [
    "OrderTable", "build_stream", "gram_ids",
    "KneserNeyLM", "estimate_discounts",
    "NgramResult", "VocabGeometry", "evaluate_ngram_bpb",
    "iter_documents", "take_split",
]
