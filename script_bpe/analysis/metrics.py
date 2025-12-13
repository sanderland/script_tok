"""Corpus evaluation metrics for tokenizers."""

import math

from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.bpe import BPETokenizer


def evaluate_on_corpus(model, corpus_name: str) -> dict:
    """
    Evaluate model on a corpus.

    For Unigram models: compute objective using lattice marginal (like training)
    For BPE models: compute compression metrics only (no probabilistic objective)

    Args:
        model: A tokenizer model (UnigramModel or BPETokenizer)
        corpus_name: Name of corpus to evaluate on (loaded via registry)

    Returns:
        dict with keys:
        - objective: float or None (Unigram only)
        - tokens: int (total token count)
        - atomic_tokens: int (total atomic token count)
        - bytes: int (total byte count)
    """
    is_bpe = isinstance(model, BPETokenizer)
    corpus = load_corpus_by_name(corpus_name, model.pretokenizer)

    objective = 0.0
    total_tokens = 0
    total_atomic_len = 0
    total_bytes = 0

    for atomic_tokens, count in corpus:
        text = model.pretokenizer.decode(atomic_tokens)
        chunk_bytes = len(text.encode("utf-8"))

        if is_bpe:
            token_ids = model.encode(text)
            total_tokens += len(token_ids) * count
        else:
            # Unigram: use lattice for objective
            lattice = model.make_lattice(atomic_tokens)
            z, _ = lattice.calc_marginal()
            assert not math.isnan(z), f"NaN likelihood for pretoken {atomic_tokens}"
            objective -= z * count
            # Use Viterbi path length for token count
            viterbi_path, _ = lattice.viterbi()
            total_tokens += len(viterbi_path) * count

        total_atomic_len += len(atomic_tokens) * count
        total_bytes += chunk_bytes * count

    if total_atomic_len > 0 and not is_bpe:
        objective /= total_atomic_len

    return {
        "objective": objective if not is_bpe else None,
        "tokens": total_tokens,
        "atomic_tokens": total_atomic_len,
        "bytes": total_bytes,
    }

