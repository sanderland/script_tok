#!/usr/bin/env python3
"""Train unigram models with supertoken initialization.

Supertokens are n-gram sequences extracted from a pre-trained tokenizer,
enabling transfer of tokenization patterns across corpora.

Flow:
1. Train initial model on corpus A with vocab size VA (word-level pretokenization)
2. For corpus B:
   a. Pretokenize into word-level chunks
   b. Tokenize each chunk with model A
   c. Find consecutive spans of single-token pretokens
   d. Extract n-grams from those spans
3. Filter and score supertokens
4. Train final model on LINE-pretokenized corpus B (supertokens can span words)
"""

from collections import Counter
from pathlib import Path

from cyclopts import App

from script_bpe import get_pretokenizer
from script_bpe.corpus import load_corpus_by_name
from script_bpe.tokenizers.unigram import UnigramModel
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from script_bpe.train import train_tokenizer
from script_bpe.utils import create_logger

from paper_utils.super.utils import (
    PRETOKENIZER_NAME,
    LINE_PRETOKENIZER_NAME,
    RESULTS_DIR,
    SUPERTOKEN_FILTERS,
)

logger = create_logger("supertoken", verbose=True)


def is_supertoken(pretokenizer, atomic_tokens: tuple[int, ...]) -> bool:
    """A supertoken spans more than 1 pretoken when decoded and re-pretokenized."""
    text = pretokenizer.decode(atomic_tokens)
    pretokens = pretokenizer.pretokenize(text)
    return len(pretokens) > 1


def get_model_save_path(
    corpus_a: str,
    corpus_b: str,
    vocab_a: int,
    vocab_b: int,
    max_ngram: int,
    filter_name: str,
    pretokenizer_name: str,
    fsp: bool,
) -> Path:
    """Get the save path for a supertoken model."""
    fsp_suffix = "_fsp" if fsp else ""
    filename = f"super_va{vocab_a}_n{max_ngram}_{filter_name}{fsp_suffix}.json.gz"
    if corpus_a == corpus_b:
        return RESULTS_DIR / corpus_b / f"vb{vocab_b}" / pretokenizer_name / filename
    else:
        return RESULTS_DIR / f"{corpus_a}_to_{corpus_b}" / f"vb{vocab_b}" / pretokenizer_name / filename


def extract_ngrams_from_span(
    span_tokens: list,
    ngram_counts: Counter,
    freq: int,
    max_ngram: int,
) -> None:
    """Extract n-grams from a span of single-token pretokens.

    Args:
        span_tokens: List of UnigramToken objects (single-token pretokens)
        ngram_counts: Counter to update with n-gram counts
        freq: Frequency to add for each n-gram
        max_ngram: Maximum n-gram size
    """
    if len(span_tokens) < 2:
        return

    token_ids = [t.id for t in span_tokens]
    for n in range(2, min(max_ngram + 1, len(token_ids) + 1)):
        for i in range(len(token_ids) - n + 1):
            ngram = tuple(token_ids[i : i + n])
            ngram_counts[ngram] += freq


def extract_ngrams_from_corpus(
    model: UnigramModel,
    corpus_name: str,
    max_ngram: int,
) -> tuple[Counter[tuple[int, ...]], Counter[tuple[int, ...]]]:
    """Extract n-grams from spans of single-token pretokens.

    For each LINE in the corpus:
    1. Decode line back to text
    2. Re-pretokenize with WORD pretokenizer to get word chunks
    3. Tokenize each chunk with model
    4. Find chunks that tokenize to exactly 1 token (single-token pretokens)
    5. Find consecutive spans of these single-token pretokens
    6. Extract n-grams from those spans

    Returns:
        unigram_counts: Counter of single token occurrences (token_id,) -> count
        ngram_counts: Counter of n-gram token sequences (token_id, ...) -> count
    """
    word_pretokenizer = model.pretokenizer
    # Load corpus with LINE pretokenizer so each entry is a full line (not single words)
    line_pretokenizer = get_pretokenizer(LINE_PRETOKENIZER_NAME)
    corpus = load_corpus_by_name(corpus_name, line_pretokenizer)

    unigram_counts: Counter[tuple[int, ...]] = Counter()
    ngram_counts: Counter[tuple[int, ...]] = Counter()

    logger.info(f"Extracting n-grams (n <= {max_ngram}) from single-token pretoken spans...")

    total_chunks = 0
    single_token_chunks = 0

    for pretoken_seq, freq in corpus:
        # Decode line to text, then re-pretokenize with WORD pretokenizer to get word chunks
        text = line_pretokenizer.decode(pretoken_seq)
        chunks = word_pretokenizer.pretokenize(text)

        # Tokenize each chunk, find single-token chunks
        chunk_results: list[object | None] = []
        for chunk in chunks:
            total_chunks += freq
            lattice = model.make_lattice(chunk)
            path, _ = lattice.viterbi()
            if len(path) == 1:
                # Single-token pretoken!
                single_token_chunks += freq
                chunk_results.append(path[0])
                unigram_counts[(path[0].id,)] += freq
            else:
                # Multi-token pretoken - acts as a break point
                chunk_results.append(None)

        # Find consecutive spans of single-token pretokens and extract n-grams
        span_tokens: list = []
        for result in chunk_results:
            if result is None:
                # End of span - extract n-grams
                extract_ngrams_from_span(span_tokens, ngram_counts, freq, max_ngram)
                span_tokens = []
            else:
                span_tokens.append(result)
        # Handle final span
        extract_ngrams_from_span(span_tokens, ngram_counts, freq, max_ngram)

    logger.info(f"Processed {total_chunks:,} chunks, {single_token_chunks:,} ({100*single_token_chunks/total_chunks:.1f}%) are single-token")
    logger.info(f"Extracted {len(unigram_counts):,} unigrams and {len(ngram_counts):,} n-grams")
    return unigram_counts, ngram_counts


def build_supertoken_vocab(
    model: UnigramModel,
    unigram_counts: Counter[tuple[int, ...]],
    ngram_counts: Counter[tuple[int, ...]],
    filter_name: str,
    initial_vocab_size: int,
) -> list:
    """Build initial vocabulary from n-grams with filtering and scoring.

    Args:
        model: The trained model (used to convert token IDs to atomic sequences)
        unigram_counts: Counter of unigram occurrences
        ngram_counts: Counter of n-gram occurrences
        filter_name: Name of the filter to apply
        initial_vocab_size: Target size for initial vocabulary

    Returns:
        List of UnigramToken for forced_initial_vocab
    """
    pretokenizer = model.pretokenizer
    filter_fn = SUPERTOKEN_FILTERS[filter_name]

    # Score n-grams: count * n (longer n-grams get higher scores)
    scores: Counter[tuple[int, ...]] = Counter()
    rejected_count = 0

    for ngram_ids, count in ngram_counts.items():
        # Convert token IDs to atomic token sequence
        seq = tuple(at for tid in ngram_ids for at in model.tokens_by_id[tid].atomic_tokens)

        if filter_fn(pretokenizer, seq):
            n = len(ngram_ids)
            scores[seq] += count * n
        else:
            rejected_count += 1

    logger.info(f"Filter '{filter_name}': accepted {len(scores):,}, rejected {rejected_count:,} n-grams")

    # Log top allowed supertokens
    top_allowed = sorted(scores.items(), key=lambda x: -x[1])[:10]
    logger.info("Top 10 allowed supertokens:")
    for seq, score in top_allowed:
        logger.info(f"  {pretokenizer.tokens_repr(seq)}: {score:,}")

    # Add base tokens from the model (unigrams with their counts)
    for token in model.tokens.values():
        seq = tuple(token.atomic_tokens)
        # Use count from corpus, minimum 1
        scores[seq] += max(1, unigram_counts.get((token.id,), 0))

    # Normalize scores and create initial vocab
    all_tokens = [(score, seq) for seq, score in scores.items()]
    initial_vocab = UnigramTrainer.init_vocab_normalize_scores(all_tokens, initial_vocab_size)

    logger.info(f"Created initial vocab with {len(initial_vocab):,} tokens")
    return initial_vocab


def train_supertoken_model(
    corpus_a: str,
    corpus_b: str,
    vocab_a: int,
    vocab_b: int,
    max_ngram: int = 4,
    filter_name: str = "all",
    pretokenizer_name: str = PRETOKENIZER_NAME,
    fsp: bool = False,
    retrain: bool = False,
) -> UnigramModel:
    """Train a unigram model with supertoken initialization.

    Args:
        corpus_a: Corpus to train initial model on
        corpus_b: Corpus for n-gram extraction and final training
        vocab_a: Vocabulary size for initial model
        vocab_b: Vocabulary size for final model
        max_ngram: Maximum n-gram size to extract
        filter_name: Name of supertoken filter to apply
        pretokenizer_name: Name of pretokenizer to use
        fsp: Use flat-score pruning
        retrain: Force retraining even if model exists

    Returns:
        Trained UnigramModel with supertokens
    """
    save_path = get_model_save_path(
        corpus_a, corpus_b, vocab_a, vocab_b, max_ngram, filter_name, pretokenizer_name, fsp
    )

    # Try to load existing model
    if not retrain and save_path.exists():
        logger.info(f"Loading existing model from {save_path}")
        return UnigramModel.load(str(save_path))

    logger.info(f"Training supertoken model: {corpus_a} -> {corpus_b}")
    logger.info(f"  vocab_a={vocab_a}, vocab_b={vocab_b}, max_ngram={max_ngram}")
    logger.info(f"  filter={filter_name}, fsp={fsp}")

    # Step 1: Train or load initial model on corpus A
    logger.info(f"Step 1: Getting initial model (vocab={vocab_a}) on {corpus_a}...")
    initial_model = train_tokenizer(
        pretokenizer_name=pretokenizer_name,
        model_name="unigram",
        corpus_name=corpus_a,
        additional_vocab_size=vocab_a,
        n_cpus=4,
        retrain=False,
    )
    assert initial_model is not None, f"Failed to train initial model on {corpus_a}"

    # Step 2: Extract n-grams from corpus B
    logger.info(f"Step 2: Extracting n-grams from {corpus_b}...")
    unigram_counts, ngram_counts = extract_ngrams_from_corpus(initial_model, corpus_b, max_ngram)

    # Step 3: Build supertoken vocabulary
    logger.info("Step 3: Building supertoken vocabulary...")
    # Use 3x the final vocab size for initial vocab (will be pruned down)
    initial_vocab_size = vocab_b * 3
    initial_vocab = build_supertoken_vocab(
        initial_model, unigram_counts, ngram_counts, filter_name, initial_vocab_size
    )

    # Step 4: Train final model on LINE-pretokenized corpus B
    # Using line-regex pretokenizer allows supertokens to span word boundaries
    logger.info(f"Step 4: Training final model (vocab={vocab_b}) on {corpus_b} with line pretokenizer...")
    line_pretokenizer = get_pretokenizer(LINE_PRETOKENIZER_NAME)
    line_corpus = load_corpus_by_name(corpus_b, line_pretokenizer)

    config = UnigramTrainerConfig(
        additional_vocab_size=vocab_b,
        forced_initial_vocab=initial_vocab,
        flat_score_prune=fsp,
        pre_final_vocab_factor=1.0 if fsp else 1.1,
    )

    trainer = UnigramTrainer(line_pretokenizer, line_corpus, config)
    model = trainer.train()

    # Add metadata
    model.metadata = model.metadata or {}
    model.metadata["supertoken_config"] = {
        "corpus_a": corpus_a,
        "corpus_b": corpus_b,
        "vocab_a": vocab_a,
        "vocab_b": vocab_b,
        "max_ngram": max_ngram,
        "filter_name": filter_name,
        "fsp": fsp,
        "word_pretokenizer": pretokenizer_name,
        "line_pretokenizer": LINE_PRETOKENIZER_NAME,
    }

    # Save model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info(f"Saved model to {save_path}")

    return model


def run_test_mode(
    corpus: str = "smol_eng_latn_300mb",
    vocab_a: int = 16000,
    vocab_b: int = 16000,
    max_ngram: int = 4,
    filter_name: str = "all",
    pretokenizer_name: str = PRETOKENIZER_NAME,
) -> None:
    """Run supertoken training with detailed analysis output."""
    word_pretokenizer = get_pretokenizer(pretokenizer_name)
    
    # ========== Step 1: Train/load initial model ==========
    print("\n" + "=" * 60)
    print("STEP 1: Initial Model")
    print("=" * 60)
    
    initial_model = train_tokenizer(
        pretokenizer_name=pretokenizer_name,
        model_name="unigram",
        corpus_name=corpus,
        additional_vocab_size=vocab_a,
        n_cpus=4,
        retrain=False,
    )
    assert initial_model is not None
    print(f"Initial model vocab size: {len(initial_model.tokens)}")
    
    # ========== Step 2: Extract n-grams ==========
    print("\n" + "=" * 60)
    print("STEP 2: N-gram Extraction")
    print("=" * 60)
    
    unigram_counts, ngram_counts = extract_ngrams_from_corpus(initial_model, corpus, max_ngram)
    
    print(f"\nUnigrams extracted: {len(unigram_counts):,}")
    print(f"N-grams extracted: {len(ngram_counts):,}")
    
    # Show top 20 n-grams
    print("\nTop 20 n-grams by count:")
    top_ngrams = ngram_counts.most_common(20)
    for i, (ngram_ids, count) in enumerate(top_ngrams, 1):
        seq = tuple(at for tid in ngram_ids for at in initial_model.tokens_by_id[tid].atomic_tokens)
        text = word_pretokenizer.decode(seq)
        text_repr = repr(text)
        print(f"  {i:2}. {text_repr:30} n={len(ngram_ids)}, count={count:,}")
    
    # ========== Step 3: Build initial vocab ==========
    print("\n" + "=" * 60)
    print("STEP 3: Initial Vocab Construction")
    print("=" * 60)
    
    initial_vocab_size = vocab_b * 3
    initial_vocab = build_supertoken_vocab(
        initial_model, unigram_counts, ngram_counts, filter_name, initial_vocab_size
    )
    
    # Analyze initial vocab
    supertokens_init = [t for t in initial_vocab if is_supertoken(word_pretokenizer, tuple(t.atomic_tokens))]
    regular_tokens = [t for t in initial_vocab if len(t.atomic_tokens) > 1 and not is_supertoken(word_pretokenizer, tuple(t.atomic_tokens))]
    base_tokens = [t for t in initial_vocab if len(t.atomic_tokens) == 1]
    
    print("\nInitial vocab composition:")
    print(f"  Total tokens: {len(initial_vocab):,}")
    print(f"  Base tokens (atomic_len=1): {len(base_tokens):,}")
    print(f"  Regular tokens (1 pretoken): {len(regular_tokens):,}")
    print(f"  Supertokens (>1 pretoken): {len(supertokens_init):,}")
    
    # Sort supertokens by log_prob
    supertokens_init_sorted = sorted(supertokens_init, key=lambda t: -t.log_prob)
    
    print("\nTop 50 supertokens in initial vocab (by log_prob):")
    print(f"  {'Rank':<5} {'Token':<40} {'#PT':<4} {'Log-prob':<10}")
    print(f"  {'-'*5} {'-'*40} {'-'*4} {'-'*10}")
    for i, token in enumerate(supertokens_init_sorted[:50], 1):
        text = word_pretokenizer.decode(token.atomic_tokens)
        n_pretokens = len(word_pretokenizer.pretokenize(text))
        text_repr = repr(text)[:38]
        print(f"  {i:<5} {text_repr:<40} {n_pretokens:<4} {token.log_prob:<10.4f}")
    
    # Track supertoken atomic sequences for later comparison
    supertoken_seqs = {tuple(t.atomic_tokens) for t in supertokens_init}
    supertoken_init_logprobs = {tuple(t.atomic_tokens): t.log_prob for t in supertokens_init}
    
    # ========== Step 4: Train final model ==========
    print("\n" + "=" * 60)
    print("STEP 4: Training Final Model")
    print("=" * 60)
    
    line_pretokenizer = get_pretokenizer(LINE_PRETOKENIZER_NAME)
    line_corpus = load_corpus_by_name(corpus, line_pretokenizer)
    
    config = UnigramTrainerConfig(
        additional_vocab_size=vocab_b,
        forced_initial_vocab=initial_vocab,
        flat_score_prune=False,
        pre_final_vocab_factor=1.1,
    )
    
    trainer = UnigramTrainer(line_pretokenizer, line_corpus, config)
    final_model = trainer.train()
    
    print(f"\nFinal model vocab size: {len(final_model.tokens):,}")
    
    # ========== Step 5: Analyze survival ==========
    print("\n" + "=" * 60)
    print("STEP 5: Supertoken Survival Analysis")
    print("=" * 60)
    
    # Find surviving supertokens (ignoring base tokens)
    surviving_supertokens = []
    non_supertokens = []
    
    for token in final_model.tokens.values():
        seq = tuple(token.atomic_tokens)
        if len(seq) <= 1:
            continue
        
        if seq in supertoken_seqs:
            surviving_supertokens.append((token, supertoken_init_logprobs[seq]))
        else:
            non_supertokens.append(token)
    
    print("\nSurvival stats (excluding base tokens):")
    print(f"  Supertokens added: {len(supertokens_init):,}")
    print(f"  Supertokens surviving: {len(surviving_supertokens):,}")
    print(f"  Non-supertokens in final: {len(non_supertokens):,}")
    
    if surviving_supertokens:
        surviving_sorted = sorted(surviving_supertokens, key=lambda x: -x[0].log_prob)
        
        print("\nTop 50 surviving supertokens (by final log_prob):")
        print(f"  {'Rank':<5} {'Token':<35} {'#PT':<4} {'Init LP':<9} {'Final LP':<9} {'Δ LP':<8}")
        print(f"  {'-'*5} {'-'*35} {'-'*4} {'-'*9} {'-'*9} {'-'*8}")
        
        for i, (token, init_lp) in enumerate(surviving_sorted[:50], 1):
            text = word_pretokenizer.decode(token.atomic_tokens)
            n_pretokens = len(word_pretokenizer.pretokenize(text))
            text_repr = repr(text)[:33]
            delta = token.log_prob - init_lp
            print(f"  {i:<5} {text_repr:<35} {n_pretokens:<4} {init_lp:<9.4f} {token.log_prob:<9.4f} {delta:+.4f}")
        
        # Top 10 longest supertokens
        longest_sorted = sorted(surviving_supertokens, key=lambda x: -len(word_pretokenizer.pretokenize(word_pretokenizer.decode(x[0].atomic_tokens))))
        print("\nTop 10 longest supertokens (by #pretokens):")
        print(f"  {'Rank':<5} {'Token':<50} {'#PT':<4} {'Final LP':<9}")
        print(f"  {'-'*5} {'-'*50} {'-'*4} {'-'*9}")
        for i, (token, _) in enumerate(longest_sorted[:10], 1):
            text = word_pretokenizer.decode(token.atomic_tokens)
            n_pretokens = len(word_pretokenizer.pretokenize(text))
            text_repr = repr(text)[:48]
            print(f"  {i:<5} {text_repr:<50} {n_pretokens:<4} {token.log_prob:<9.4f}")
        
        # Model metrics
        print("\nModel Metrics:")
        print(f"  Objective: {final_model.metadata.get('objective', 'N/A'):.4f}")
        print(f"  Total tokens: {final_model.metadata.get('total_tokens', 'N/A'):,}")
        print(f"  Tokens/pretoken: {final_model.metadata.get('tokens/pretoken', 'N/A'):.2f}")
    else:
        print("\n⚠️  NO SUPERTOKENS SURVIVED!")
        print("This suggests they were all pruned during EM training.")
        
        print("\nTop 20 non-base tokens in final model:")
        non_base = [(t, t.log_prob) for t in final_model.tokens.values() if len(t.atomic_tokens) > 1]
        non_base_sorted = sorted(non_base, key=lambda x: -x[1])[:20]
        
        for i, (token, lp) in enumerate(non_base_sorted, 1):
            text = line_pretokenizer.decode(token.atomic_tokens)
            text_repr = repr(text)[:40]
            print(f"  {i:2}. {text_repr:<42} len={len(token.atomic_tokens):<3} lp={lp:.4f}")


# ========== CLI ==========

app = App()


@app.default
def cli(
    corpus_a: str = "eng_latn_300mb",
    corpus_b: str | None = None,
    vocab_a: int = 64000,
    vocab_b: int = 32000,
    max_ngram: int = 4,
    filter_name: str = "all",
    pretokenizer: str = PRETOKENIZER_NAME,
    fsp: bool = False,
    retrain: bool = False,
    test: bool = False,
) -> None:
    """Train a supertoken model.

    Args:
        corpus_a: Corpus for training initial model
        corpus_b: Corpus for n-gram extraction and final training (defaults to corpus_a)
        vocab_a: Vocabulary size for initial model
        vocab_b: Vocabulary size for final model
        max_ngram: Maximum n-gram size to extract
        filter_name: Supertoken filter (all, words, words_nocomma, len_8c, len_16c)
        pretokenizer: Pretokenizer name
        fsp: Use flat-score pruning
        retrain: Force retraining
        test: Run test mode with detailed analysis on smol corpus
    """
    if test:
        run_test_mode(
            corpus="smol_eng_latn_300mb",
            vocab_a=16000,
            vocab_b=16000,
            max_ngram=max_ngram,
            filter_name=filter_name,
            pretokenizer_name=pretokenizer,
        )
        return

    if corpus_b is None:
        corpus_b = corpus_a

    model = train_supertoken_model(
        corpus_a=corpus_a,
        corpus_b=corpus_b,
        vocab_a=vocab_a,
        vocab_b=vocab_b,
        max_ngram=max_ngram,
        filter_name=filter_name,
        pretokenizer_name=pretokenizer,
        fsp=fsp,
        retrain=retrain,
    )

    # Print summary
    print("\nModel trained successfully!")
    print(f"  Vocab size: {len(model.tokens):,}")
    if model.metadata:
        print(f"  Objective: {model.metadata.get('objective', 'N/A')}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    app()

