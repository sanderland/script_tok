from __future__ import annotations
from typing import Sequence

import heapq
from collections import defaultdict

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.tokenizers.uniformgram.model import UniformGramModel, UniformGramToken
from script_bpe.tokenizers.unigram.init_algorithms import (
    compute_substring_frequencies_simple,
    compute_substring_frequencies_corpus,
)
from script_bpe.utils import token_array
from pydantic import ConfigDict


class UniformGramTrainerConfig(TrainerConfig):
    """Configuration for UniformGram training.

    UniformGram uses iterative Viterbi counting and pruning instead of EM.
    All tokens have uniform probability (log_prob = 0.0), so Viterbi naturally
    selects the segmentation with the fewest tokens (longest available tokens).
    """

    max_token_len: int = 32
    initial_vocab_factor: int = 10
    pruning_shrinking_factor: float = 0.75
    max_iterations: int = 100
    init_vocab_algo: str | tuple = (
        "corpus_fallback"  # one of {"simple", "corpus_long", "corpus_intermediate", "corpus_fallback"}
    )
    forced_initial_vocab: Sequence[UniformGramToken] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class UniformGramTrainer(BaseTrainer):
    """
    Trainer for UniformGram tokenizer.

    Unlike Unigram which uses EM to learn token probabilities, UniformGram
    uses a simpler iterative approach:

    1. Initialize large vocabulary (same as Unigram)
    2. Iteratively:
       a. Count token usage in Viterbi segmentations of corpus
       b. Prune tokens with low Viterbi counts (keep top-N)
       c. Repeat until target vocabulary size is reached

    With uniform probabilities, Viterbi always selects the segmentation with
    the fewest tokens, effectively greedily choosing the longest available tokens.
    """

    def __init__(self, pretokenizer: Pretokenizer, corpus: PretokenizedCorpus, config: UniformGramTrainerConfig):
        super().__init__(pretokenizer, corpus, config)

    def train(self) -> UniformGramModel:
        cfg = self.config
        vocab, initial_vocab_size = self.make_initial_vocab()
        total_pretokens = sum(freq for _, freq in self.corpus)
        final_vocab_size = len(self.pretokenizer.atomic_tokens) + cfg.additional_vocab_size
        totals_removed = defaultdict(list)

        model = UniformGramModel(self.pretokenizer, vocab)
        self.logger.info(f"🚀 Starting UniformGram training with {len(model.tokens):,} initial tokens")

        for iter in range(cfg.max_iterations):
            current_size = len(model.tokens)
            self.logger.info(f"🔄 Iteration {iter + 1}. Model size {current_size:,}")

            if current_size <= final_vocab_size:
                self.logger.info("🎯 Target vocabulary size reached")
                self.logger.debug(f"   ├─ Current: {current_size:,}")
                self.logger.debug(f"   └─ Target:  {final_vocab_size:,}")
                break

            # Count token usage via Viterbi
            token_count, total_tokens = self.viterbi_count_step(model)
            avg_tokens_per_pretoken = 1.0 * total_tokens / total_pretokens
            self.logger.debug(f"   ├─ Total tokens: {total_tokens:,d}")
            self.logger.debug(f"   └─ Avg tokens/pretoken: {avg_tokens_per_pretoken:.4f}")

            # Prune tokens by Viterbi count
            model, num_unused, num_pruned = self.prune_by_count(model, token_count, final_vocab_size)
            totals_removed["Unused (zero count)"].append(num_unused)
            totals_removed["Pruned (low count)"].append(num_pruned)

        # Final count for statistics
        token_count, total_tokens = self.viterbi_count_step(model)
        avg_tokens_per_pretoken = 1.0 * total_tokens / total_pretokens

        stats = {
            "total_tokens": total_tokens,
            "tokens/pretoken": avg_tokens_per_pretoken,
            "num_iterations": iter + 1,
            "totals_removed": totals_removed,
            "initial_vocab_size": initial_vocab_size,
        }

        self.logger.info(f"🎉 Training completed successfully! Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")
        self.logger.debug("  📊 Token Removal Statistics:")
        for key, value in totals_removed.items():
            self.logger.debug(
                f"   ├─ {key:<20} {sum(value):6,d} tokens" + (f" in steps {value}" if len(value) > 1 else "")
            )
        self.logger.debug("  📊 Compression Statistics:")
        self.logger.debug(f"   ├─ Total tokens: {stats['total_tokens']:,d}")
        self.logger.debug(f"   └─ Avg tokens/pretoken: {stats['tokens/pretoken']:.4f}")

        model.metadata = {
            **stats,
            "config": self.config.model_dump(),
        }
        return model

    # ---- Helper methods ----

    def log_examples(self, tokens_with_scores: list[tuple[UniformGramToken, float]], score_label="score", n=5):
        """Log example tokens with their scores."""
        tokens_with_scores.sort(key=lambda x: x[1])
        for i, (t, score) in enumerate(tokens_with_scores):
            if len(tokens_with_scores) > 2 * n:
                if i == n:
                    self.logger.debug("   │  ├─ ...")
                if n < i < len(tokens_with_scores) - n:
                    continue
            list_item = " ├─" if i < len(tokens_with_scores) - 1 else " └─"
            self.logger.debug(
                f"   │ {list_item} {repr(self.pretokenizer.decode(t.atomic_tokens, errors='backslashreplace')):25}  {score_label} = {score:10.3g}  atomic_tokens = {list(t.atomic_tokens)}"
            )

    @staticmethod
    def init_vocab_normalize_scores(all_tokens: list[tuple[float, tuple[int, ...]]], n) -> list[UniformGramToken]:
        """
        Select top-N tokens from candidates and create UniformGramTokens.

        Unlike Unigram, we don't normalize to probabilities - all tokens get log_prob = 0.0.
        """
        selected_tokens = heapq.nlargest(n, all_tokens, key=lambda item: (len(item[1]) == 1, item[0]))
        return [
            UniformGramToken(
                atomic_tokens=token_array(atomic_token_seq),
                id=i,
                log_prob=0.0,  # Uniform probability
                required=len(atomic_token_seq) == 1,
            )
            for i, (score, atomic_token_seq) in enumerate(selected_tokens)
        ]

    def make_initial_vocab(self) -> tuple[list[UniformGramToken], int]:
        """
        Create initial vocabulary using the same algorithms as Unigram.

        Returns:
            Tuple of (token list, number of candidates considered)
        """
        if self.config.forced_initial_vocab:
            self.logger.info(f"🌱 Using {len(self.config.forced_initial_vocab)} forced initial tokens.")
            return list(self.config.forced_initial_vocab), len(self.config.forced_initial_vocab)

        additional_num_tokens = self.config.additional_vocab_size * self.config.initial_vocab_factor
        max_token_length = self.config.max_token_len
        atomic_tokens = {(t,) for t in self.pretokenizer.atomic_tokens}

        algo = self.config.init_vocab_algo
        if isinstance(algo, tuple):  # Handle special case from Unigram
            algo, nocb_corpus = algo
            self.logger.info(f"🔍 Creating initial vocabulary using {algo} algorithm, no pretokenizer")
            self.config.init_vocab_algo = algo
            algo = algo.removesuffix("_no_pt")
            flat_corpus = [(tuple(seq), freq) for seq, freq in nocb_corpus]
        else:
            flat_corpus = [(tuple(seq), freq) for seq, freq in self.corpus]

        self.logger.info(
            f"🔍 Creating initial vocabulary using {algo} algorithm, max token length {max_token_length}, additional tokens {additional_num_tokens}"
        )

        if algo == "simple":
            substring_freq = compute_substring_frequencies_simple(self.pretokenizer, flat_corpus, max_token_length)
        elif algo in ["corpus_long", "corpus_intermediate", "corpus_fallback"]:
            substring_freq = compute_substring_frequencies_corpus(
                self.pretokenizer,
                flat_corpus,
                max_token_length,
                strategy=algo.removeprefix("corpus_"),
            )
            self.logger.info(f"🔍 Corpus based init vocab {algo!r}. Number of tokens: {len(substring_freq):,} tokens")
        else:
            raise ValueError(
                f"Unknown init_vocab_algo: {self.config.init_vocab_algo}. Use one of 'simple', 'corpus_long', 'corpus_intermediate', 'corpus_fallback'."
            )

        # Ensure atomic tokens are present (with at least frequency 0)
        for t in atomic_tokens:
            if t not in substring_freq:
                substring_freq[t] = 0

        # Score by frequency * length (same heuristic as Unigram)
        all_tokens = [(max(freq, 1) * len(token), token) for token, freq in substring_freq.items()]
        tokens = self.init_vocab_normalize_scores(all_tokens, len(atomic_tokens) + additional_num_tokens)

        self.logger.info(
            f"🌱 Selected {len(self.pretokenizer.atomic_tokens):,} + {self.config.additional_vocab_size * self.config.initial_vocab_factor:,} = {len(tokens):,} initial tokens from {len(all_tokens):,} candidates"
        )
        self.logger.debug(f"   ├─ Max length: {max_token_length}")
        self.logger.debug(f"   └─ Source: {self.corpus.name}: {self.corpus.metadata}")
        return tokens, len(all_tokens)

    def viterbi_count_step(self, model: UniformGramModel) -> tuple[dict[int, float], int]:
        """
        Count token usage in Viterbi segmentations of the corpus.

        Returns:
            Tuple of (token_count dict, total_tokens)
        """
        token_count = {t.id: 0.0 for t in model.tokens.values()}
        total_tokens = 0

        for atomic_token_seq, freq in self.corpus:
            lattice = model.make_lattice(atomic_token_seq)
            viterbi_path, _ = lattice.viterbi()
            for token in viterbi_path:
                token_count[token.id] += freq
            total_tokens += len(viterbi_path) * freq

        return token_count, total_tokens

    def prune_by_count(
        self, model: UniformGramModel, token_count: dict[int, float], final_vocab_size: int
    ) -> tuple[UniformGramModel, int, int]:
        """
        Prune tokens based on Viterbi counts.

        Strategy:
        1. Keep all required (atomic) tokens
        2. Remove tokens with zero count (unused)
        3. Shrink remaining tokens toward final_vocab_size using shrinking_factor
        4. Keep top-N tokens by count

        Args:
            model: Current model
            token_count: Token usage counts from Viterbi
            final_vocab_size: Target vocabulary size

        Returns:
            Tuple of (new_model, num_unused, num_pruned)
        """
        num_non_atomic_tokens = len(model.tokens) - len(self.pretokenizer.atomic_tokens)
        shrink_n = int(num_non_atomic_tokens * (1 - self.config.pruning_shrinking_factor))
        target_size = max(final_vocab_size, num_non_atomic_tokens - shrink_n)

        # Separate tokens by category
        required_tokens = []
        unused_tokens = []
        candidate_tokens = []

        for token in model.tokens.values():
            if token.required:
                required_tokens.append(token)
            elif token_count[token.id] == 0:
                unused_tokens.append((token, 0.0))
            else:
                candidate_tokens.append((token, token_count[token.id]))

        # Sort candidates by count (descending)
        candidate_tokens.sort(key=lambda x: -x[1])

        # Keep top-N candidates
        num_to_keep = target_size - len(required_tokens)
        kept_tokens = required_tokens + [t for t, _ in candidate_tokens[:num_to_keep]]
        pruned_tokens = candidate_tokens[num_to_keep:]

        self.logger.info(
            f"✂️  Pruning vocabulary from {len(model.tokens):,} to target {target_size:,} -> new vocab size {len(kept_tokens):,}"
        )
        self.logger.debug(
            f"   ├─ Target size: {target_size:,} based on shrinking factor {self.config.pruning_shrinking_factor} * non base tokens {num_non_atomic_tokens:,} and final vocab size {final_vocab_size}"
        )

        if unused_tokens:
            self.logger.debug(f"   ├─ Dropped {len(unused_tokens):,} tokens not in any optimal path")
            self.log_examples(unused_tokens, "count")

        self.logger.debug(f"   ├─ Kept {len(required_tokens):,} required tokens")

        if pruned_tokens:
            self.logger.debug(f"   └─ Pruned {len(pruned_tokens):,} tokens from {len(candidate_tokens):,} candidates")
            self.log_examples(pruned_tokens, "count")

        new_model = UniformGramModel(model.pretokenizer, kept_tokens)
        return new_model, len(unused_tokens), len(pruned_tokens)
