import math
from collections import defaultdict
from typing import Literal

from pydantic import ConfigDict

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.tokenizers._mi_prune import compute_mi_table, select_drop_batch
from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.mingram.model import DEFAULT_SCORE_DELTA, MinGramModel
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array


class MinGramTrainerConfig(TrainerConfig):
    overshoot_factor: float = 1.1
    num_em_iterations: int = 2
    pruning_shrinking_factor: float = 0.0
    # Path score = score_delta * sum(log_prob) - num_tokens. Default (1/100_000) == historical
    # behaviour; 0 == pure min-token + longest-token tiebreak (PathPiece-equivalent).
    score_delta: float = DEFAULT_SCORE_DELTA
    # "usage_count" (default): prune the least-used tokens (sort by -log_prob == expected count).
    # "mi": careful pruning -- drop the tokens with the smallest corpus-token-count increase
    # (Minimum Increase, Schmidt et al. 2024), recomputed each prune round; log_prob kept as
    # tie-break. Pairs best with iterative pruning (pruning_shrinking_factor > 0).
    prune_criterion: Literal["usage_count", "mi"] = "usage_count"
    # skip_substring (mi prune_criterion only, i.e. PathPiece-style MI pruning; Craig Schmidt,
    # pers. comm.): within a prune batch, skip a candidate whose atomic sequence is a contiguous
    # substring of an already-selected token -- avoids dropping too many overlapping tokens at
    # once (limited effect in practice; default off).
    skip_substring_in_batch: bool = False
    model_config = ConfigDict(extra="forbid")


class MinGramTrainer(BaseTrainer):
    MAX_ITERATIONS = 100

    def __init__(self, pretokenizer: Pretokenizer, corpus: PretokenizedCorpus, config: MinGramTrainerConfig):
        super().__init__(pretokenizer, corpus, config)

    def train(self) -> MinGramModel:
        cfg = self.config
        final_vocab_size = len(self.pretokenizer.atomic_tokens) + cfg.additional_vocab_size
        total_pretokens = sum(freq for _, freq in self.corpus)
        corpus_atomic_length = sum(len(atomic_token_seq) * freq for atomic_token_seq, freq in self.corpus)

        initial_tokens = self._build_bpe_init_tokens()
        model = MinGramModel(self.pretokenizer, initial_tokens, score_delta=cfg.score_delta)
        self.logger.info(f"Initialized MinGram from BPE with {len(model.tokens):,} tokens")

        totals_removed = defaultdict(list)
        objective = 0.0
        total_tokens = 0
        iter_idx = 0
        for iter_idx in range(self.MAX_ITERATIONS):
            for sub in range(cfg.num_em_iterations):
                self.logger.info(f"EM {iter_idx + 1}.{sub + 1}. Model size {len(model.tokens):,}")
                expected_count, objective, total_tokens = self.run_e_step(model, corpus_atomic_length)
                model, removed = self.run_m_step(model, expected_count)
                totals_removed["M Step Low Count"].append(removed)

            if len(model.tokens) <= final_vocab_size:
                self.logger.info("Target vocabulary size reached during EM")
                break

            model, num_pruned = self.prune_tokens(model, final_vocab_size)
            totals_removed["Prune/Score"].append(num_pruned)

        model, finalize_removed = self.score_based_prune(model, final_vocab_size)
        totals_removed["Finalize"].append(finalize_removed)

        expected_count, objective, total_tokens = self.run_e_step(model, corpus_atomic_length)
        model.metadata = {
            "tokenizer_variant": "mingram",
            "objective": objective,
            "total_tokens": total_tokens,
            "tokens/pretoken": total_tokens / total_pretokens,
            "num_iterations": iter_idx + 1,
            "totals_removed": totals_removed,
            "config": self.config.model_dump(),
        }
        self.logger.info(f"MinGram training completed. Avg tokens/pretoken: {model.metadata['tokens/pretoken']:.4f}")
        return model

    def _build_bpe_init_tokens(self) -> list[UnigramToken]:
        bpe_additional_size = int(self.config.additional_vocab_size * self.config.overshoot_factor)
        bpe_cfg = BPETrainerConfig(
            additional_vocab_size=bpe_additional_size,
            num_workers=self.config.num_workers,
            verbose=self.config.verbose,
        )
        self.logger.info(
            f"Training BPE init model with additional vocab {bpe_additional_size:,} "
            f"(overshoot factor={self.config.overshoot_factor})"
        )
        bpe_model = BPETrainer(self.pretokenizer, self.corpus, bpe_cfg).train()

        total_count = sum(max(1, t.current_count) for t in bpe_model.tokens.values())
        return [
            UnigramToken(
                id=bpe_token.id,
                atomic_tokens=token_array(bpe_token.atomic_tokens),
                log_prob=math.log(max(1, bpe_token.current_count) / total_count),
                required=len(bpe_token.atomic_tokens) == 1,
            )
            for bpe_token in bpe_model.tokens.values()
        ]

    def run_e_step(self, model: MinGramModel, corpus_atomic_length: int) -> tuple[dict[int, float], float, int]:
        """Hard EM E-step: minimum-token-count segmentation."""
        expected_count = defaultdict(float)
        objective = 0.0
        total_tokens = 0
        for atomic_token_seq, freq in self.corpus:
            path = model.encode_chunk(atomic_token_seq)
            path_logprob = sum(token.log_prob for token in path)
            assert not math.isnan(path_logprob), f"NaN path log-prob for pretoken {atomic_token_seq} with freq={freq}"
            for token in path:
                expected_count[token.id] += freq
            total_tokens += len(path) * freq
            objective -= path_logprob * freq
        objective /= corpus_atomic_length
        return expected_count, objective, total_tokens

    def run_m_step(self, model: MinGramModel, expected_count: dict[int, float]) -> tuple[MinGramModel, int]:
        filtered_tokens = [t for t in model.tokens.values() if expected_count[t.id] > 0 or t.required]
        num_removed = len(model.tokens) - len(filtered_tokens)
        if num_removed > 0:
            model = MinGramModel(self.pretokenizer, filtered_tokens, score_delta=self.config.score_delta)

        total_freq = sum(expected_count[t.id] for t in model.tokens.values())
        for t in model.tokens.values():
            count = expected_count[t.id]
            t.log_prob = math.log(count / total_freq) if count > 0 else float("-inf")
        return model, num_removed

    def prune_tokens(self, model: MinGramModel, desired_vocab_size: int) -> tuple[MinGramModel, int]:
        num_non_atomic_tokens = len(model.tokens) - len(self.pretokenizer.atomic_tokens)
        shrink_n = int(num_non_atomic_tokens * (1 - self.config.pruning_shrinking_factor))
        target_size = max(desired_vocab_size, num_non_atomic_tokens - shrink_n)
        return self.score_based_prune(model, target_size)

    def score_based_prune(self, model: MinGramModel, target_size: int) -> tuple[MinGramModel, int]:
        if self.config.prune_criterion == "mi":
            return self._mi_based_prune(model, target_size)
        kept_tokens: dict[int, UnigramToken] = {token.id: token for token in model.tokens.values() if token.required}
        for token in sorted(model.tokens.values(), key=lambda x: -x.log_prob):
            if token.id in kept_tokens:
                continue
            if len(kept_tokens) >= target_size:
                break
            kept_tokens[token.id] = token
        removed = len(model.tokens) - len(kept_tokens)
        return MinGramModel(model.pretokenizer, list(kept_tokens.values()), score_delta=self.config.score_delta), removed

    def _mi_based_prune(self, model: MinGramModel, target_size: int) -> tuple[MinGramModel, int]:
        """MinGram-PP prune: drop the non-required tokens whose removal adds the
        fewest extra tokens to the corpus's min-token segmentation, recomputed each call. Ties
        broken by log_prob ascending (least-likely removed first). Only the removal rule differs
        from the usage-count prune -- the EM/log_prob score is retained."""
        required_ids = {t.id for t in model.tokens.values() if t.required}
        n_removable = len(model.tokens) - len(required_ids)
        n_remove = min(max(0, len(model.tokens) - target_size), n_removable)
        if n_remove == 0:
            return model, 0
        max_width = max((len(t.atomic_tokens) for t in model.tokens.values()), default=1)
        mi_by_id, _ = compute_mi_table(model, self.corpus, max_width, logger=self.logger)
        candidates = [t for t in model.tokens.values() if t.id not in required_ids]
        candidates.sort(key=lambda t: (mi_by_id.get(t.id, 0.0), t.log_prob))
        ordered_ids = [t.id for t in candidates]
        to_remove = select_drop_batch(
            ordered_ids, n_remove, model.tokens, self.config.skip_substring_in_batch
        )
        kept = [t for t in model.tokens.values() if t.id not in to_remove]
        return MinGramModel(self.pretokenizer, kept, score_delta=self.config.score_delta), len(to_remove)
