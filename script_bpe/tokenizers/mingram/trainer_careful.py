"""EXPERIMENTAL: MinGram with PathPiece-style *careful* (Minimum-Increase) pruning.

Kept deliberately separate from ``trainer.py`` for easy reverting -- default MinGram
is untouched; delete this file to revert.

Stock MinGram prunes by **usage count** (``score_based_prune`` keeps the highest
``log_prob`` tokens, and ``log_prob`` is proportional to the EM expected count). That
asks "is this token rare?" but not "is it replaceable?". This trainer swaps in the
exact **corpus-token-count increase (MI)** criterion from Schmidt et al. (2024) -- the
same ``compute_mi_table`` PathPiece uses -- which measures how many *extra* tokens the
corpus would need if each token were removed, accounting for the fall-back
segmentation. ``log_prob`` is retained as the tie-break among equal-MI tokens (and for
the EM objective / reporting), so the model still "keeps a score": this is PathPiece's
careful prune, but MinGram keeps its likelihood.

Only ``score_based_prune`` is overridden; ``prune_tokens`` (the per-iteration shrink)
and ``train`` are inherited unchanged and route through it, so both the in-loop prune
and the finalize prune become MI-based.
"""

from script_bpe.tokenizers._mi_prune import compute_mi_table
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.mingram.trainer import MinGramTrainer, MinGramTrainerConfig


class CarefulMinGramTrainerConfig(MinGramTrainerConfig):
    """Same knobs as MinGram; distinct class so the variant is explicit in metadata."""


class CarefulMinGramTrainer(MinGramTrainer):
    """MinGram whose pruning uses the MI (token-count-increase) criterion."""

    @staticmethod
    def _model_max_width(model: MinGramModel) -> int:
        return max((len(t.atomic_tokens) for t in model.tokens.values()), default=1)

    def score_based_prune(self, model: MinGramModel, target_size: int) -> tuple[MinGramModel, int]:
        """Drop the non-required tokens with the smallest corpus-token-count increase.

        MI(t) = extra tokens the corpus needs without t (0 for never-segmented tokens,
        which are therefore pruned first -- matching the stock M-step's dead-token drop).
        Ties broken by ``log_prob`` ascending (least-likely removed first), so the kept
        vocabulary still respects the model's score.
        """
        required_ids = {t.id for t in model.tokens.values() if t.required}
        n_removable = len(model.tokens) - len(required_ids)
        n_remove = min(max(0, len(model.tokens) - target_size), n_removable)
        if n_remove == 0:
            return model, 0

        L = self._model_max_width(model)
        mi_by_id, _ = compute_mi_table(model, self.corpus, L, logger=self.logger)

        candidates = [t for t in model.tokens.values() if t.id not in required_ids]
        # smallest MI first; among equal MI, smallest log_prob first
        candidates.sort(key=lambda t: (mi_by_id.get(t.id, 0.0), t.log_prob))
        to_remove = {t.id for t in candidates[:n_remove]}

        kept = [t for t in model.tokens.values() if t.id not in to_remove]
        return MinGramModel(self.pretokenizer, kept), len(to_remove)
