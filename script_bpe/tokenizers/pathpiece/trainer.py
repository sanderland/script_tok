"""PathPiece vocabulary construction trainer.

Top-down "Minimum Increase" (MI) pruning, following §3.2 and Eqs. (1)-(5)
of Schmidt et al. (2024). At every iteration we:

  1. Run Algorithm 1's forward DP per pretokenized chunk to produce
     ``pl[i]`` (min tokens to reach position i) and the chosen-token
     widths/identities, plus a list of all vocabulary matches ending at
     each byte position (``tokens_ending_at``).
  2. Run the algorithm backwards to produce ``bpl[i]`` (min tokens from
     position i to the end of the chunk).
  3. For each occurrence of a (non-required) token in the chosen
     segmentation, compute its Minimum Increase MI using:
       - Case 1 (break inside the token at split j): pl[j] + bpl[j].
       - Case 2 (replace with a strict superset token at [s', e')):
         pl[s'] + bpl[e'] + 1.
     Aggregate ``MI(t) += freq(chunk) * min(MI_break, MI_superset)``.
  4. Drop the batch of non-required tokens with the smallest MI.

Atomic (single-base-token) tokens are forced to remain in the
vocabulary, matching the paper's requirement that "all single-byte
tokens are included in V".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Literal

from pydantic import ConfigDict

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import Pretokenizer
from script_bpe.tokenizers._mi_prune import compute_mi_table, select_drop_batch
from script_bpe.tokenizers.base import BaseTrainer, TrainerConfig
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.tokenizers.pathpiece.model import PathPieceModel
from script_bpe.tokenizers.unigram.model import UnigramToken
from script_bpe.utils import token_array


class PathPieceTrainerConfig(TrainerConfig):
    """Hyperparameters for PathPiece vocabulary construction."""

    init: Literal["ngram", "bpe"] = "ngram"
    init_vocab_size: int = 262144  # 2^18, matching the paper
    max_token_width: int = 16
    prune_batch_fraction: float = 0.10
    # ``ngram_init_rank`` controls how candidate n-grams are scored
    # when building the n-gram seed. ``count`` is the canonical
    # PathPiece-N (paper-faithful: top-K by raw frequency). ``count_width``
    # is a non-canonical variant that scores each n-gram by
    # ``count * (width - 1)`` -- a proxy for compression value (each
    # occurrence of a width-w token saves w-1 single-character tokens).
    ngram_init_rank: Literal["count", "count_width"] = "count"
    # Craig's rule: within a prune batch, skip a candidate whose atomic sequence is a
    # contiguous substring of an already-selected token -- avoids dropping too many
    # overlapping tokens at once (default off = paper behaviour).
    skip_substring_in_batch: bool = False

    model_config = ConfigDict(extra="forbid")


class PathPieceTrainer(BaseTrainer):
    MAX_ITERATIONS = 500

    def __init__(
        self,
        pretokenizer: Pretokenizer,
        corpus: PretokenizedCorpus,
        config: PathPieceTrainerConfig,
    ):
        super().__init__(pretokenizer, corpus, config)

    # ------------------------------------------------------------------
    # Public training entry point
    # ------------------------------------------------------------------
    def train(self) -> PathPieceModel:
        cfg = self.config
        target_size = len(self.pretokenizer.atomic_tokens) + cfg.additional_vocab_size
        L = cfg.max_token_width

        if cfg.init == "ngram":
            init_tokens = self._build_ngram_init_vocab(cfg.init_vocab_size, L)
        elif cfg.init == "bpe":
            init_tokens = self._build_bpe_init_vocab(cfg.init_vocab_size, L)
        else:
            raise ValueError(f"Unknown PathPiece init: {cfg.init!r}")

        model = PathPieceModel(self.pretokenizer, init_tokens)
        self.logger.info(
            f"Initialized PathPiece ({cfg.init}) with {len(model.tokens):,} tokens "
            f"(target {target_size:,}, L={L})"
        )

        history: list[dict] = []
        last_ctc = -1
        for it in range(self.MAX_ITERATIONS):
            if len(model.tokens) <= target_size:
                break

            mi_by_id, total_ctc = self._compute_mi_table(model, L)
            n_non_atomic = len(model.tokens) - len(self.pretokenizer.atomic_tokens)
            k_batch = max(1, int(cfg.prune_batch_fraction * n_non_atomic))
            max_removable = len(model.tokens) - target_size
            k = min(k_batch, max_removable)

            removable = sorted(mi_by_id.items(), key=lambda kv: kv[1])
            to_remove = select_drop_batch(
                [tid for tid, _ in removable], k, model.tokens, cfg.skip_substring_in_batch
            )
            kept = [t for t in model.tokens.values() if t.id not in to_remove]
            history.append(
                {"iter": it, "vocab_before": len(model.tokens), "ctc": total_ctc, "removed": len(to_remove)}
            )
            self.logger.info(
                f"Iter {it + 1}: |V|={len(model.tokens):,} CTC={total_ctc:,} removed {len(to_remove):,} -> |V|={len(kept):,}"
            )
            model = PathPieceModel(self.pretokenizer, kept)
            last_ctc = total_ctc

        _, total_ctc = self._compute_mi_table(model, L)
        history.append({"iter": "final", "vocab_before": len(model.tokens), "ctc": total_ctc, "removed": 0})
        self.logger.info(f"Final |V|={len(model.tokens):,} CTC={total_ctc:,}")
        model.metadata = {
            "tokenizer_variant": "pathpiece",
            "init": cfg.init,
            "init_vocab_size": cfg.init_vocab_size,
            "max_token_width": L,
            "final_vocab_size": len(model.tokens),
            "final_corpus_token_count": total_ctc,
            "history": history,
            "config": cfg.model_dump(),
        }
        return model

    # ------------------------------------------------------------------
    # Initial vocabulary builders
    # ------------------------------------------------------------------
    def _build_ngram_init_vocab(self, init_vocab_size: int, L: int) -> list[UnigramToken]:
        """All single atomic tokens plus the top frequent n-grams of widths 2..L.

        Pretokenizer chunks bound where n-grams may form (a token may not
        cross a pretokenization boundary), which matches the paper's
        rule. We dedup across the corpus via the chunk-level frequency
        weighting that PretokenizedCorpus already provides.
        """
        counter: Counter[tuple] = Counter()
        for chunk, freq in self.corpus:
            n = len(chunk)
            view = memoryview(chunk).tolist()
            for w in range(2, L + 1):
                if w > n:
                    break
                for i in range(n - w + 1):
                    counter[tuple(view[i : i + w])] += freq

        atomic_ids = sorted(self.pretokenizer.atomic_tokens.keys())
        max_id = max(atomic_ids) if atomic_ids else -1

        tokens: list[UnigramToken] = []
        for aid in atomic_ids:
            tokens.append(
                UnigramToken(
                    id=aid,
                    atomic_tokens=token_array([aid]),
                    log_prob=0.0,
                    required=True,
                )
            )

        budget = max(0, init_vocab_size - len(atomic_ids))
        if budget == 0:
            self.logger.warning(
                f"init_vocab_size={init_vocab_size} is not larger than the atomic vocabulary "
                f"({len(atomic_ids)}); no n-gram tokens will be added."
            )
            return tokens

        if self.config.ngram_init_rank == "count":
            ranked = counter.most_common(budget)
        elif self.config.ngram_init_rank == "count_width":
            # Score = count * (width - 1). The "-1" makes single-byte tokens score 0
            # (they're locked in as required anyway), so this purely re-orders
            # multi-atomic-token candidates by approximate compression value.
            scored = [(ngram, count * (len(ngram) - 1)) for ngram, count in counter.items()]
            scored.sort(key=lambda kv: -kv[1])
            ranked = scored[:budget]
            self.logger.info(f"Using non-canonical ngram_init_rank=count_width (count*(width-1))")
        else:
            raise ValueError(f"Unknown ngram_init_rank: {self.config.ngram_init_rank!r}")

        next_id = max_id + 1
        for ngram, _score in ranked:
            tokens.append(
                UnigramToken(
                    id=next_id,
                    atomic_tokens=token_array(ngram),
                    log_prob=0.0,
                    required=False,
                )
            )
            next_id += 1
        return tokens

    def _build_bpe_init_vocab(self, init_vocab_size: int, L: int) -> list[UnigramToken]:
        """BPE-trained seed vocabulary capped at ``init_vocab_size``.

        BPE merges are not length-capped by default; tokens of width > L
        cannot survive PathPiece's max-width constraint and are dropped
        on conversion.
        """
        bpe_additional = max(0, init_vocab_size - len(self.pretokenizer.atomic_tokens))
        self.logger.info(
            f"Training BPE seed with additional_vocab_size={bpe_additional:,} (target init={init_vocab_size:,})"
        )
        bpe_cfg = BPETrainerConfig(
            additional_vocab_size=bpe_additional,
            num_workers=self.config.num_workers,
            verbose=self.config.verbose,
        )
        bpe_model = BPETrainer(self.pretokenizer, self.corpus, bpe_cfg).train()

        tokens: list[UnigramToken] = []
        dropped = 0
        for bpe_token in bpe_model.tokens.values():
            if len(bpe_token.atomic_tokens) > L:
                dropped += 1
                continue
            tokens.append(
                UnigramToken(
                    id=bpe_token.id,
                    atomic_tokens=token_array(bpe_token.atomic_tokens),
                    log_prob=0.0,
                    required=len(bpe_token.atomic_tokens) == 1,
                )
            )
        if dropped:
            self.logger.info(f"Dropped {dropped} BPE tokens of width > L={L} from PathPiece init")
        return tokens

    # ------------------------------------------------------------------
    # MI computation (delegates to shared utility)
    # ------------------------------------------------------------------
    def _compute_mi_table(self, model: PathPieceModel, L: int) -> tuple[dict[int, float], int]:
        return compute_mi_table(model, self.corpus, L, logger=self.logger)


__all__ = ["PathPieceTrainer", "PathPieceTrainerConfig"]
