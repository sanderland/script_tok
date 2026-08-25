"""Interpolated modified Kneser-Ney over token-id streams (Chen & Goodman, 1998).

Why modified KN and not something simpler: comparing tokenizers by held-out bits means
the smoothing has to be *fair* to vocabularies of different shapes. A crude estimator
(add-k, plain backoff) leaks probability mass in a way that scales with vocabulary size
and sparsity, so it would rank vocabularies by how well they suit the estimator rather
than by how predictable they make the text. Modified KN is the standard the n-gram
literature settled on, and it is what KenLM's `lmplz` builds.

Two properties are load-bearing for the bits-per-byte metric:

1. **The model is a proper distribution.** Every level interpolates down, and level 1
   interpolates into a uniform over the whole emittable alphabet (the tokenizer's tokens
   plus EOS). So no token can ever get probability zero, and the per-position
   probabilities sum to exactly 1 -- summed `-log2 p` is a real code length, not a score.

2. **Vocabulary size enters honestly.** The uniform floor is `1 / (V + 1)`, so a
   vocabulary that carries tokens it never earns back pays for them in the held-out
   bits. That is a property we want a tokenizer metric to have, not a nuisance.

Counts follow the standard recipe: the highest order uses raw occurrence counts, every
lower order uses continuation counts `N1+(. g)` -- the number of distinct types that
extend `g` to the left.
"""

from dataclasses import dataclass, field

import numpy as np

from script_bpe.ngram.counts import OrderTable, iter_gram_ids

# Chen & Goodman's estimator needs n1..n4 > 0; below that it is undefined and we fall back
# to plain interpolated absolute discounting, whose conventional constant is 0.75.
FALLBACK_DISCOUNT = 0.75
# Discounts are clamped away from both ends. D_i <= 0 would make gamma zero for some
# context and hand an unseen continuation probability zero (infinite bits); D_i >= i would
# make a discounted count negative.
MIN_DISCOUNT = 0.1
DISCOUNT_MARGIN = 0.01


def estimate_discounts(counts: np.ndarray) -> np.ndarray:
    """Modified Kneser-Ney discounts D1, D2, D3+ from a level's count-of-counts.

    Returns a length-4 array indexed by `min(count, 3)`; entry 0 is unused padding so the
    caller can index it directly with a clipped count.
    """
    n = [int((counts == i).sum()) for i in (1, 2, 3, 4)]
    if not all(n):
        return np.array([0.0, FALLBACK_DISCOUNT, FALLBACK_DISCOUNT, FALLBACK_DISCOUNT])
    n1, n2, n3, n4 = n
    y = n1 / (n1 + 2 * n2)
    raw = [1 - 2 * y * n2 / n1, 2 - 3 * y * n3 / n2, 3 - 4 * y * n4 / n3]
    return np.array([0.0] + [float(np.clip(d, MIN_DISCOUNT, i - DISCOUNT_MARGIN))
                             for i, d in zip((1, 2, 3), raw)])


@dataclass
class Level:
    """One backoff level of the model, indexed by dense n-gram id.

    `count` is raw at the top order and continuation counts below it. `ctx_total` and
    `gamma` are indexed by the *parent* order's ids, i.e. by the context.
    """

    count: np.ndarray
    ctx_total: np.ndarray
    gamma: np.ndarray
    discounts: np.ndarray


@dataclass
class KneserNeyLM:
    """A fitted interpolated modified Kneser-Ney model over token ids."""

    order: int
    radix: int
    alphabet_size: int
    tables: list[OrderTable]
    levels: dict[int, Level] = field(default_factory=dict)
    p1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p1_unseen: float = 0.0

    @classmethod
    def fit(cls, stream: np.ndarray, mask: np.ndarray, doc_pos: np.ndarray, *, order: int,
            radix: int, alphabet_size: int) -> "KneserNeyLM":
        """Count `stream` and estimate every level's discounts and interpolation weights."""
        tables: list[OrderTable] = []
        for _, _, tables in iter_gram_ids(stream, mask, doc_pos, order, radix):
            pass
        model = cls(order=order, radix=radix, alphabet_size=alphabet_size, tables=tables)

        # Continuation counts: how many distinct (k+1)-grams extend each k-gram leftward.
        # Only the top order uses raw counts; every level below is a continuation model.
        count_used: dict[int, np.ndarray] = {order: tables[order - 1].counts}
        for k in range(order - 1, 0, -1):
            # Only types that actually occur contribute: the tables also index grams that
            # exist purely as contexts (those ending on BOS padding), and a gram that
            # predicts BOS is not an observed left-extension of anything.
            higher = tables[k]
            count_used[k] = np.bincount(higher.suffix[higher.counts > 0],
                                        minlength=len(tables[k - 1])).astype(np.int64)

        for k in range(2, order + 1):
            counts = count_used[k]
            parent = tables[k - 1].parent
            n_ctx = len(tables[k - 2])
            discounts = estimate_discounts(counts)
            ctx_total = np.bincount(parent, weights=counts.astype(np.float64), minlength=n_ctx)
            # gamma(h) = sum_i D_i * N_i(h .), the mass this level discounts away from h
            # and hands to the level below. Equals ctx_total - sum_w max(c - D, 0).
            gamma = np.zeros(n_ctx, dtype=np.float64)
            for i in (1, 2, 3):
                sel = parent[counts == i] if i < 3 else parent[counts >= 3]
                gamma += discounts[i] * np.bincount(sel, minlength=n_ctx)
            model.levels[k] = Level(count=counts, ctx_total=ctx_total, gamma=gamma, discounts=discounts)

        # Level 1 interpolates into a uniform over everything the tokenizer can emit, which
        # is what keeps the distribution proper and prices unused vocabulary.
        c1 = count_used[1].astype(np.float64)
        d1 = estimate_discounts(count_used[1])
        total1 = float(c1.sum())
        gamma1 = float(sum(d1[i] * ((count_used[1] == i).sum() if i < 3 else (count_used[1] >= 3).sum())
                           for i in (1, 2, 3)))
        uniform = 1.0 / alphabet_size
        if total1 <= 0:
            model.p1 = np.zeros(len(tables[0]))
            model.p1_unseen = uniform
        else:
            discounted = np.maximum(c1 - d1[np.clip(count_used[1], 0, 3)], 0.0)
            model.p1 = (discounted + gamma1 * uniform) / total1
            model.p1_unseen = (gamma1 * uniform) / total1
        model.levels[1] = Level(count=count_used[1], ctx_total=np.array([total1]),
                                gamma=np.array([gamma1]), discounts=d1)
        return model

    def log2_probs(self, stream: np.ndarray, mask: np.ndarray, doc_pos: np.ndarray) -> np.ndarray:
        """log2 p(token | history) at every masked position of `stream`."""
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return np.zeros(0)

        p = np.zeros(len(idx))
        prev_gid = None
        # Orders arrive lowest first, and level k needs only order k and order k-1, so the
        # interpolation can be built up in one pass alongside the id generator.
        for k, gid, _ in iter_gram_ids(stream, mask, doc_pos, self.order, self.radix, tables=self.tables):
            if k == 1:
                id1 = gid[idx]
                p = np.where(id1 >= 0, self.p1[np.maximum(id1, 0)], self.p1_unseen)
                prev_gid = gid
                continue
            level = self.levels[k]
            ctx = prev_gid[idx - 1]            # the (k-1)-gram ending one position back
            gid_here = gid[idx]                # the full k-gram ending here
            ctx_safe = np.maximum(ctx, 0).astype(np.int64)
            denom = level.ctx_total[ctx_safe]
            # A context absent from training, or one that never appeared *as* a context,
            # contributes nothing: the model backs off wholly to the level below.
            known = (ctx >= 0) & (denom > 0)
            seen = gid_here >= 0
            c = level.count[np.maximum(gid_here, 0).astype(np.int64)].astype(np.float64)
            numer = np.where(seen, np.maximum(c - level.discounts[np.clip(c, 0, 3).astype(np.int64)], 0.0), 0.0)
            p = np.where(known, (numer + level.gamma[ctx_safe] * p) / np.maximum(denom, 1.0), p)
            prev_gid = gid

        if not np.all(p > 0):
            raise FloatingPointError("Kneser-Ney assigned zero probability; discounts collapsed")
        return np.log2(p)
