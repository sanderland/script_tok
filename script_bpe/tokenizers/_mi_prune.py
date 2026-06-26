"""MI pruning utility, shared by PathPiece and MinGram.

Implements the corpus-level MI computation from Schmidt et al. (2024),
§3.2, Eqs. (1)-(5). For each non-required token, MI is a lower bound on
the corpus token-count (CTC) increase that would result from removing
the token from the vocabulary, computed analytically from forward and
backward path-length arrays.

The function is agnostic to *which* model produced the segmentation, as
long as the model exposes:
  * ``model.tokens``: dict[int, UnigramToken-like]
  * ``model.trie``: a Trie of those tokens (atomic_token_seq -> token)
  * ``model.encode_chunk(chunk)``: returns the token list for one
    pretokenized chunk under the model's segmentation rule.

For PathPiece, the model's segmentation is min-token with a longest-token
tiebreak; for MinGram, min-token with a log-prob tiebreak. In both
cases the MI bound is valid because both segmentations achieve the same
``K_d`` (the min-token-count path); only the chosen tokens differ
slightly between tiebreak rules.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def compute_mi_table(
    model: Any,
    corpus: Iterable,
    max_token_width: int,
    logger: Any | None = None,
) -> tuple[dict[int, float], int]:
    """Return ``(mi_by_id, total_ctc)``.

    ``mi_by_id[tid]`` is the aggregated MI for token id ``tid``, in units
    of *additional tokens that would be produced if the token were
    dropped, summed over all of its occurrences in the current
    segmentation, weighted by chunk frequency*. Tokens with
    ``required=True`` are omitted (they cannot be pruned).
    """
    mi_by_id: dict[int, float] = {
        tid: 0.0 for tid, t in model.tokens.items() if not getattr(t, "required", False)
    }
    trie_root = model.trie.root
    total_ctc = 0
    n_chunks = 0
    L = max_token_width

    for chunk, freq in corpus:
        n = len(chunk)
        if n == 0:
            continue
        n_chunks += 1
        unreachable = n + 1

        # ---- Forward DP ----
        pl = [unreachable] * (n + 1)
        pl[0] = 0
        tokens_ending_at: list[list[tuple[int, Any]]] = [[] for _ in range(n + 1)]
        for s in range(n):
            base = pl[s]
            node = trie_root
            limit = n if L >= n - s else s + L
            for i in range(s, limit):
                node = node.get(chunk[i])
                if node is None:
                    break
                tok = node.get(None)
                if tok is not None:
                    e = i + 1
                    w = e - s
                    tokens_ending_at[e].append((w, tok))
                    if base < unreachable and base + 1 < pl[e]:
                        pl[e] = base + 1

        K_d = pl[n]
        if K_d >= unreachable:
            # No valid min-token path — should not happen with required atomics.
            continue
        total_ctc += K_d * freq

        # ---- Backward DP ----
        bpl = [unreachable] * (n + 1)
        bpl[n] = 0
        for e in range(n - 1, -1, -1):
            node = trie_root
            limit = n if L >= n - e else e + L
            best = unreachable
            for i in range(e, limit):
                node = node.get(chunk[i])
                if node is None:
                    break
                if node.get(None) is not None:
                    cand = bpl[i + 1]
                    if cand < unreachable and cand + 1 < best:
                        best = cand + 1
            bpl[e] = best

        # ---- Model-chosen segmentation ----
        seg_tokens = model.encode_chunk(chunk)
        # Translate the token list into (s, e, tok) ranges by accumulating widths.
        # The model's encode_chunk must return tokens in left-to-right order
        # whose atomic_tokens widths sum to len(chunk).
        segmentation: list[tuple[int, int, Any]] = []
        pos = 0
        for tok in seg_tokens:
            w = len(tok.atomic_tokens)
            segmentation.append((pos, pos + w, tok))
            pos += w
        assert pos == n, f"Segmentation width {pos} != chunk len {n}"

        # ---- MI per occurrence (Case 1 + Case 2) ----
        for s, e, tok in segmentation:
            if getattr(tok, "required", False):
                continue

            # Case 1: an internal break at j ∈ (s, e). Only defined for width >= 2.
            if e - s >= 2:
                best_break = unreachable
                for j in range(s + 1, e):
                    cand = pl[j] + bpl[j]
                    if cand < best_break:
                        best_break = cand
                mi_b = best_break - K_d
            else:
                mi_b = unreachable

            # Case 2: a strict superset token t' with start s' ≤ s, end e' ≥ e,
            # (s', e') ≠ (s, e), width e'-s' ≤ L.
            mi_s = unreachable
            ep_limit = min(s + L, n) + 1
            for ep in range(e, ep_limit):
                for w_prime, _tok_prime in tokens_ending_at[ep]:
                    sp = ep - w_prime
                    if sp > s:
                        continue
                    if sp == s and ep == e:
                        continue
                    cand = pl[sp] + bpl[ep] + 1
                    if cand < mi_s:
                        mi_s = cand
            mi_s -= K_d

            local_mi = min(mi_b, mi_s)
            if local_mi >= unreachable:
                local_mi = float("inf")
            mi_by_id[tok.id] += freq * local_mi

    if logger is not None:
        logger.debug(f"MI scan: {n_chunks:,} chunks, CTC={total_ctc:,}")
    return mi_by_id, total_ctc


def select_drop_batch(ordered_ids, k, tokens_by_id, skip_substring=False, max_sub_len=16):
    """Pick up to ``k`` token ids to drop from ``ordered_ids`` (already in drop-priority
    order, lowest MI first).

    With ``skip_substring`` (Craig's rule), a candidate is skipped when its atomic-token
    sequence is a contiguous subsequence of an already-selected token in this batch -- i.e.
    don't drop a token together with one that contains it, which avoids removing too many
    overlapping tokens in a single (stale-MI) batch. ``max_sub_len`` bounds the enumerated
    subsequence length for efficiency; candidates longer than it are never skipped (rare)."""
    if not skip_substring:
        return set(ordered_ids[:k])
    dropped: set = set()
    sub_seqs: set = set()  # contiguous subsequences (len <= max_sub_len) of already-dropped tokens
    for tid in ordered_ids:
        if len(dropped) >= k:
            break
        seq = tuple(tokens_by_id[tid].atomic_tokens)
        if len(seq) <= max_sub_len and seq in sub_seqs:
            continue  # substring of an already-dropped token -> keep it this round
        dropped.add(tid)
        n = len(seq)
        for i in range(n):
            for j in range(i + 1, min(i + max_sub_len, n) + 1):
                sub_seqs.add(seq[i:j])
    return dropped


__all__ = ["compute_mi_table", "select_drop_batch"]
