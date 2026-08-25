"""N-gram count tables over a token-id stream, built with NumPy.

The one idea that makes this fast enough to be useful: *dense prefix ids*. Counting
k-grams by hashing the k token ids is O(k) per position and needs a Python dict; instead
we compact each order to dense ids and build the next order from the previous one:

    key_k[i] = gram_id_{k-1}[i - 1] * radix + stream[i]

`gram_id_{k-1}` is already dense (< number of distinct (k-1)-grams <= stream length), so
the key fits comfortably in int64 for any order, and one `np.unique` per order does the
counting. Cost is O(n log n) per order with no Python-level loop over positions.

Throughout, `gram_id_k[i]` is the dense id of the k-gram *ending at* position i, i.e.
`stream[i - k + 1 : i + 1]`.

Documents are padded with `order - 1` BOS ids, so every position that predicts a real
token has a full history inside its own document, and no n-gram ever straddles two
documents. `mask[i]` marks the positions that are predicted (real tokens and EOS, never
BOS padding).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrderTable:
    """One order's worth of n-gram types, in dense-id order.

    Attributes:
        keys: sorted int64 packed keys; `searchsorted` on this maps a (parent id, token)
            pair to its dense id, which is how eval-side lookups are vectorized.
        counts: raw occurrence count per type.
        parent: dense id of each type's (k-1)-gram *prefix* (`keys // radix`).
        suffix: dense id of each type's (k-1)-gram *suffix* -- the history dropped from
            the front. Prefix ids come free from the construction; suffix ids have to be
            scattered from the stream, and they are what continuation counts need.
    """

    keys: np.ndarray
    counts: np.ndarray
    parent: np.ndarray
    suffix: np.ndarray

    def __len__(self) -> int:
        return len(self.keys)


def build_stream(doc_token_ids, order: int, bos_id: int, eos_id: int):
    """Concatenate encoded documents into one padded stream plus a target mask.

    Each document contributes `order - 1` BOS ids, its own tokens, then one EOS. The EOS
    is predicted like any other symbol, which is what makes the model a proper
    distribution over documents rather than over a single unending sequence.

    Returns:
        (stream, mask, doc_pos): the token ids, the predicted positions, and each
        position's index within its own document. `doc_pos` is what keeps an n-gram from
        straddling two documents -- a k-window is in-document exactly when
        `doc_pos >= k - 1`.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    pad = order - 1
    pieces, masks, positions = [], [], []
    for ids in doc_token_ids:
        arr = np.asarray(ids, dtype=np.int32)
        length = pad + len(arr) + 1
        pieces.append(np.full(pad, bos_id, dtype=np.int32))
        pieces.append(arr)
        pieces.append(np.array([eos_id], dtype=np.int32))
        masks.append(np.zeros(pad, dtype=bool))
        masks.append(np.ones(len(arr) + 1, dtype=bool))
        positions.append(np.arange(length, dtype=np.int32))
    if not pieces:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int32)
    return np.concatenate(pieces), np.concatenate(masks), np.concatenate(positions)


def iter_gram_ids(stream: np.ndarray, mask: np.ndarray, doc_pos: np.ndarray, order: int,
                  radix: int, tables: list[OrderTable] | None = None):
    """Yield `(k, gram_id_k, tables)` for k = 1..order, holding only two orders at a time.

    Ids are assigned at every position holding a full in-document k-window, but counts are
    accumulated only at *masked* positions. The two sets differ, and the difference
    matters: the history of a document's first token ends on BOS padding, which is never
    itself predicted. Counting there would invent occurrences of a symbol that never
    occurs; not indexing there would throw away every document-initial context and make
    the model back off to nothing at the start of every document.

    With `tables=None` this builds the tables from `stream` (training). With `tables` given
    it looks ids up in them (evaluation): a k-gram absent from training gets id -1, and -1
    propagates upward, since a k-gram whose (k-1)-prefix was never seen cannot itself have
    been seen.

    A generator rather than a list because each order's id array is one int32 per stream
    position, and every consumer needs only the current order and the one below it. On a
    50M-token corpus at order 5 that is the difference between holding 1 GB of ids and
    400 MB of them.
    """
    n = len(stream)
    if n > np.iinfo(np.int32).max:
        raise ValueError(f"stream of {n:,} positions exceeds the int32 id space; split the corpus")
    building = tables is None
    out_tables: list[OrderTable] = [] if building else list(tables)
    prev_gid: np.ndarray | None = None

    for k in range(1, order + 1):
        # Positions whose k-window lies wholly inside one document get an id; only the
        # masked ones among them contribute counts.
        idx = np.flatnonzero(doc_pos >= k - 1)
        counted = mask[idx]
        if k == 1:
            keys = stream[idx].astype(np.int64)
        else:
            # int64 explicitly: ids are int32 to keep them cheap, but under NEP 50 an
            # int32 array times a Python int stays int32, and `id * radix` passes 2^31 at
            # a few hundred thousand types -- which silently wraps negative and collides
            # distinct n-grams onto one another. Widen before multiplying, not after.
            prev_at_ctx = prev_gid[idx - 1].astype(np.int64)
            keys = np.where(prev_at_ctx < 0, -1, prev_at_ctx * radix + stream[idx].astype(np.int64))

        gid = np.full(n, -1, dtype=np.int32)
        if building:
            valid = keys >= 0
            uniq, inv = np.unique(keys[valid], return_inverse=True)
            gid[idx[valid]] = inv
            counts = np.bincount(inv[counted[valid]], minlength=len(uniq)).astype(np.int64)
            out_tables.append(OrderTable(
                keys=uniq,
                counts=counts,
                parent=(uniq // radix) if k > 1 else np.zeros(len(uniq), dtype=np.int64),
                suffix=np.zeros(len(uniq), dtype=np.int32),  # filled in below
            ))
        else:
            table = out_tables[k - 1]
            hit = keys >= 0
            target = idx[hit]
            if len(table.keys):
                # searchsorted returns the insertion point, which is len(keys) for a probe
                # past the end; clip before indexing, then confirm the key really matches.
                pos = np.minimum(np.searchsorted(table.keys, keys[hit]), len(table.keys) - 1)
                found = table.keys[pos] == keys[hit]
                gid[target[found]] = pos[found].astype(np.int32)
        del keys

        # Scatter suffix ids: at any position the k-gram and the (k-1)-gram ending there
        # share their last k-1 tokens, so the latter *is* the former's suffix. Every
        # occurrence agrees, so a last-write-wins scatter is exact.
        if building and k > 1:
            sel = idx[(gid[idx] >= 0) & (prev_gid[idx] >= 0)]
            suffix = np.zeros(len(out_tables[k - 1]), dtype=np.int32)
            suffix[gid[sel]] = prev_gid[sel]
            out_tables[k - 1] = OrderTable(
                keys=out_tables[k - 1].keys,
                counts=out_tables[k - 1].counts,
                parent=out_tables[k - 1].parent,
                suffix=suffix,
            )

        yield k, gid, out_tables
        prev_gid = gid


def gram_ids(stream: np.ndarray, mask: np.ndarray, doc_pos: np.ndarray, order: int, radix: int,
             tables: list[OrderTable] | None = None):
    """`iter_gram_ids` collected into `(ids, tables)`, for tests and small inputs.

    Materializes every order's ids at once; prefer `iter_gram_ids` on real corpora.
    """
    ids, out_tables = [], []
    for _, gid, out_tables in iter_gram_ids(stream, mask, doc_pos, order, radix, tables):
        ids.append(gid)
    return ids, out_tables
