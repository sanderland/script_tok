"""Make a script_bpe tokenizer look like a nanochat tokenizer.

nanochat's training/eval reach a tokenizer only through `get_tokenizer()` and a
small surface (see `nanochat/tokenizer.py::RustBPETokenizer`). script_bpe's
`BaseTokenizer` has a *different*, smaller surface:

    BaseTokenizer.encode(text: str) -> array[int]      # single string only
    BaseTokenizer.decode(ids, errors="replace") -> str
    BaseTokenizer.tokens: Mapping[int, BaseToken]      # id -> token

This adapter wraps one and exposes exactly what nanochat calls:

    __call__(text, prepend, append, num_threads)       # core_eval uses tokenizer(...)
    encode(text|list[str], prepend, append, num_threads)
    decode(ids) -> str
    encode_special(s) -> int
    get_bos_token_id() -> int
    get_vocab_size() -> int
    get_special_tokens() -> set[str]
    id_to_token(id) -> str

Two wrinkles drive the id bookkeeping (verified against real script_bpe models):

  * script_bpe ids are NOT contiguous (1-indexed; pruned tokenizers like MinGram keep
    sparse ids from a larger init vocab — e.g. 34,684 tokens spread over ids up to
    42,874). Using ``max(id)`` as the vocab would give the model thousands of dead
    embedding rows AND a tokenizer-dependent vocab size, making cross-tokenizer
    comparison unfair (different params + token budget). So we **remap to a dense
    [0, n) space** (n = number of real tokens), giving every tokenizer the same
    hole-free ``vocab_size = n + 1``.
  * script_bpe has no BOS / special token, but nanochat's dataloader and CORE eval
    both require one — we synthesize it as dense id ``n`` (``vocab_size = n + 1``).

`encode` maps script_bpe ids -> dense; `decode` maps dense -> script_bpe ids (dropping
the synthetic BOS), so the model only ever sees the contiguous dense space.
"""

from __future__ import annotations

import atexit
import multiprocessing

BOS_TOKEN = "<|bos|>"

# Below this batch size, pool dispatch overhead isn't worth it — tokenize serially.
_MIN_PARALLEL_BATCH = 8

# --- multiprocessing batch encode --------------------------------------------
# script_bpe.encode is pure-Python and GIL-bound, so threads don't help (they hurt);
# processes scale ~linearly. The dataloader calls encode() *during* training when CUDA
# is live, and forking a CUDA-initialized (multithreaded) process can deadlock. So we
# build the pool with plain `fork` but do it EARLY — `init_encode_pool` is called from
# bootstrap.py, before nanochat initializes CUDA, when the process is still clean and
# single-threaded. Workers then inherit the already-loaded tokenizer via copy-on-write
# (no reload, no pickling). `forkserver`/`spawn` are unreliable here (the train child
# runs under `python -c`), so pre-CUDA fork is both the simplest and the safest option.

_POOL = None         # the worker pool (created pre-CUDA by init_encode_pool)
_WORKER_TOK = None   # tokenizer the workers encode with (inherited at fork time)
_WORKER_REMAP = None  # old-id -> dense-id map (inherited at fork time)


def _worker_encode(text: str) -> list[int]:
    return [_WORKER_REMAP[i] for i in _WORKER_TOK.encode(text)]


def init_encode_pool(adapter, workers: int):
    """Create the fork pool for parallel batch encode. MUST run before CUDA init.

    Sets the worker tokenizer + id remap in this process, then forks `workers` children
    that inherit them copy-on-write. Idempotent.
    """
    global _POOL, _WORKER_TOK, _WORKER_REMAP
    if _POOL is not None or workers <= 1:
        return _POOL
    _WORKER_TOK = adapter._st
    _WORKER_REMAP = adapter._old2dense
    _POOL = multiprocessing.get_context("fork").Pool(workers)
    atexit.register(_close_pool)  # avoid a noisy Pool.__del__ traceback at shutdown
    return _POOL


def _close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.terminate()
        _POOL = None


class ScriptBPETokenizerAdapter:
    """Wrap a ``script_bpe.tokenizers.base.BaseTokenizer`` to nanochat's surface."""

    def __init__(self, st_tokenizer, bos_token: str = BOS_TOKEN):
        self._st = st_tokenizer
        self._bos_token = bos_token
        # Remap script_bpe's sparse/non-contiguous ids to a dense [0, n) space so the
        # vocab is hole-free and identical across tokenizers (fair comparison).
        old_ids = sorted(st_tokenizer.tokens)
        self._dense2old = old_ids                       # dense id -> script_bpe id
        self._old2dense = {o: d for d, o in enumerate(old_ids)}
        n = len(old_ids)
        self._bos_id = n                                # synthetic BOS = next dense id
        self._vocab_size = n + 1

    # --- nanochat surface -------------------------------------------------

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_bos_token_id(self) -> int:
        return self._bos_id

    def get_special_tokens(self) -> set[str]:
        return {self._bos_token}

    def encode_special(self, text: str) -> int:
        if text == self._bos_token:
            return self._bos_id
        raise ValueError(f"unknown special token {text!r}; adapter only defines {self._bos_token!r}")

    def id_to_token(self, id: int) -> str:
        if id == self._bos_id:
            return self._bos_token
        return self.decode([id])

    def decode(self, ids) -> str:
        # Map dense ids back to script_bpe ids; drop the synthetic BOS / out-of-range.
        d2o = self._dense2old
        n = len(d2o)
        real = [d2o[i] for i in ids if 0 <= i < n]
        return self._st.decode(real)  # not all script_bpe tokenizers accept errors= (e.g. MinGram)

    def encode(self, text, prepend=None, append=None, num_threads: int = 8):
        # prepend/append arrive as ints (the bos id) from the dataloader, or as
        # special-token strings from eval code; mirror RustBPETokenizer's handling.
        prepend_id = None if prepend is None else (prepend if isinstance(prepend, int) else self.encode_special(prepend))
        append_id = None if append is None else (append if isinstance(append, int) else self.encode_special(append))
        o2d = self._old2dense

        def _one(s: str) -> list[int]:
            ids = [o2d[i] for i in self._st.encode(s)]  # script_bpe ids -> dense
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
            return ids

        def _wrap(ids: list[int]) -> list[int]:
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
            return ids

        if isinstance(text, str):
            return _one(text)
        if isinstance(text, list):
            # num_threads (passed by nanochat) would imply threads, but script_bpe.encode
            # is GIL-bound so threads slow it down. We parallelize across the process pool
            # (set up pre-CUDA by init_encode_pool) instead, if one exists.
            if _POOL is not None and len(text) >= _MIN_PARALLEL_BATCH:
                chunksize = max(1, len(text) // (_POOL._processes * 4))
                raw = _POOL.map(_worker_encode, text, chunksize=chunksize)
                return [_wrap(ids) for ids in raw]
            return [_one(s) for s in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)
