"""The tokenizer contract pynanochat evaluates against — tokenizer-agnostic.

pynanochat doesn't know or care how a tokenizer was built (BPE, unigram,
minimum-token models, byte-level, ...). A caller (e.g. a tokenizer-research package)
provides anything satisfying `Tokenizer` below and pynanochat pretrains a
nanochat base model on it and scores CORE + bpb.

The surface mirrors what nanochat's training/eval actually touch:

    encode(text, prepend=None, append=None) -> list[int]
    decode(ids) -> str
    get_bos_token_id() -> int
    get_vocab_size() -> int
    get_special_tokens() -> list[str]
    id_to_token(id) -> str

Two ways to satisfy nanochat with such a tokenizer:

  - inject(tok): runtime-patch nanochat.get_tokenizer to return `tok`. Works for
    ANY tokenizer (the general path). No nanochat fork.
  - write_tiktoken_pickle(...): if the tokenizer happens to be rank-based BPE,
    emit a tiktoken.Encoding pickle nanochat loads natively. Optional fast path.

`write_token_bytes` is required either way (the bpb metric reads it) and is fully
general — just UTF-8 byte length of each token, 0 for specials.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str, prepend=None, append=None) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def get_bos_token_id(self) -> int: ...
    def get_vocab_size(self) -> int: ...
    def get_special_tokens(self) -> list[str]: ...
    def id_to_token(self, id: int) -> str: ...


def write_token_bytes(tokenizer: Tokenizer, tokenizer_dir: str | Path):
    """Write `<tokenizer_dir>/token_bytes.pt`: int32[vocab_size], specials -> 0.

    Faithful to nanochat/scripts/tok_train.py. Needs only get_vocab_size(),
    get_special_tokens(), decode([id]).
    """
    import torch

    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    special = set(tokenizer.get_special_tokens())
    counts = [
        0 if (s := tokenizer.decode([tid])) in special else len(s.encode("utf-8"))
        for tid in range(tokenizer.get_vocab_size())
    ]
    token_bytes = torch.tensor(counts, dtype=torch.int32, device="cpu")
    with open(tokenizer_dir / "token_bytes.pt", "wb") as f:
        torch.save(token_bytes, f)
    return token_bytes


def inject(tokenizer: Tokenizer):
    """Make `nanochat.tokenizer.get_tokenizer()` return `tokenizer` (general path)."""
    import nanochat.tokenizer as nt
    nt.get_tokenizer = lambda: tokenizer


def write_tiktoken_pickle(mergeable_ranks, special_tokens, pat_str, tokenizer_dir):
    """Optional fast path for rank-based BPE: pickle a tiktoken.Encoding that
    nanochat's RustBPETokenizer.from_directory() loads natively.

    TODO: implement once a concrete BPE tokenizer needs it; not required for the
    general `inject` path above.
    """
    raise NotImplementedError("optional fast path; use inject() for the general case")
