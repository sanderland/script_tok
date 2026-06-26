"""PathPiece tokenizer model (Schmidt et al., EMNLP 2024).

Segments using the minimum-token shortest-path DP (Algorithm 1 in the
paper): for each end position e and each candidate width w in 1..L, if
d[s:e] is in the vocabulary, update pl[e] = min(pl[e], pl[s-1] + 1)
and record wid[e] = w. Ties on path length are broken by preferring
the longest token (the ``≤`` rule in the paper).
"""

import gzip
import json
import os

from script_bpe.pretokenize import Pretokenizer, export_pretokenizer, load_pretokenizer
from script_bpe.tokenizers.base import BaseTokenizer
from script_bpe.tokenizers.unigram.model import Trie, UnigramToken, reindex_tokens
from script_bpe.utils import InputTokenSeq, TokenSeq, token_array


class PathPieceModel(BaseTokenizer):
    VERSION = "sepathpiece-v1"
    REPORT_TITLE = "PathPiece Tokenizer Report"

    def __init__(
        self,
        pretokenizer: Pretokenizer,
        tokens: list[UnigramToken],
        metadata: dict | None = None,
    ):
        self.pretokenizer = pretokenizer
        self.tokens = {t.id: t for t in (tokens or [])}
        self.trie = Trie(tokens)
        self.metadata = metadata or {}
        self.tokens_by_id = self.tokens
        self._max_token_width = max(
            (len(t.atomic_tokens) for t in self.tokens.values()), default=1
        )

    def encode_chunk(self, chunk: TokenSeq) -> list[UnigramToken]:
        """Shortest-path segmentation with longest-token tiebreak.

        Implemented as a walk-forward-from-each-start DP. For a given end
        position e, contributions arrive in order of decreasing token
        width as the outer loop s advances. Using a strict ``<``
        comparator therefore keeps the first (largest-width) writer on
        ties, equivalent to the paper's ``pl[e] ← nl if nl ≤ pl[e]``
        when iterating w ascending.
        """
        n = len(chunk)
        unreachable = n + 1
        pl = [unreachable] * (n + 1)
        pl[0] = 0
        last_tok: list[UnigramToken | None] = [None] * (n + 1)

        trie_root = self.trie.root
        max_w = self._max_token_width
        for s in range(n):
            base = pl[s]
            if base >= unreachable:
                continue
            base += 1
            node = trie_root
            limit = n if max_w >= n - s else s + max_w
            for i in range(s, limit):
                node = node.get(chunk[i])
                if node is None:
                    break
                tok = node.get(None)
                if tok is not None:
                    e = i + 1
                    if base < pl[e]:
                        pl[e] = base
                        last_tok[e] = tok

        tokens: list[UnigramToken] = []
        e = n
        while e > 0:
            tok = last_tok[e]
            assert tok is not None, f"No valid PathPiece segmentation for chunk {chunk!r}"
            tokens.append(tok)
            e -= len(tok.atomic_tokens)
        tokens.reverse()
        return tokens

    def encode(self, text: str, return_tokens: bool = False) -> list[UnigramToken] | list[int]:
        tokens = [
            t
            for chunk in self.pretokenizer.pretokenize(text)
            for t in self.encode_chunk(chunk)
        ]
        if return_tokens:
            return tokens
        return [t.id for t in tokens]

    def decode(self, ids: InputTokenSeq) -> str:
        atomic_tokens = [
            tid for token_id in ids for tid in self.tokens_by_id[token_id].atomic_tokens
        ]
        return self.pretokenizer.decode(atomic_tokens)

    @classmethod
    def load(cls, file: str, reindex: bool = False):
        open_func = gzip.open if file.endswith(".gz") else open
        with open_func(file, "rt") as f:
            data = json.load(f)
        pretokenizer = load_pretokenizer(data["pretokenizer"])
        tokens = [
            UnigramToken(
                id=d["id"],
                atomic_tokens=token_array(d["atomic_tokens"]),
                log_prob=float(d.get("log_prob", 0.0)),
                required=bool(d.get("required", len(d["atomic_tokens"]) == 1)),
            )
            for d in data["tokens"]
        ]
        if reindex:
            tokens = reindex_tokens(tokens)
        return cls(pretokenizer=pretokenizer, tokens=tokens, metadata=data.get("metadata"))

    def save(self, file_path: str) -> str:
        dirname = os.path.dirname(file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        open_func = gzip.open if file_path.endswith(".gz") else open
        with open_func(file_path, "wt") as f:
            json.dump(
                {
                    "info": {"version": self.VERSION},
                    "pretokenizer": export_pretokenizer(self.pretokenizer),
                    "tokens": [
                        {
                            "id": t.id,
                            "atomic_tokens": list(t.atomic_tokens),
                            "required": t.required,
                        }
                        for t in self.tokens.values()
                    ],
                    "metadata": self.metadata,
                },
                f,
                indent=2,
            )
        return file_path

    def report_details(self, n_longest: int = 20) -> dict[str, list[dict]]:
        metadata_rows: list[dict] = []
        for k, v in self.metadata.items():
            if k != "tokens":
                metadata_rows.append({"Key": k, "Value": v})
        sections: dict[str, list[dict]] = {}
        if metadata_rows:
            sections["Metadata"] = metadata_rows
        return sections


__all__ = ["PathPieceModel"]
