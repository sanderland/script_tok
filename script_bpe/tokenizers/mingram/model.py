import gzip
import json
import os

from script_bpe.pretokenize import Pretokenizer, export_pretokenizer, load_pretokenizer
from script_bpe.tokenizers.base import BaseTokenizer
from script_bpe.tokenizers.unigram.model import Trie, UnigramToken, reindex_tokens
from script_bpe.utils import InputTokenSeq, TokenSeq

ATOMIC_FALLBACK_LOG_PROB_FLOOR = -100.0
# Path score = score_delta * sum(log_prob) - num_tokens. The default 1/100_000 is argmax-
# equivalent to the historical "sum(log_prob) - 100_000*num_tokens" (it's that, divided by
# 100_000), so fewer tokens always wins with log_prob as a tiebreaker. score_delta=0 gives
# pure minimum-token segmentation with the longest-token tiebreak (== PathPiece); larger
# score_delta weights likelihood more (toward Unigram).
DEFAULT_SCORE_DELTA = 1.0 / 100_000.0


class MinGramModel(BaseTokenizer):
    VERSION = "semingram-v1"
    REPORT_TITLE = "MinGram Tokenizer Report"

    def __init__(self, pretokenizer: Pretokenizer, tokens: list[UnigramToken], metadata: dict | None = None,
                 score_delta: float | None = None):
        self.pretokenizer = pretokenizer
        self.tokens = {t.id: t for t in (tokens or [])}
        self.trie = Trie(tokens)
        self.metadata = metadata or {}
        self.tokens_by_id = self.tokens
        if score_delta is None:
            cfg = self.metadata.get("config", {}) or {}
            score_delta = cfg.get("score_delta", self.metadata.get("score_delta", DEFAULT_SCORE_DELTA))
        self.score_delta = score_delta

    def encode_chunk(self, chunk: TokenSeq) -> list[UnigramToken]:
        chunk_len = len(chunk)
        # Path score = score_delta * sum(log_prob) - num_tokens (see DEFAULT_SCORE_DELTA).
        sd = self.score_delta

        best_score: list[float] = [float("-inf")] * (chunk_len + 1)
        best_prev: list[UnigramToken | None] = [None] * (chunk_len + 1)
        best_score[0] = 0.0

        trie_root = self.trie.root
        for pos in range(chunk_len):
            score = best_score[pos]
            # A position can be reachable even with score -inf if every path to it
            # uses a required atomic fallback token. Only skip genuinely
            # unreachable positions with no backpointer.
            if pos > 0 and best_prev[pos] is None:
                continue

            node = trie_root
            for i in range(pos, chunk_len):
                node = node.get(chunk[i])
                if node is None:
                    break

                token = node.get(None)
                if token is not None:
                    nxt = i + 1
                    # sd==0 must skip the multiply: 0 * (-inf) (unused atomic fallback) is NaN.
                    contrib = sd * token.log_prob if sd != 0.0 else 0.0
                    new_score = score + contrib - 1.0
                    # >= with epsilon for float tiebreaks (last write wins left-to-right,
                    # matching LIFO longest-first behaviour of the original DP). At score_delta=0
                    # all equal-length paths tie exactly, so longest-first wins == PathPiece.
                    if new_score >= best_score[nxt] - 1e-12:
                        best_score[nxt] = new_score
                        best_prev[nxt] = token

        tokens: list[UnigramToken] = []
        pos = chunk_len
        while pos > 0:
            token = best_prev[pos]
            assert token is not None, f"No valid segmentation found for chunk: {chunk!r}"
            tokens.append(token)
            pos -= len(token.atomic_tokens)
        tokens.reverse()
        return tokens

    def encode(self, text: str, return_tokens: bool = False) -> list[UnigramToken] | list[int]:
        tokens = [token for chunk in self.pretokenizer.pretokenize(text) for token in self.encode_chunk(chunk)]
        if return_tokens:
            return tokens
        return [token.id for token in tokens]

    def decode(self, ids: InputTokenSeq) -> str:
        atomic_tokens = [tid for token_id in ids for tid in self.tokens_by_id[token_id].atomic_tokens]
        return self.pretokenizer.decode(atomic_tokens)

    @classmethod
    def load(cls, file: str, reindex: bool = False):
        """Load a MinGramModel from a file.

        Args:
                file: Path to the model file (.json or .json.gz)
                reindex: When False, preserve serialized token ids exactly.
                    When True, rewrite model token ids to dense ``0..n-1`` in
                    serialized token-list order. This does not modify
                    ``atomic_tokens``, which remain aligned with the serialized
                    pretokenizer's base-token ids.

        Returns:
                MinGramModel: The loaded model
        """
        open_func = gzip.open if file.endswith(".gz") else open
        with open_func(file, "rt") as f:
            data = json.load(f)
        pretokenizer = load_pretokenizer(data["pretokenizer"])
        tokens = [UnigramToken(**t) for t in data["tokens"]]
        for token in tokens:
            if len(token.atomic_tokens) == 1 and token.log_prob == float("-inf"):
                token.log_prob = ATOMIC_FALLBACK_LOG_PROB_FLOOR
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
                    "tokens": [t.to_dict() for t in self.tokens.values()],
                    "metadata": self.metadata,
                },
                f,
                indent=2,
            )
        return file_path
