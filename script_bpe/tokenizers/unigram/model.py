import gzip
import json
import math
import os
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass


from script_bpe.pretokenize import Pretokenizer, export_pretokenizer, load_pretokenizer
from script_bpe.utils import InputTokenSeq, TokenSeq, token_array
from script_bpe.tokenizers import BaseToken
from script_bpe.tokenizers.base import BaseTokenizer


def logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)) for two finite terms."""
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


@dataclass(slots=True)
class UnigramToken(BaseToken):
    log_prob: float = 0.0
    required: bool = False

    def to_dict(self):
        """Convert the token to a dictionary for serialization."""
        return {
            "id": self.id,
            "atomic_tokens": list(self.atomic_tokens),
            "log_prob": self.log_prob,
        }


def reindex_tokens(tokens: list[UnigramToken]) -> list[UnigramToken]:
    """Return tokens with dense ids while preserving token-list order.

    This only rewrites the higher-level token ids used by the model. The
    `atomic_tokens` stored inside each token are left unchanged because they
    refer to pretokenizer base-token ids.
    """
    return [
        UnigramToken(
            id=new_id,
            atomic_tokens=token.atomic_tokens,
            log_prob=token.log_prob,
            required=token.required,
        )
        for new_id, token in enumerate(tokens)
    ]


class Trie:
    def __init__(self, tokens: list[UnigramToken]):
        self.root: dict[int, dict[int | None, UnigramToken]] = {}
        for token in tokens:
            self.insert(token)

    def insert(self, token: UnigramToken):
        node = self.root
        for tid in token.atomic_tokens:
            node = node.setdefault(tid, {})
        node[None] = token

    def find_prefixes(self, atomic_token_seq: TokenSeq) -> list[UnigramToken]:
        results = []
        node = self.root
        for tid in atomic_token_seq:
            if tid not in node:
                break
            node = node[tid]
            if None in node:
                results.append(node[None])
        return results


class Lattice:
    def __init__(self, atomic_token_seq: TokenSeq, tokens_from_pos: list[list[UnigramToken]], token_bias: float = 0.0):
        self.atomic_token_seq = atomic_token_seq
        self.tokens_from_pos = tokens_from_pos
        self.token_bias = token_bias

    def _effective_log_prob(self, token: UnigramToken) -> float:
        """Get effective log prob with bias applied."""
        return token.log_prob - self.token_bias

    def viterbi(self, allow_single_token=True) -> tuple[list[UnigramToken], float]:
        best_at_pos = [(None, 0.0)] + [(None, float("-inf"))] * len(self.atomic_token_seq)
        for pos in range(len(self.atomic_token_seq)):
            for token in self.tokens_from_pos[pos]:
                end = pos + len(token.atomic_tokens)
                if pos == 0 and not allow_single_token and end == len(self.atomic_token_seq):
                    continue  # do not allow direct path
                score = best_at_pos[pos][1] + self._effective_log_prob(token)
                if score > best_at_pos[end][1]:
                    best_at_pos[end] = (token, score)

        path = []
        pos = len(self.atomic_token_seq)
        while pos > 0:
            token, score = best_at_pos[pos]
            if token is None:
                break
            path.append(token)
            pos -= len(token.atomic_tokens)
        return path[::-1], best_at_pos[-1][1]

    def all_paths(self, starting_pos: int = 0) -> Generator[tuple[tuple[UnigramToken], float], None, None]:
        if starting_pos == len(self.atomic_token_seq):
            yield (tuple(), 0.0)
            return
        for token in self.tokens_from_pos[starting_pos]:
            for sub_path, sub_prob in self.all_paths(starting_pos + len(token.atomic_tokens)):
                yield ((token,) + sub_path, sub_prob + self._effective_log_prob(token))

    def _forward_backward(self) -> tuple[list[float], list[float]]:
        """returns
        alpha(pos) = total prob of path to pos
        beta(pos) = total prob of path from pos to end
        """
        alpha = [0] + [float("-inf")] * len(self.atomic_token_seq)
        beta = [float("-inf")] * (len(self.atomic_token_seq)) + [0]

        for pos in range(len(self.atomic_token_seq)):
            if alpha[pos] != float("-inf"):
                for token in self.tokens_from_pos[pos]:
                    end = pos + len(token.atomic_tokens)
                    alpha[end] = logaddexp(alpha[end], alpha[pos] + self._effective_log_prob(token))

        for pos in range(len(self.atomic_token_seq) - 1, -1, -1):
            for token in self.tokens_from_pos[pos]:
                end = pos + len(token.atomic_tokens)
                if beta[end] != float("-inf"):
                    beta[pos] = logaddexp(beta[pos], beta[end] + self._effective_log_prob(token))
        return alpha, beta

    def calc_marginal(self) -> tuple[float, dict[int, float]]:
        alpha, beta = self._forward_backward()
        z = alpha[-1]
        assert z != float("-inf"), (
            f"Lattice for {self.atomic_token_seq!r} has no valid paths with tokens_from_pos {self.tokens_from_pos}"
        )
        token_prob: dict[int, float] = defaultdict(float)
        for pos in range(len(self.atomic_token_seq)):
            for token in self.tokens_from_pos[pos]:
                token_logprob = alpha[pos] + self._effective_log_prob(token) + beta[pos + len(token.atomic_tokens)] - z
                token_prob[token.id] += math.exp(max(-100, token_logprob))  # Avoid underflow

        return z, token_prob


class UnigramModel(BaseTokenizer):
    """
    A Unigram language model for tokenization.

    This implements the Unigram algorithm described in:
    Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation
    Models with Multiple Subword Candidates. https://arxiv.org/abs/1804.10959
    """

    VERSION = "seunigram-v1"
    REPORT_TITLE = "Unigram Tokenizer Report"

    def __init__(self, pretokenizer: Pretokenizer, tokens: list[UnigramToken], metadata: dict | None = None):
        """
        Initialize the Unigram model.

        Args:
                pretokenizer: The pretokenizer to use for encoding/decoding
                tokens: List of UnigramToken objects (optional)
                metadata: Additional metadata to store with the model
        """
        self.pretokenizer = pretokenizer
        # Canonical store: dict[int, UnigramToken]
        self.tokens = {t.id: t for t in (tokens or [])}
        self.trie = Trie(tokens)
        self.metadata = metadata or {}
        self.tokens_by_id = self.tokens

    def make_lattice(self, atomic_token_seq: TokenSeq, token_bias: float = 0.0) -> Lattice:
        tokens_from_pos = [self.trie.find_prefixes(atomic_token_seq[i:]) for i in range(len(atomic_token_seq))]
        return Lattice(atomic_token_seq, tokens_from_pos, token_bias)

    def encode(self, text: str, return_tokens=False) -> list[UnigramToken] | list[int]:
        # Flatten pretokenized chunks into a single atomic token sequence
        chunks = self.pretokenizer.pretokenize(text)
        atomic_token_seq = token_array([tid for chunk in chunks for tid in chunk])
        lattice = self.make_lattice(atomic_token_seq)
        tokens = lattice.viterbi()[0]
        if return_tokens:
            return tokens
        else:
            return [token.id for token in tokens]

    def decode(self, ids: InputTokenSeq) -> str:
        return self.pretokenizer.decode([tid for token_id in ids for tid in self.tokens_by_id[token_id].atomic_tokens])

    @classmethod
    def load(cls, file: str, reindex: bool = False):
        """Load a UnigramModel from a file.

        Args:
                file: Path to the model file (.json or .json.gz)
                reindex: When False, preserve serialized token ids exactly.
                    When True, rewrite model token ids to dense ``0..n-1`` in
                    serialized token-list order. This does not modify
                    ``atomic_tokens``, which remain aligned with the serialized
                    pretokenizer's base-token ids.

        Returns:
                UnigramModel: The loaded model
        """
        open_func = gzip.open if file.endswith(".gz") else open
        with open_func(file, "rt") as f:
            data = json.load(f)

        pretokenizer = load_pretokenizer(data["pretokenizer"])
        tokens = [UnigramToken(**t) for t in data["tokens"]]
        if reindex:
            tokens = reindex_tokens(tokens)
        return cls(pretokenizer=pretokenizer, tokens=tokens, metadata=data.get("metadata"))

    def save(self, file_path: str) -> str:
        """Save the model to a file.

        Args:
                file_path: Path to save the model to (.json or .json.gz)

        Returns:
                str: The path the model was saved to
        """
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

    def stats(self, n_longest=20) -> dict:
        base_stats = super().stats(n_longest=n_longest)
        # lowest probability among non-required tokens
        probs = [math.exp(t.log_prob) for t in self.tokens.values() if not t.required]
        base_stats["lowest_prob"] = min(probs) if probs else 0.0
        return base_stats

    def report_details(self, n_longest=20) -> dict[str, list[dict]]:
        # Non-base tokens by probability
        tokens_with_prob = [
            {
                "ID": token.id,
                "Probability": f"{math.exp(token.log_prob):.6g}",
                "Log Probability": f"{token.log_prob:.4f}",
                "Text": repr(self.pretokenizer.tokens_repr(token.atomic_tokens)),
            }
            for token in sorted(self.tokens.values(), key=lambda t: -t.log_prob)
            if not token.required
        ]

        # Metadata table (if any)
        metadata_rows: list[dict] = []
        if self.metadata and len(self.metadata) > 0:
            for k, v in self.metadata.items():
                if k != "tokens":
                    metadata_rows.append({"Key": k, "Value": v})

        sections: dict[str, list[dict]] = {
            "Non-base tokens by probability": tokens_with_prob,
        }
        if metadata_rows:
            sections["Metadata"] = metadata_rows
        return sections
