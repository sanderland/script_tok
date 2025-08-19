import gzip
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import tabulate

from script_bpe.pretokenize import Pretokenizer, export_pretokenizer, make_pretokenizer
from script_bpe.utils import TokenSeq


def logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)) for two finite terms."""
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


class UnigramToken:
    def __init__(self, id: int, base_tokens: TokenSeq, log_prob: float, required: bool = False, **kwargs):
        self.id = id
        self.base_tokens = base_tokens
        self.log_prob = log_prob
        self.required = required

    def to_dict(self):
        """Convert the token to a dictionary for serialization."""
        return {
            "id": self.id,
            "base_tokens": list(self.base_tokens),
            "log_prob": self.log_prob,
        }


class Trie:
    def __init__(self, tokens: list[UnigramToken]):
        self.root = {}
        for token in tokens:
            self.insert(token)

    def insert(self, token: UnigramToken):
        node = self.root
        for tid in token.base_tokens:
            node = node.setdefault(tid, {})
        node[None] = token

    def find_prefixes(self, base_token_seq: TokenSeq) -> list[UnigramToken]:
        results = []
        node = self.root
        for tid in base_token_seq:
            if tid not in node:
                break
            node = node[tid]
            if None in node:
                results.append(node[None])
        return results


class Lattice:
    def __init__(self, base_token_seq: TokenSeq, tokens_from_pos: list[list[UnigramToken]]):
        self.base_token_seq = base_token_seq
        self.tokens_from_pos = tokens_from_pos

    def viterbi(self, allow_single_token=True) -> tuple[list[UnigramToken], float]:
        best_at_pos = [(None, 0)] + [(None, float("-inf"))] * len(self.base_token_seq)
        for pos in range(len(self.base_token_seq)):
            for token in self.tokens_from_pos[pos]:
                end = pos + len(token.base_tokens)
                if pos == 0 and not allow_single_token and end == len(self.base_token_seq):
                    continue  # do not allow direct path
                score = best_at_pos[pos][1] + token.log_prob
                if score > best_at_pos[end][1]:
                    best_at_pos[end] = (token, score)

        path = []
        pos = len(self.base_token_seq)
        while pos > 0:
            token, score = best_at_pos[pos]
            if token is None:
                break
            path.append(token)
            pos -= len(token.base_tokens)
        return path[::-1], best_at_pos[-1][1]

    def all_paths(self, starting_pos: int = 0) -> Iterable[tuple[tuple[UnigramToken], float]]:
        if starting_pos == len(self.base_token_seq):
            yield (tuple(), 0.0)
            return
        for token in self.tokens_from_pos[starting_pos]:
            for sub_path, sub_prob in self.all_paths(starting_pos + len(token.base_tokens)):
                yield ((token,) + sub_path, sub_prob + token.log_prob)

    def _forward_backward(self) -> tuple[list[float], list[float]]:
        """returns
        alpha(pos) = total prob of path to pos
        beta(pos) = total prob of path from pos to end
        """
        alpha = [0] + [float("-inf")] * len(self.base_token_seq)
        beta = [float("-inf")] * (len(self.base_token_seq)) + [0]

        for pos in range(len(self.base_token_seq)):
            if alpha[pos] != float("-inf"):
                for token in self.tokens_from_pos[pos]:
                    end = pos + len(token.base_tokens)
                    alpha[end] = logaddexp(alpha[end], alpha[pos] + token.log_prob)

        for pos in range(len(self.base_token_seq) - 1, -1, -1):
            for token in self.tokens_from_pos[pos]:
                end = pos + len(token.base_tokens)
                if beta[end] != float("-inf"):
                    beta[pos] = logaddexp(beta[pos], beta[end] + token.log_prob)
        return alpha, beta

    def calc_marginal(self) -> tuple[float, dict[int, float]]:
        alpha, beta = self._forward_backward()
        z = alpha[-1]
        assert z != float("-inf"), (
            f"Lattice for {self.base_token_seq!r} has no valid paths with tokens_from_pos {self.tokens_from_pos}"
        )
        token_prob = defaultdict(float)
        for pos in range(len(self.base_token_seq)):
            for token in self.tokens_from_pos[pos]:
                token_logprob = alpha[pos] + token.log_prob + beta[pos + len(token.base_tokens)] - z
                token_prob[token.id] += math.exp(max(-100, token_logprob))  # Avoid underflow

        return z, token_prob


class UnigramModel:
    """
    A Unigram language model for tokenization.

    This implements the Unigram algorithm described in:
    Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation
    Models with Multiple Subword Candidates. https://arxiv.org/abs/1804.10959
    """

    VERSION = "seunigram-v1"

    def __init__(self, pretokenizer: Pretokenizer, tokens: list[UnigramToken], metadata: dict = None):
        """
        Initialize the Unigram model.

        Args:
            pretokenizer: The pretokenizer to use for encoding/decoding
            tokens: List of UnigramToken objects (optional)
            metadata: Additional metadata to store with the model
        """
        self.pretokenizer = pretokenizer
        self.tokens = tokens or []
        self.trie = Trie(self.tokens)
        self.metadata = metadata or {}
        self.tokens_by_id = {t.id: t for t in self.tokens} if self.tokens else {}

    def make_lattice(self, base_token_seq: TokenSeq) -> Lattice:
        tokens_from_pos = [self.trie.find_prefixes(base_token_seq[i:]) for i in range(len(base_token_seq))]
        return Lattice(base_token_seq, tokens_from_pos)

    def encode(self, text: str, return_tokens=False) -> list[UnigramToken] | list[int]:
        base_token_seq = self.pretokenizer.encode(text)
        lattice = self.make_lattice(base_token_seq)
        tokens = lattice.viterbi()[0]
        if return_tokens:
            return tokens
        else:
            return [token.id for token in tokens]

    def decode(self, ids: list[int]) -> str:
        return self.pretokenizer.decode([tid for token_id in ids for tid in self.tokens_by_id[token_id].base_tokens])

    @classmethod
    def load(cls, file):
        """Load a UnigramModel from a file.

        Args:
            file: Path to the model file (.json or .json.gz)

        Returns:
            UnigramModel: The loaded model
        """
        open_func = gzip.open if file.endswith(".gz") else open
        with open_func(file, "rt") as f:
            data = json.load(f)

        pretokenizer = make_pretokenizer(data["pretokenizer"])
        tokens = [UnigramToken(**t) for t in data["tokens"]]
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
                    "tokens": [t.to_dict() for t in self.tokens],
                    "metadata": self.metadata,
                },
                f,
                indent=2,
            )
        return file_path

    def stats(self, n_longest=20) -> dict:
        """Compute and return statistics about the Unigram tokenizer.

        Returns:
            dict: Dictionary containing various statistics about the tokenizer
        """
        # Basic counts
        num_tokens = len(self.tokens)
        base_tokens = [t for t in self.tokens if len(t.base_tokens) == 1]
        multi_token = [t for t in self.tokens if len(t.base_tokens) > 1]

        # Token lengths
        token_lengths = [len(t.base_tokens) for t in self.tokens]
        char_lengths = [len(self.decode([t.id])) for t in self.tokens]

        # Find undecodable tokens
        is_undecodable = {
            t.id: "�" in self.pretokenizer.decode(t.base_tokens, errors="replace")
            for t in self.tokens
            if not t.required
        }

        return {
            # Basic counts
            "num_tokens": num_tokens,
            "num_base_tokens": len(base_tokens),
            "num_multi_tokens": len(multi_token),
            "num_undecodable": sum(is_undecodable.values()),
            # Length statistics
            "avg_token_length_bt": sum(token_lengths) / num_tokens if num_tokens > 0 else 0,
            "avg_char_length": sum(char_lengths) / num_tokens if num_tokens > 0 else 0,
            # Longest tokens for reporting
            "longest_tokens": sorted(self.tokens, key=lambda t: -len(t.base_tokens))[:n_longest],
        }

    def report(self, n_longest=20) -> str:
        """Generate a markdown formatted report about the tokenizer."""
        stats = self.stats(n_longest=n_longest)

        report = [
            "# Unigram Tokenizer Report",
            "",
            "## Model Overview",
            f"- **Total tokens:** {stats['num_tokens']:,d}",
            f"- **Base tokens (single token):** {stats['num_base_tokens']:,d}",
            f"- **Multi-token sequences:** {stats['num_multi_tokens']:,d}",
            f"- **Undecodable tokens:** {stats['num_undecodable']:,d}",
            "",
            "## Token Length Statistics",
            f"- **Average base tokens per token:** {stats['avg_token_length_bt']:.4f}",
            f"- **Average characters per token:** {stats['avg_char_length']:.4f}",
            "",
        ]

        # Add longest tokens section
        longest_tokens_table = []
        for token in stats["longest_tokens"]:
            token_str = self.pretokenizer.tokens_to_readable_string(token.base_tokens)
            longest_tokens_table.append(
                {
                    "ID": token.id,
                    "Base Tokens": len(token.base_tokens),
                    "Log Prob": f"{token.log_prob:.3f}",
                    "Text": repr(token_str),
                }
            )

        report.extend(
            [
                "## Longest Tokens (by base tokens)",
                "",
                tabulate.tabulate(longest_tokens_table, headers="keys", tablefmt="github"),
                "",
            ]
        )

        # Add non-base tokens by probability
        tokens_with_prob = [
            {
                "ID": token.id,
                "Log Probability": f"{token.log_prob:.4f}",
                "Text": repr(self.pretokenizer.tokens_to_readable_string(token.base_tokens)),
            }
            for token in sorted(self.tokens, key=lambda t: -t.log_prob)
            if not token.required
        ]

        report.extend(
            [
                "## Non-base tokens by probability",
                "",
                tabulate.tabulate(tokens_with_prob, headers="keys", tablefmt="github"),
                "",
            ]
        )

        # Add metadata section if available
        if self.metadata and len(self.metadata) > 0:
            metadata_items = [[k, v] for k, v in self.metadata.items() if k != "tokens"]
            if metadata_items:
                report.extend(
                    [
                        "## Metadata",
                        "",
                        tabulate.tabulate(metadata_items, headers=["Key", "Value"], tablefmt="github"),
                        "",
                    ]
                )

        return "\n".join(report)
