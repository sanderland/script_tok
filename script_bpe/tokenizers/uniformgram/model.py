import gzip
import json
import os
from dataclasses import dataclass

from script_bpe.pretokenize import Pretokenizer, export_pretokenizer, load_pretokenizer
from script_bpe.utils import InputTokenSeq, TokenSeq, token_array
from script_bpe.tokenizers import BaseToken
from script_bpe.tokenizers.base import BaseTokenizer

# Reuse Trie and Lattice from Unigram (they work generically with any tokens that have log_prob)
from script_bpe.tokenizers.unigram.model import Trie, Lattice


@dataclass(slots=True)
class UniformGramToken(BaseToken):
    """
    Token for UniformGram model.
    All tokens have uniform probability (log_prob = 0.0), so Viterbi naturally
    selects the segmentation with the fewest tokens (longest available tokens).
    """

    log_prob: float = 0.0
    required: bool = False

    def to_dict(self):
        """Convert the token to a dictionary for serialization."""
        return {
            "id": self.id,
            "atomic_tokens": list(self.atomic_tokens),
            "log_prob": self.log_prob,
        }


class UniformGramModel(BaseTokenizer):
    """
    A UniformGram model for tokenization.

    Unlike Unigram, this model assigns uniform probabilities to all tokens (log_prob = 0.0).
    This means Viterbi decoding always selects the segmentation with the fewest tokens,
    effectively greedily choosing the longest available tokens.

    Training uses iterative Viterbi counting and pruning rather than EM:
    1. Initialize large vocabulary
    2. Count token usage in Viterbi segmentations of corpus
    3. Prune tokens with low Viterbi counts
    4. Repeat until target vocabulary size is reached
    """

    VERSION = "uniformgram-v1"
    REPORT_TITLE = "UniformGram Tokenizer Report"

    def __init__(self, pretokenizer: Pretokenizer, tokens: list[UniformGramToken], metadata: dict | None = None):
        """
        Initialize the UniformGram model.

        Args:
            pretokenizer: The pretokenizer to use for encoding/decoding
            tokens: List of UniformGramToken objects
            metadata: Additional metadata to store with the model
        """
        self.pretokenizer = pretokenizer
        # Canonical store: dict[int, UniformGramToken]
        self.tokens = {t.id: t for t in (tokens or [])}
        self.trie = Trie(tokens)
        self.metadata = metadata or {}
        self.tokens_by_id = self.tokens

    def make_lattice(self, atomic_token_seq: TokenSeq) -> Lattice:
        """Build a lattice for the given atomic token sequence."""
        tokens_from_pos = [self.trie.find_prefixes(atomic_token_seq[i:]) for i in range(len(atomic_token_seq))]
        return Lattice(atomic_token_seq, tokens_from_pos)

    def encode(self, text: str, return_tokens=False) -> list[UniformGramToken] | list[int]:
        """
        Encode text into token IDs using Viterbi decoding.

        With uniform probabilities (all log_prob = 0.0), this greedily selects
        the longest available tokens (fewest total tokens).

        Args:
            text: The text to encode
            return_tokens: If True, return UniformGramToken objects instead of IDs

        Returns:
            List of token IDs or UniformGramToken objects
        """
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
        """Decode token IDs back to text."""
        return self.pretokenizer.decode([tid for token_id in ids for tid in self.tokens_by_id[token_id].atomic_tokens])

    @classmethod
    def load(cls, file):
        """Load a UniformGramModel from a file.

        Args:
            file: Path to the model file (.json or .json.gz)

        Returns:
            UniformGramModel: The loaded model
        """
        open_func = gzip.open if file.endswith(".gz") else open
        with open_func(file, "rt") as f:
            data = json.load(f)

        pretokenizer = load_pretokenizer(data["pretokenizer"])
        tokens = [UniformGramToken(**t) for t in data["tokens"]]
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
        """Get statistics about the tokenizer."""
        return super().stats(n_longest=n_longest)

    def report_details(self, n_longest=20) -> dict[str, list[dict]]:
        """Get detailed report sections for the tokenizer."""
        # Non-base tokens (all have uniform probability, so just show them by ID)
        tokens_list = [
            {
                "ID": token.id,
                "Text": repr(self.pretokenizer.tokens_repr(token.atomic_tokens)),
                "Length": len(token.atomic_tokens),
            }
            for token in sorted(self.tokens.values(), key=lambda t: t.id)
            if not token.required
        ]

        # Metadata table (if any)
        metadata_rows: list[dict] = []
        if self.metadata and len(self.metadata) > 0:
            for k, v in self.metadata.items():
                if k != "tokens":
                    metadata_rows.append({"Key": k, "Value": v})

        sections: dict[str, list[dict]] = {
            "Non-base tokens": tokens_list,
        }
        if metadata_rows:
            sections["Metadata"] = metadata_rows
        return sections
