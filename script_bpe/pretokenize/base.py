import copy
import hashlib
import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from script_bpe.utils import (
    InputTokenSeq,
    PretokenizedT,
    remove_unassigned_private_surrogate,
)

class BasePretokenizer(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the BasePretokenizer with a configuration.
        :param config: A dictionary containing 'starting_token_id' and 'normalization' (default 'NFC').
        """
        self.config = copy.deepcopy(config)
        self.starting_token_id: int = config.get("starting_token_id", 1)
        self.remove_unassigned = True
        self.normalization: str = config.get("normalization", "NFC")
        if self.normalization not in ["NFC", "NFD", "NFKC", "NFKD", None]:
            raise ValueError(f"Invalid normalization form: {self.normalization}")
        self.base_tokens: dict[int, str] = {}  # Dictionary mapping token IDs to strings

    def hash(self) -> str:
        hash = hashlib.sha1()
        hash.update(str(self.starting_token_id).encode("utf-8"))
        hash.update(str(self.normalization).encode("utf-8"))
        hash.update(str(self.remove_unassigned).encode("utf-8"))
        return "Base" + hash.hexdigest()[:8]

    def encode_and_chunk(self, text: str) -> PretokenizedT:
        """normalize -> chunk -> encode"""
        return self._encode_and_chunk(self.normalize(text))

    @abstractmethod
    def _encode_and_chunk(self, text: str) -> PretokenizedT:
        """abstract method for encoding and chunking, to be implemented by subclasses"""
        pass

    @abstractmethod
    def decode(self, token_ids: InputTokenSeq, errors: str = "replace") -> str:
        """decodes a list of base tokens back to string"""
        pass

    def chunk(self, text: str) -> list[str]:
        """Returns chunks as strings, mainly for debugging"""
        return [self.decode(c) for c in self.encode_and_chunk(text)]

    def tokens_to_readable_string(self, base_token_ids: InputTokenSeq) -> str:
        """Representation that is able to handle partial/broken sequences"""
        return self.decode(base_token_ids, errors="backslashreplace")

    def normalize(self, text: str) -> str:
        """
        Normalize the input text using Unicode normalization.
        :param text: The input string to normalize.
        :return: The normalized string.
        """
        if self.remove_unassigned:
            text = remove_unassigned_private_surrogate(text)
        if self.normalization is not None:
            text = unicodedata.normalize(self.normalization, text)
        return text

    def bpe_merge_allowed(self, token_seq1: InputTokenSeq, token_seq2: InputTokenSeq) -> bool:
        """In BPE training, can token_seq1 and token_seq2 be merged?"""
        return True

    def token_allowed(self, token_seq: InputTokenSeq) -> bool:
        return True
