import functools
import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Any

import regex as re

from script_bpe.utils import (
    InputTokenSeq,
    PretokenizedT,
    TokenSeq,
    token_array,
    utf_byte_type,
)

from ..pretokenize.base import BasePretokenizer


class RegexBytePretokenizer(BasePretokenizer):
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the RegexBytePretokenizer with a configuration.
        :param config: A dictionary containing 'starting_token_id', 'normalization', and 'regex'.
        """
        super().__init__(config)
        self.regex = re.compile(config["regex"], re.DOTALL)  # No default, must be provided in config
        self.enforce_char_boundaries: bool = config["enforce_char_boundaries"]
        self.base_tokens.update({i + self.starting_token_id: f"<|BYTE_{i}|>" for i in range(256)})

    def hash(self) -> str:  # char_boundaries does not affect corpus, so no in hash
        hash = hashlib.sha1()
        hash.update(super().hash().encode("utf-8"))
        hash.update(str(self.config["regex"]).encode("utf-8"))
        return "Regex-" + hash.hexdigest()[:8]

    def _encode_and_chunk(self, text: str) -> PretokenizedT:
        """Tokenize the input text into UTF-8 byte chunks based on the regex."""
        chunks = re.findall(self.regex, text)
        byte_tokens = [
            token_array([self.starting_token_id + byte for byte in chunk.encode("utf-8")]) for chunk in chunks if chunk
        ]
        return byte_tokens

    def _to_bytes(self, base_token_ids: InputTokenSeq) -> bytes:
        return bytes([id - self.starting_token_id for id in base_token_ids])

    def decode(self, base_token_ids: InputTokenSeq, errors="replace") -> str:
        return self._to_bytes(base_token_ids).decode("utf-8", errors=errors)

    def _is_partial_seq(self, seq: InputTokenSeq) -> tuple[int, bool]:
        if len(seq) > 4:
            return 1, False  # multiple full chars
        bytes = self._to_bytes(seq)
        first_byte_type = utf_byte_type(bytes[0])
        if first_byte_type == 0:
            return first_byte_type, True  # continuation byte
        if sum(utf_byte_type(b) in {1, 2, 3, 4} for b in bytes) > 1:
            return first_byte_type, False
        return first_byte_type, len(seq) != first_byte_type

    def bpe_merge_allowed(self, token_seq1: InputTokenSeq, token_seq2: InputTokenSeq) -> bool:
        if not self.enforce_char_boundaries:
            return True
        type1, partial_seq1 = self._is_partial_seq(token_seq1)
        type2, partial_seq2 = self._is_partial_seq(token_seq2)
        if partial_seq1 and partial_seq2:
            return type1 != 0 and type2 == 0  # starting byte, continuation byte
        elif not partial_seq1 and not partial_seq2:
            return True  # both are full sequences
        return False

    def token_allowed(self, token_seq: InputTokenSeq) -> bool:
        if not self.enforce_char_boundaries:
            return True
        try:
            self.decode(token_seq, errors="strict")
            return True
        except UnicodeDecodeError: # should be a single char or prefix, so [not-cont] + [cont,...]
            bytes = self._to_bytes(token_seq)
            return utf_byte_type(bytes[0]) != 0 and not any(utf_byte_type(b)==0 for b in bytes[1:])
            
            
