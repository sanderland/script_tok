from pydantic import BaseModel
from abc import abstractmethod
import hashlib
from typing import Literal
import regex as re
import itertools
import unicodedata

from script_bpe.utils import InputTokenSeq, PretokenizedT
from script_bpe.pretokenize.scriptenc import ScriptConfig, ScriptEncodingV1, ScriptBlock

DigitHandlingT = Literal["RTL3", "SPLIT"] | None
TokenPairT = tuple[int, int]


class PretokenizerConfig(BaseModel):
    starting_token_id: int = 1  # 0 reserved for pad
    script_config: ScriptConfig = ScriptEncodingV1
    # normalization
    normalization: str | None = "NFC"
    remove_unassigned: bool = True
    # splitting
    regex_pattern: str | None = None
    digit_handling: DigitHandlingT = None
    # token restrictions
    enforce_char_boundaries: bool = True


class UTF8PretokenizerConfig(PretokenizerConfig):
    pass


class ScriptPretokenizerConfig(PretokenizerConfig):
    script_split: bool = True


class ScriptCharEnc:
    __slots__ = ("script_id", "combines_with_spaces", "atomic_token_ids", "inherited")

    def __init__(self, block: ScriptBlock, token_pair: TokenPairT):
        self.script_id = block.script_id
        self.combines_with_spaces = block.combines_with_spaces
        self.atomic_token_ids = token_pair
        self.inherited = block.script == "Inherited"

    def __repr__(self):
        return f"ScriptCharEnc(script_id={self.script_id}, atomic_token_ids={self.atomic_token_ids})"


class UTF8CharEnc:
    __slots__ = "atomic_token_ids"

    def __init__(self, byte_token: int):
        self.atomic_token_ids = [byte_token]

    def __repr__(self):
        return f"UTF8CharEnc(atomic_token_ids={self.atomic_token_ids})"


CharEncT = ScriptCharEnc | UTF8CharEnc


def group_digits(digits: str, method: DigitHandlingT) -> list[str]:
    l = len(digits)
    if method == "RTL3":
        return [digits[i : i + 3] for i in range(l % 3, l, 3)]
    elif method == "SPLIT":
        return list(digits)


class Pretokenizer:
    def __init__(self, config: PretokenizerConfig) -> None:
        self.config = config
        if self.config.remove_unassigned:
            self.valid_chars = {c for block in self.config.script_config.blocks for c in block.chars}
            assert self.valid_chars, "No valid chars found, check script config"
        self.atomic_tokens: dict[int, str] = {}  # not to be confused with kittens
        self.token_to_id: dict[str, int] = {}
        self._next_token_id = self.config.starting_token_id
        self._build_atomic_tokens()
        if self.config.digit_handling is not None:
            self._build_digit_tokens()

    def _register_token(self, text) -> int:
        self.atomic_tokens[self._next_token_id] = text
        self.token_to_id[text] = self._next_token_id
        self._next_token_id += 1
        return self._next_token_id - 1

    def _build_atomic_tokens(self):
        raise NotImplementedError

    def _build_digit_tokens(self):
        if self.config.digit_handling == "SPLIT":
            for d in range(10):
                self._register_token(str(d))
        elif self.config.digit_handling == "RTL3":
            for range, pad in [(1000, 3), (100, 2), (10, 1)]:
                for i in range(range):
                    self._register_token(str(i).zfill(pad))

    def hash(self) -> str:
        hash = hashlib.sha1()
        hash.update(str(self.config).encode("utf-8"))
        return "PT-" + hash.hexdigest()[:8]

    def pretokenize(self, text: str) -> PretokenizedT:
        text = self.normalize(text)
        encoded_chunks = self.split_unencoded_and_encode(text)
        encoded_chunks = [subchunk for chunk in encoded_chunks for subchunk in self.split_encoded(chunk)]
        chunks = [[tid for chr in chunk for tid in chr.atomic_token_ids] for chunk in encoded_chunks]
        return chunks

    @abstractmethod
    def decode(self, token_ids: InputTokenSeq, errors: str = "replace") -> str:
        """decodes a list of base tokens back to string"""
        raise NotImplementedError

    def encode_text(self, text: str) -> list[CharEncT]:
        """encodes a string to a list of base tokens, without any other pretokenization"""
        raise NotImplementedError

    def normalize(self, text: str) -> str:
        """Normalize and remove unassigned private surrogates"""
        if self.config.normalization is not None:
            text = unicodedata.normalize(self.config.normalization, text)
        if self.config.remove_unassigned:
            text = "".join(c for c in text if c in self.valid_chars)
        return text

    def split_unencoded_and_encode(self, text: str) -> list[str]:
        """1. Maybe split off digits, based on method"""
        if self.config.digit_handling is None:
            encoded_chunks = [text]  # is even, so works
        else:
            encoded_chunks = re.split("([0-9]+)", text)
        for i in range(len(encoded_chunks)):
            if i % 2 == 0 or self.config.digit_handling is None:  # non-digits
                encoded_chunks[i] = [self.encode_text(c) for c in self.regex_split(encoded_chunks[i])]
            else:  # only digits
                encoded_chunks[i] = [
                    self.encode_digits(c) for c in group_digits(encoded_chunks[i], self.config.digit_handling)
                ]

        return [g for chunk in encoded_chunks for g in chunk]

    def regex_split(self, text: str) -> list[str]:
        if self.config.regex_pattern is None:
            return [text]
        return re.findall(self.config.regex_pattern, text)

    def split_encoded(self, text_chunks: list[CharEncT]) -> list[list[CharEncT]]:
        return [text_chunks]

    def encode_digits(self, digit_groups: list[str]) -> list[int]:
        return [self.token_to_id[dgroup] for dgroup in digit_groups]

    def token_repr(self, base_token_ids: InputTokenSeq) -> str:
        """Representation that is able to handle partial/broken sequences"""
        return self.decode(base_token_ids, errors="backslashreplace")

    def token_allowed(self, token_seq: InputTokenSeq) -> bool:
        return True


class UTF8Pretokenizer(Pretokenizer):
    def __init__(self, config: UTF8PretokenizerConfig) -> None:
        super().__init__(config)

    def _build_atomic_tokens(self):
        self.byte_ids = {b: self._register_token(f"<BYTE_{b:02X}>") for b in range(256)}
        self.token_to_byte = {tid: b for b, tid in self.byte_ids.items()}

    def decode(self, base_token_ids: InputTokenSeq, errors="replace") -> str:
        byteseq = bytes([self.token_to_byte[id] for id in base_token_ids])
        return byteseq.decode("utf-8", errors=errors)

    def encode_text(self, text: str) -> list[UTF8CharEnc]:
        return [UTF8CharEnc(self.byte_ids[c]) for c in text.encode("utf-8")]


class ScriptPretokenizer(Pretokenizer):
    def __init__(self, config: ScriptPretokenizerConfig) -> None:
        if config.script_split and config.regex_pattern is not None:
            raise ValueError("script_split and regex_pattern should probably not be used together")
        super().__init__(config)
        self.space_group = self.encode_text(" ")

    def _build_atomic_tokens(self):
        self.num_index_tokens = max(len(block.chars) for block in self.config.script_config.blocks)
        self.detokenize_map = {}
        self.char_encoding = {}
        index_tokens = [self._register_token(f"<|SCRIPT_INDEX_{i}|>") for i in range(self.num_index_tokens)]
        for block in self.config.script_config.blocks:
            block_token = self._register_token(f"<|BLOCK_{block.script}_{block.category}_{block.sub_block_id}|>")
            for i, c in enumerate(block.chars):
                token_pair = (block_token, index_tokens[i])
                self.detokenize_map[token_pair] = c
                self.char_encoding[c] = ScriptCharEnc(block, token_pair)

    def decode(self, tokenization: InputTokenSeq, errors="replace") -> str:
        decoded = ""
        i = 0
        while i < len(tokenization):
            script_tok = tokenization[i]
            ix_tok = tokenization[i + 1] if i + 1 < len(tokenization) else None
            if (script_tok, ix_tok) in self.detokenize_map:
                decoded += self.detokenize_map[(script_tok, ix_tok)]
                i += 2
            else:
                if errors == "backslashreplace":  # not backslash, but compatible with bytes version
                    decoded += self.base_tokens[script_tok]
                elif errors == "replace":
                    decoded += "�"
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok}) is not a valid token pair!")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        return decoded

    def encode_text(self, text: str) -> list[ScriptCharEnc]:
        return [self.char_encoding[c] for c in text]

    def split_encoded(self, encoding: list[ScriptCharEnc]) -> list[list[ScriptCharEnc]]:
        """
        Pretokenize the encoding by grouping adjacent tokens with the same script.
        Special cases:
        - If a script combines with spaces, it will be merged with the space group.
        - Inherited scripts are merged with the previous group, and do not split the group.
        """
        if not self.config.script_split:
            return [encoding]
        space_group = self.space_group
        script_groups = [list(g) for _, g in itertools.groupby(encoding, key=lambda x: x.script_id)]

        merged_groups = []
        i = 0
        while i < len(script_groups) - 1:
            current_group, next_group = script_groups[i], script_groups[i + 1]
            current_script, next_script = current_group[0].script_id, next_group[0].script_id
            if next_group[0].combines_with_spaces and current_group == space_group:
                merged_group = current_group + next_group
                current_script = next_script
                i += 2
            else:
                merged_group = current_group
                i += 1
            while i < len(script_groups) and (
                script_groups[i][0].inherited or script_groups[i][0].script_id == current_script
            ):
                merged_group += script_groups[i]
                i += 1
            merged_groups.append(merged_group)
        if i < len(script_groups):
            merged_groups.append(script_groups[i])  # add last group
        return merged_groups


if __name__ == "__main__":
    GPT2_REGEX = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ptoks = [
        UTF8Pretokenizer(UTF8PretokenizerConfig(regex_pattern=GPT2_REGEX)),
        ScriptPretokenizer(ScriptPretokenizerConfig()),
    ]
    for ptok in ptoks:
        print(ptok.__class__)
        pretokens = ptok.pretokenize("hello world")
        for pt in pretokens:
            print(f"{pt} -> {ptok.decode(pt)!r}")
