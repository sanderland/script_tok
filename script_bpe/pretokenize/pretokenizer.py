from pydantic import BaseModel, ConfigDict
from abc import abstractmethod
import hashlib
from typing import Sequence, Literal
import regex as re
import itertools
import unicodedata
import json

from script_bpe.utils import InputTokenSeq, PretokenizedT, token_array, DigitHandlingT, TokenPairT
from script_bpe.pretokenize.scriptencoding import ScriptConfig, ScriptEncodingV1, ScriptBlock, unicode_script_map

# consolidated in utils


class PretokenizerConfig(BaseModel):
    cls: str
    starting_token_id: int = 1  # 0 reserved for pad
    script_config: ScriptConfig = ScriptEncodingV1
    # normalization
    normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] | None = "NFC"
    remove_unassigned: bool = True
    # splitting
    regex_pattern: str | None = None
    digit_handling: DigitHandlingT = None
    # token restrictions
    enforce_char_boundaries: bool = True
    enforce_inherited: bool = False
    # disallow extra fields
    model_config = ConfigDict(extra="forbid")


class UTF8PretokenizerConfig(PretokenizerConfig):
    cls: str = "UTF8Pretokenizer"


class ScriptPretokenizerConfig(PretokenizerConfig):
    cls: str = "ScriptPretokenizer"
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


class DigitsEnc:
    __slots__ = ("script_id", "combines_with_spaces", "atomic_token_ids", "inherited")

    def __init__(self, token_ids: list[int]):
        # Sentinel script id to ensure it forms its own group in script-based splitting
        self.script_id = -1
        self.combines_with_spaces = False
        self.inherited = False
        self.atomic_token_ids = token_ids

    def __repr__(self):
        return f"DigitsEnc(atomic_token_ids={self.atomic_token_ids})"


CharEncT = ScriptCharEnc | UTF8CharEnc | DigitsEnc


def group_digits(digits: str, method: DigitHandlingT) -> list[str]:
    num_digits = len(digits)
    if method == "RTL3":
        # Group digits from the right in chunks of 3, keeping a possible leading remainder
        remainder = num_digits % 3
        groups: list[str] = []
        if remainder:
            groups.append(digits[:remainder])
        groups.extend(digits[i : i + 3] for i in range(remainder, num_digits, 3))
        return groups
    elif method == "SPLIT":
        return list(digits)
    else:
        return [digits]


class Pretokenizer:
    REGISTRY: dict[str, tuple[type[PretokenizerConfig], type["Pretokenizer"]]] = {}

    def __init_subclass__(cls, config_type: type[PretokenizerConfig]) -> None:
        cls.REGISTRY[cls.__name__] = (config_type, cls)
        super().__init_subclass__()

    def __init__(self, config: PretokenizerConfig) -> None:
        self.config = config
        if self.config.remove_unassigned:
            self.valid_chars = {c for block in self.config.script_config.blocks for c in block.chars}
            assert self.valid_chars, "No valid chars found, check script config"
        self.atomic_tokens: dict[int, str] = {}  # not to be confused with kittens
        self.token_to_id: dict[str, int] = {}
        self._next_token_id = self.config.starting_token_id
        self.is_initial_char_tokens: set[int] = set()
        self._build_atomic_tokens()
        if self.config.digit_handling is not None:
            self._build_digit_tokens()
        self.inherited_chars: set[str] = (
            {c for c, info in unicode_script_map().items() if info["script"] == "Inherited"}
            if self.config.enforce_inherited
            else set()
        )

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
            for upper, pad in [(1000, 3), (100, 2), (10, 1)]:
                for i in range(upper):
                    self._register_token(str(i).zfill(pad))

    def hash(self) -> str:
        h = hashlib.sha1()
        config = self.config.model_copy(deep=True)
        h.update(json.dumps(config.model_dump(mode="json", by_alias=True), sort_keys=True).encode("utf-8"))
        return "PT-" + h.hexdigest()[:8]

    def pretokenize(self, text: str) -> PretokenizedT:
        text = self.normalize(text)
        encoded_chunks = self.split_unencoded_and_encode(text)
        encoded_chunks = [subchunk for chunk in encoded_chunks for subchunk in self.split_encoded(chunk)]
        chunks = [token_array([tid for chr in chunk for tid in chr.atomic_token_ids]) for chunk in encoded_chunks]
        return chunks

    @abstractmethod
    def decode(self, token_ids: InputTokenSeq, errors: str = "replace") -> str:
        """decodes a list of base tokens back to string"""
        raise NotImplementedError

    def encode_text(self, text: str) -> Sequence[CharEncT]:
        """encodes a string to a list of base tokens, without any other pretokenization"""
        raise NotImplementedError

    def normalize(self, text: str) -> str:
        """Normalize and remove unassigned private surrogates"""
        if self.config.normalization is not None:
            text = unicodedata.normalize(self.config.normalization, text)
        if self.config.remove_unassigned:
            text = "".join(c for c in text if c in self.valid_chars)
        return text

    def split_unencoded_and_encode(self, text: str) -> list[Sequence[CharEncT]]:
        """1. Maybe split off digits, based on method"""
        parts = [text] if self.config.digit_handling is None else re.split("([0-9]+)", text)
        output_chunks: list[Sequence[CharEncT]] = []
        for i, part in enumerate(parts):
            if i % 2 == 0 or self.config.digit_handling is None:  # non-digits
                for token in self.regex_split(part):
                    output_chunks.append(self.encode_text(token))
            else:  # only digits
                for group in group_digits(part, self.config.digit_handling):
                    output_chunks.append(self.encode_digits([group]))
        return output_chunks

    def regex_split(self, text: str) -> list[str]:
        if self.config.regex_pattern is None:
            return [text]
        return re.findall(self.config.regex_pattern, text)

    def split_encoded(self, text_chunks: Sequence[CharEncT]) -> list[Sequence[CharEncT]]:
        return [text_chunks]

    def encode_digits(self, digit_groups: list[str]) -> list[DigitsEnc]:
        return [DigitsEnc([self.token_to_id[dgroup]]) for dgroup in digit_groups]

    def tokens_repr(self, atomic_token_ids: InputTokenSeq) -> str:
        """Representation that is able to handle partial/broken sequences"""
        return self.decode(atomic_token_ids, errors="backslashreplace")

    def pretokens_repr(self, pretokens: PretokenizedT) -> list[str]:
        return [self.decode(token_ids) for token_ids in pretokens]

    # ---- Helper checks for token policies ----
    def try_decode_strict(self, token_seq: InputTokenSeq) -> str | None:
        """Return decoded text if the base token sequence decodes without errors under strict mode, otherwise None."""
        try:
            return self.decode(token_seq, errors="strict")
        except ValueError:
            return None

    def char_boundary_ok(self, token_seq: InputTokenSeq) -> bool:
        """Return True if the sequence respects the char-boundary policy when decoding fails."""
        return [i for i, tid in enumerate(token_seq) if tid in self.is_initial_char_tokens] == [0]

    def violates_inherited_order(self, text: str) -> bool:
        """Return True if text does not start with an Inherited char."""
        return len(text) == 1 or text[0] not in self.inherited_chars

    def token_allowed(self, token_seq: InputTokenSeq) -> bool:
        # Strict decodable sequences are allowed unless they violate inherited-start policy
        if self.config.enforce_char_boundaries:
            text = self.try_decode_strict(token_seq)
            if text is None and not self.char_boundary_ok(token_seq):
                return False
            elif self.config.enforce_inherited and text is not None and self.violates_inherited_order(token_seq):
                return False
        return True

    def bpe_merge_allowed(self, a: InputTokenSeq, b: InputTokenSeq) -> bool:
        return self.token_allowed(list(a) + list(b))


class UTF8Pretokenizer(Pretokenizer, config_type=UTF8PretokenizerConfig):
    def __init__(self, config: UTF8PretokenizerConfig) -> None:
        super().__init__(config)
        self.token_to_bytes = {tid: bytes([b]) for b, tid in self.byte_ids.items()}
        self.token_to_bytes.update(
            {tid: text.encode("utf-8") for tid, text in self.atomic_tokens.items() if text.isdigit()}
        )
        self.is_initial_char_tokens = {
            tid for tid, bs in self.token_to_bytes.items() if bs[0] & 0b11000000 != 0b10000000
        }

    def _build_atomic_tokens(self):
        self.byte_ids = {b: self._register_token(f"<BYTE_{b:02X}>") for b in range(256)}

    def decode(self, atomic_token_ids: InputTokenSeq, errors="replace") -> str:
        byteseq = b"".join(self.token_to_bytes[tid] for tid in atomic_token_ids)
        return byteseq.decode("utf-8", errors=errors)

    def encode_text(self, text: str) -> Sequence[CharEncT]:
        return [UTF8CharEnc(self.byte_ids[c]) for c in text.encode("utf-8")]


class ScriptPretokenizer(Pretokenizer, config_type=ScriptPretokenizerConfig):
    def __init__(self, config: ScriptPretokenizerConfig) -> None:
        if config.script_split and config.regex_pattern is not None:
            raise ValueError("script_split and regex_pattern should probably not be used together")
        super().__init__(config)
        self._script_split: bool = config.script_split
        self.space_group = self.encode_text(" ")

    def _build_atomic_tokens(self):
        self.num_index_tokens = max(len(block.chars) for block in self.config.script_config.blocks)
        self.detokenize_map = {}
        self.char_encoding = {}
        index_tokens = [self._register_token(f"<|SCRIPT_INDEX_{i}|>") for i in range(self.num_index_tokens)]
        self.is_initial_char_tokens = set()
        for block in self.config.script_config.blocks:
            block_token = self._register_token(f"<|BLOCK_{block.script}_{block.category}_{block.sub_block_id}|>")
            self.is_initial_char_tokens.add(block_token)
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
                    decoded += self.atomic_tokens[script_tok]
                elif errors == "replace":
                    decoded += "�"
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok}) is not a valid token pair!")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        return decoded

    def encode_text(self, text: str) -> Sequence[CharEncT]:
        return [self.char_encoding[c] for c in text]

    def split_encoded(self, encoding: Sequence[CharEncT]) -> list[Sequence[CharEncT]]:
        """
        Pretokenize the encoding by grouping adjacent tokens with the same script.
        Special cases:
        - If a script combines with spaces, it will be merged with the space group.
        - Inherited scripts are merged with the previous group, and do not split the group.
        """
        if not self._script_split:
            return [encoding]
        space_group = self.space_group
        script_encoding = [c for c in encoding if isinstance(c, ScriptCharEnc)]
        script_groups: list[list[ScriptCharEnc]] = [
            list(g) for _, g in itertools.groupby(script_encoding, key=lambda x: x.script_id)
        ]

        merged_groups: list[Sequence[CharEncT]] = []
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
