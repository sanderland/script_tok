import functools
import hashlib
import itertools
from typing import Any

import regex as re

from script_bpe.utils import InputTokenSeq, PretokenizedT, token_array

from ..pretokenize.base import BasePretokenizer

TokenPairT = tuple[int, int]
ScriptEncT = tuple[int, TokenPairT]
ScriptEncNoneT = tuple[int | None, TokenPairT]

class CharEnc:
    __slots__ = ('script_id', 'combines_with_spaces', 'token_pair', 'inherited')
    def __init__(self, block: dict, token_pair: TokenPairT):
        self.script_id = block['script_id']
        self.combines_with_spaces = block['combines_with_spaces']
        self.token_pair = token_pair
        self.inherited = block['script'] == "Inherited"

    def __repr__(self):
        return f"CharEnc(script_id={self.script_id}, token_pair={self.token_pair})"

class ScriptEncodingPretokenizer(BasePretokenizer):
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the ScriptEncodingPretokenizer with a configuration.
        :param config: A dictionary containing 'starting_token_id' and 'normalization'.
        """
        super().__init__(config)

        self.enforce_char_boundaries: bool = config["enforce_char_boundaries"]
        self.blocks = config["blocks"]
        self.num_index_tokens: int = config["num_index_tokens"]
        assert len(self.blocks) == config["num_blocks"], "Number of blocks does not match the configuration"
        self.build_tokenization_maps()
        self.space_group = self.script_encode(" ")

    def hash(self) -> str:
        hash = hashlib.sha1()
        hash.update(super().hash().encode("utf-8"))
        hash.update(str(self.blocks).encode("utf-8"))
        return "SE-" + hash.hexdigest()[:8]

    # loading
    def build_tokenization_maps(self):
        self.char_encoding = {}
        self.base_tokens.update(
            {i + self.starting_token_id: f"<|SCRIPT_INDEX_{i}|>" for i in range(self.num_index_tokens)}
        )
        self.detokenize_map = {}
        for block_id, block in enumerate(self.blocks):
            block_token_id = self.starting_token_id + self.num_index_tokens + block_id
            self.base_tokens[block_token_id] = f"<|BLOCK_{block["script"]}_{block["category"]}_{block["sub_block_id"]}|>"
            for ix, c in enumerate(block["chars"]):
                token_pair = (block_token_id, self.starting_token_id + ix)
                self.char_encoding[c] = CharEnc(block, token_pair)
                self.detokenize_map[token_pair] = c

    # pretokenization
    def script_encode(self, text: str) -> list[CharEnc]:
        """Encode the input text"""
        return [self.char_encoding[c] for c in text]

    def chunk_char_encoding(self, encoding: list[CharEnc]) -> list[list[tuple[int, TokenPairT]]]:
        """
        Pretokenize the encoding by grouping adjacent tokens with the same script.
        Special cases:
        - If a script combines with spaces, it will be merged with the space group.
        - Inherited scripts are merged with the previous group, and do not split the group.
        """
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
            while i < len(script_groups) and (script_groups[i][0].inherited or script_groups[i][0].script_id == current_script):
                merged_group += script_groups[i]
                i += 1
            merged_groups.append(merged_group)
        if i < len(script_groups):
            merged_groups.append(script_groups[i])  # add last group
        return merged_groups

    def _encode_and_chunk(self, text: str) -> PretokenizedT:
        encoded_and_grouped = self.chunk_char_encoding(self.script_encode(text))
        # Concatenate token pairs
        return [token_array([t for ce in ces for t in ce.token_pair]) for ces in encoded_and_grouped]

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

    def is_index_token(self, i: int) -> bool:
        return self.starting_token_id <= i < self.num_index_tokens + self.starting_token_id

    def bpe_merge_allowed(self, token_seq1: InputTokenSeq, token_seq2: InputTokenSeq) -> bool:
        if not self.enforce_char_boundaries:
            return True
        if len(token_seq1) >= 2 and len(token_seq2) >= 2:
            return True  # both are full sequences
        if len(token_seq1) >= 2 or len(token_seq2) >= 2:
            return False  # one is full, the other is partial
        return not self.is_index_token(token_seq1[0])

    def token_allowed(self, token_seq: InputTokenSeq) -> bool:
        if not self.enforce_char_boundaries:
            return True
        if len(token_seq) % 2 != 0:
            return len(token_seq) == 1 
        return not self.is_index_token(token_seq[0]) # index token only as singleton

    # utility
    @functools.cache
    def token_script_name(self, block_token_id: int) -> str:  # maps token to block name without sub-block
        return re.sub(r"<\|BLOCK_(.*)_\d+\|>", r"\1", self.base_tokens[block_token_id])

    def sequence_script_name(self, seq: list[int]) -> str | None:
        si = 0
        if self.is_index_token(seq[si]) and len(seq) > 1:
            si += 1
        if tuple(seq[si : si + 2]) == self.space_group[0][1] and len(seq) > si + 2:
            si += 2
        if self.is_index_token(seq[si]):
            return None
        return self.token_script_name(seq[si])


class ScriptEncodingPretokenizerRegexSplitting(ScriptEncodingPretokenizer):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.regex = re.compile(config["regex"], re.DOTALL)  # No default, must be provided in config

    def hash(self) -> str:
        hash = hashlib.sha1()
        hash.update(super().hash().encode("utf-8"))
        hash.update(str(self.config["regex"]).encode("utf-8"))
        return "SE+R-" + hash.hexdigest()[:8]

    def _encode_and_chunk(self, text: str) -> PretokenizedT:
        """split with regex, and then encode with script encoding"""
        chunks = re.findall(self.regex, text)
        encoded = [self.script_encode(t) for t in chunks]
        # concat token pairs
        return [token_array([t for ce in group for t in ce.token_pair]) for group in encoded if group]
