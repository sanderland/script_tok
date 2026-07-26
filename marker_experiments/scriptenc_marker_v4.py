"""Boundary-marker pretokenizer, v4: markers on word spans AND punctuation.

Design
------
  * word spans (LM scripts that use spaces) get <|> on BOTH sides, always --
    that is the point: 'the' looks identical whether or not a space preceded it,
    which kills the ' the'/'the' duplicate vocabulary pair.
  * punctuation gets <|> ONLY on a side that actually had a single space, which
    was then elided. So none in 'a=b', one in 'a, b', two in 'a = b'.

The asymmetry is what keeps decoding unambiguous. If punctuation were wrapped
unconditionally, 'a,b' would give <|>a<|> <|>,<|> <|>b<|> -- touching markers at
both junctions, indistinguishable from 'a , b'. Marking punctuation only on a
space side preserves the invariant:

    two <|> touching  <=>  exactly one elided space
    a lone <|>        <=>  pure structural boundary, contributes no character

Word-adjacency exception
------------------------
Two *different* word scripts directly adjacent with no space (Greek letters used
as identifiers: 'upperDelta', 'spi', 'Deltax') would both emit unconditional
markers, they would touch, and decode would insert a phantom space. This is not
rare once code is in the mix -- it hit 5/500 held-out code documents. A word
therefore drops its OPENING marker when the preceding unit is also a word, which
(since _build_units absorbs same-script and inherited runs) can only mean a
script change with no space. The preceding word keeps its closing marker, which
is then a lone marker and decodes to nothing.

Chunking
--------
Units are NOT fused across an elided space. Since bpe_merge_allowed already
forbids merging across two touching markers, fusing yields exactly the same set
of legal merges while inflating the corpus ~5x in unique chunks and BPE training
~10x in wall clock. Decode reads the flat atomic stream, so chunk boundaries do
not affect reconstruction -- markers still touch across a boundary.
"""

import itertools
from typing import Sequence

from script_bpe.pretokenize.pretokenizer import ScriptPretokenizer, ScriptPretokenizerConfig, CharEncT


class MarkerCharEnc:
    __slots__ = ("script_id", "combines_with_spaces", "atomic_token_ids", "inherited")

    def __init__(self, token_id: int):
        self.script_id = -2  # sentinel, never scanned via groupby -- inserted directly
        self.combines_with_spaces = False
        self.inherited = False
        self.atomic_token_ids = [token_id]

    def __repr__(self):
        return f"MarkerCharEnc(atomic_token_ids={self.atomic_token_ids})"


class MarkerV4PretokenizerConfig(ScriptPretokenizerConfig):
    cls: str = "MarkerV4Pretokenizer"


class MarkerV4Pretokenizer(ScriptPretokenizer, config_type=MarkerV4PretokenizerConfig):
    MARKER_TEXT = "<|>"

    def _build_atomic_tokens(self):
        super()._build_atomic_tokens()
        self.marker_token_id = self._register_token(self.MARKER_TEXT)
        self.is_initial_char_tokens.add(self.marker_token_id)
        # word scripts: category LM (letters) AND combines_with_spaces. Excludes
        # Han/Hiragana/Katakana/Thai (LM but spaceless) and punctuation (which is
        # combines_with_spaces under V3 via the (ALL, "PSF") entry, but not LM).
        self.lm_wrap_script_ids = {
            block.script_id
            for block in self.config.script_config.blocks
            if block.category == "LM" and block.combines_with_spaces
        }
        # Subclass hook (see v5): extra script ids to treat like punctuation -- markable
        # on a side whose single space was elided, never marked unconditionally.
        self.extra_markable_script_ids = self._extra_markable_script_ids()

    def _extra_markable_script_ids(self) -> set:
        return set()

    def bpe_merge_allowed(self, a, b) -> bool:
        # No learned token may span an elided-space point. With per-unit chunks this
        # junction never falls inside a chunk, so it is belt-and-braces -- but it also
        # guarantees no token can contain '<|><|>', which would reintroduce per-word
        # duplication keyed on what follows instead of what precedes.
        if a[-1] == self.marker_token_id and b[0] == self.marker_token_id:
            return False
        return super().bpe_merge_allowed(a, b)

    def decode(self, tokenization, errors="replace") -> str:
        decoded = ""
        i = 0
        n = len(tokenization)
        while i < n:
            if tokenization[i] == self.marker_token_id:
                if i + 1 < n and tokenization[i + 1] == self.marker_token_id:
                    decoded += " "  # two markers touching = one elided space
                    i += 2
                else:
                    i += 1  # lone marker: structural boundary only
                continue
            script_tok = tokenization[i]
            ix_tok = tokenization[i + 1] if i + 1 < n else None
            if (script_tok, ix_tok) in self.detokenize_map:
                decoded += self.detokenize_map[(script_tok, ix_tok)]
                i += 2
            else:
                if errors == "backslashreplace":
                    decoded += self.atomic_tokens[script_tok]
                elif errors == "replace":
                    decoded += "�"
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok}) is not a valid token pair!")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        return decoded

    # unit kinds
    WORD, PUNCT, SPACE, OTHER = "word", "punct", "space", "other"

    def _build_units(self, script_groups) -> list[tuple[str, list]]:
        """Collapse script groups into units, absorbing inherited/same-script continuations."""
        lm_ids = self.lm_wrap_script_ids
        space_group = self.space_group
        units: list[tuple[str, list]] = []
        i = 0
        while i < len(script_groups):
            group = script_groups[i]
            if group == space_group:  # exactly one space character
                units.append((self.SPACE, list(group)))
                i += 1
                continue
            script_id = group[0].script_id
            if script_id in lm_ids:
                kind = self.WORD
            elif group[0].combines_with_spaces or script_id in self.extra_markable_script_ids:
                kind = self.PUNCT
            else:
                kind = self.OTHER
            content = list(group)
            i += 1
            while i < len(script_groups) and (
                script_groups[i][0].inherited or script_groups[i][0].script_id == script_id
            ):
                content += script_groups[i]
                i += 1
            units.append((kind, content))
        return units

    def split_encoded(self, encoding: Sequence[CharEncT]) -> list[Sequence[CharEncT]]:
        if not self._script_split:
            return [encoding]
        script_encoding = [c for c in encoding if hasattr(c, "script_id")]
        if len(script_encoding) != len(encoding):
            raise ValueError(f"Unexpected encoding: {encoding}")
        script_groups = [list(g) for _, g in itertools.groupby(script_encoding, key=lambda x: x.script_id)]
        units = self._build_units(script_groups)
        marker = MarkerCharEnc(self.marker_token_id)

        # A single space is elided when both neighbours are markable (word or punct); the
        # facing sides then carry a marker, so the two markers touch in the atomic stream.
        markable = (self.WORD, self.PUNCT)
        elided = [False] * len(units)
        for i, (kind, _) in enumerate(units):
            if kind != self.SPACE:
                continue
            if 0 < i < len(units) - 1 and units[i - 1][0] in markable and units[i + 1][0] in markable:
                elided[i] = True

        chunks: list[Sequence[CharEncT]] = []
        for i, (kind, content) in enumerate(units):
            if kind == self.SPACE:
                if not elided[i]:
                    chunks.append(content)  # left untouched, exactly as the baseline encodes it
                continue
            if kind == self.OTHER:
                chunks.append(content)
                continue
            if kind == self.WORD:
                right = True
                left = not (i > 0 and units[i - 1][0] == self.WORD)  # see word-adjacency note
            else:  # PUNCT: only on a side whose space was actually elided
                left = i > 0 and elided[i - 1]
                right = i + 1 < len(units) and elided[i + 1]
            chunks.append(([marker] if left else []) + content + ([marker] if right else []))
        return chunks
