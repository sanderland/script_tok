"""Boundary-marker pretokenization for SCRIPT encoding.

A single atomic token `<|>` delimits *spans*. The single space between two
adjacent delimited spans is elided at encode time and reconstructed at decode
time from the resulting pair of touching markers:

    two <|> touching  ->  exactly one space
    a lone <|>        ->  nothing (structural boundary only)

This removes the with/without-space duplication that the leading-space
convention creates (`' the'` and `'the'` as separate vocabulary entries).

Span identification
-------------------
A **word span** is a maximal run of characters from *any* space-using script
(`DEFAULT_SCRIPTS_LM_WITH_SPACES`, category LM), merged across script changes.
So `latin` immediately followed by Cyrillic `кириллица` is ONE span, delimited
only at its outer edges:

    <|>latin  кириллица<|>          (no marker between them)

Merging across scripts is what makes the scheme well defined. If each script run
were delimited separately, two unconditional markers would meet at the script
change, they would be indistinguishable from an elided space, and decode would
fabricate one. Merging first means two word spans can never be adjacent, so the
touching-marker signal is unambiguous with no special case.

Inside a span the baseline's script-based chunk split is preserved (the marker
rides on the first and last chunk), so no BPE merge ever spans a script change
that the baseline would not also allow.

Boundary targets
----------------
`boundary_targets` selects which unit kinds are delimited:

    ("word",)                     - word spans only
    ("word", "punct")             - and punctuation
    ("word", "punct", "digit")    - and digits

Word spans are delimited on BOTH sides unconditionally, which is the point: a
word has one canonical form regardless of what precedes it. Punctuation and
digits are delimited only on a side whose adjacent single space was elided.
That asymmetry is required: marking punctuation unconditionally would make `a,b`
encode as `<|>a<|> <|>,<|> <|>b<|>`, with touching markers at both junctions and
therefore indistinguishable from `a , b`. Marking only on a space side keeps the
invariant, at the cost of up to four variants per mark (`,` `,<|>` `<|>,`
`<|>,<|>`) -- affordable because punctuation and digits are closed sets, unlike
words.

Digits are a separate target because `script_category_v3` folds L/M into LM,
Z/Cc into ZC, So into So and P/S/Cf into PSF, but leaves category N alone, so
digits are neither letters nor `combines_with_spaces`.
"""

import itertools
from typing import Literal, Sequence

from pydantic import ConfigDict

from script_bpe.pretokenize.pretokenizer import (
    CharEncT,
    ScriptPretokenizer,
    ScriptPretokenizerConfig,
    group_digits,
)

BoundaryTarget = Literal["word", "punct", "digit"]


class MarkerCharEnc:
    """Stand-in char encoding for the marker token, inserted directly rather than scanned."""

    __slots__ = ("script_id", "combines_with_spaces", "atomic_token_ids", "inherited")

    def __init__(self, token_id: int):
        self.script_id = -2  # sentinel: never produced by groupby over real text
        self.combines_with_spaces = False
        self.inherited = False
        self.atomic_token_ids = [token_id]

    def __repr__(self):
        return f"MarkerCharEnc(atomic_token_ids={self.atomic_token_ids})"


class BoundaryScriptPretokenizerConfig(ScriptPretokenizerConfig):
    cls: str = "BoundaryScriptPretokenizer"
    boundary_targets: tuple[BoundaryTarget, ...] = ("word", "punct", "digit")
    model_config = ConfigDict(extra="forbid")


class BoundaryScriptPretokenizer(ScriptPretokenizer, config_type=BoundaryScriptPretokenizerConfig):
    MARKER_TEXT = "<|>"

    # unit kinds
    WORD, PUNCT, DIGIT, SPACE, OTHER = "word", "punct", "digit", "space", "other"

    def _build_atomic_tokens(self):
        super()._build_atomic_tokens()
        self.marker_token_id = self._register_token(self.MARKER_TEXT)
        self.is_initial_char_tokens.add(self.marker_token_id)
        blocks = self.config.script_config.blocks
        # the ~20 space-using writing systems, as letters
        self.word_script_ids = {b.script_id for b in blocks if b.category == "LM" and b.combines_with_spaces}
        self.digit_script_ids = {b.script_id for b in blocks if b.category == "N"}
        targets = set(self.config.boundary_targets)
        unknown = targets - {"word", "punct", "digit"}
        if unknown:
            raise ValueError(f"Unknown boundary_targets: {sorted(unknown)}")
        # kinds that carry a boundary and may therefore participate in space elision
        self.marked_kinds = frozenset(
            k for k, name in ((self.WORD, "word"), (self.PUNCT, "punct"), (self.DIGIT, "digit"))
            if name in targets
        )

    def __init__(self, config: BoundaryScriptPretokenizerConfig) -> None:
        super().__init__(config)
        # Digit-group tokens are registered by the base _build_digit_tokens AFTER
        # _build_atomic_tokens runs, and ScriptPretokenizer.decode has no path for them
        # (digit_handling was only ever exercised with UTF8Pretokenizer). Collect their
        # ids here so decode can emit them directly; they are the only atomic tokens
        # whose text is all digits.
        self.digit_token_ids = {tid for tid, txt in self.atomic_tokens.items() if txt.isdigit()}

    def bpe_merge_allowed(self, a, b) -> bool:
        # No learned token may span an elided-space point. Without this, BPE learns tokens
        # like '<|>the<|><|>' that swallow the dangling half of the next span's opening
        # marker, reintroducing per-word duplication keyed on what follows.
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
                    decoded += " "  # two markers touching == one elided space
                    i += 2
                else:
                    i += 1  # lone marker: structural boundary, no character
                continue
            if tokenization[i] in self.digit_token_ids:
                decoded += self.atomic_tokens[tokenization[i]]  # digit group, single token
                i += 1
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

    def _kind(self, group) -> str:
        script_id = group[0][1].script_id
        if script_id in self.word_script_ids:
            return self.WORD
        if script_id in self.digit_script_ids:
            return self.DIGIT
        if group[0][1].combines_with_spaces:
            return self.PUNCT
        return self.OTHER

    def _build_units(self, script_groups) -> list[tuple[str, list[list]]]:
        """Group script runs into units. A unit holds a LIST of script runs, so a word span
        that crosses scripts keeps its internal split while being one delimited span."""
        units: list[tuple[str, list[list]]] = []
        i = 0
        n = len(script_groups)
        while i < n:
            group = script_groups[i]
            if [e for _, e in group] == self.space_group:  # exactly one space character
                units.append((self.SPACE, [list(group)]))
                i += 1
                continue
            kind = self._kind(group)
            runs = [list(group)]
            i += 1
            if kind == self.WORD:
                # merge across ANY space-using-script change, plus inherited marks
                while i < n and (
                    script_groups[i][0][1].inherited or script_groups[i][0][1].script_id in self.word_script_ids
                ):
                    if (
                        script_groups[i][0][1].inherited
                        or script_groups[i][0][1].script_id == runs[-1][0][1].script_id
                    ):
                        runs[-1] = runs[-1] + list(script_groups[i])
                    else:
                        runs.append(list(script_groups[i]))
                    i += 1
            else:
                script_id = group[0][1].script_id
                while i < n and (
                    script_groups[i][0][1].inherited or script_groups[i][0][1].script_id == script_id
                ):
                    runs[-1] = runs[-1] + list(script_groups[i])
                    i += 1
            units.append((kind, runs))
        return units

    def split_unencoded_and_encode(self, text: str) -> list[Sequence[CharEncT]]:
        """Encode and chunk in one pass, keeping source characters alongside encodings.

        The base implementation splits digit runs into their own chunks *before*
        split_encoded runs. That is fatal here: a digit unit and its neighbouring word
        would land in different chunks, so the shared single space between them could
        never be seen as elidable, and `digit` as a boundary target would silently do
        nothing. The unit analysis therefore has to happen over the whole text first,
        with digit grouping applied inside a digit unit afterwards.
        """
        if self.config.regex_pattern is not None:
            raise NotImplementedError("BoundaryScriptPretokenizer does not support regex_pattern")
        enc = self.encode_text(text)  # 1:1 with characters
        pairs = list(zip(text, enc))
        groups = [list(g) for _, g in itertools.groupby(pairs, key=lambda p: p[1].script_id)]
        units = self._build_units(groups)
        marker = MarkerCharEnc(self.marker_token_id)
        marked = self.marked_kinds

        # A single space is elided when BOTH neighbours carry a boundary; their facing
        # sides then hold markers, which land adjacent in the atomic stream.
        elided = [False] * len(units)
        for i, (kind, _) in enumerate(units):
            if kind != self.SPACE:
                continue
            if 0 < i < len(units) - 1 and units[i - 1][0] in marked and units[i + 1][0] in marked:
                elided[i] = True

        chunks: list[Sequence[CharEncT]] = []
        for i, (kind, runs) in enumerate(units):
            if kind == self.SPACE:
                if not elided[i]:
                    chunks.append([e for _, e in runs[0]])  # exactly as the baseline emits it
                continue
            digits = "".join(c for run in runs for c, _ in run) if kind == self.DIGIT else ""
            # group_digits/encode_digits only have tokens for ASCII 0-9 -- the base pipeline
            # splits on re.split("([0-9]+)"). Category N is far broader (Nd for every script,
            # plus Nl/No: '½', '⅓', '٣', 'Ⅻ'), so grouping those raises KeyError. They stay on
            # the ordinary script path; they are still delimited, but their marked forms are
            # not bounded by grouping.
            if kind == self.DIGIT and self.config.digit_handling is not None and digits.isascii() and digits.isdigit():
                # Split the digit run into groups so marked forms stay bounded: with a
                # whole run as one unit, every distinct number acquires up to four marked
                # variants, which measured 1,093 wasted vocabulary slots (3.17%) for
                # English at 32,768. Splitting means only the run's first and last GROUP
                # can carry a marker -- 10 digits under SPLIT, 1110 under RTL3.
                digits = "".join(c for run in runs for c, _ in run)
                out = [self.encode_digits([g]) for g in group_digits(digits, self.config.digit_handling)]
            else:
                out = [[e for _, e in run] for run in runs]
            if kind not in marked:
                chunks.extend(out)
                continue
            if kind == self.WORD:
                left = right = True  # unconditional: one canonical form per span
            else:
                left = i > 0 and elided[i - 1]
                right = i + 1 < len(units) and elided[i + 1]
            # marker rides the first/last run, preserving any internal split
            if left:
                out[0] = [marker] + list(out[0])
            if right:
                out[-1] = list(out[-1]) + [marker]
            chunks.extend(out)
        return chunks

    def split_encoded(self, encoding: Sequence[CharEncT]) -> list[Sequence[CharEncT]]:
        # All chunking already happened in split_unencoded_and_encode.
        return [encoding]


# Named variants used by the experiments. All are ScriptEncodingV3 with
# enforce_char_boundaries=True; they differ only in which units get a boundary.
BOUNDARY_VARIANTS = {
    "bnd_w": ("word",),
    "bnd_wp": ("word", "punct"),
    "bnd_wpd": ("word", "punct", "digit"),
}


def get_boundary_pretokenizer(name: str) -> BoundaryScriptPretokenizer:
    from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3

    return BoundaryScriptPretokenizer(
        BoundaryScriptPretokenizerConfig(
            script_config=ScriptEncodingV3, boundary_targets=BOUNDARY_VARIANTS[name]
        )
    )
