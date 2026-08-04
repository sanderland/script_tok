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


class CodeCharEnc(MarkerCharEnc):
    """Stand-in char encoding for a caps code. Distinct sentinel so it never groups with
    real text or with the boundary marker."""

    def __init__(self, token_id: int):
        super().__init__(token_id)
        self.script_id = -3


class BoundaryScriptPretokenizerConfig(ScriptPretokenizerConfig):
    cls: str = "BoundaryScriptPretokenizer"
    boundary_targets: tuple[BoundaryTarget, ...] = ("word", "punct", "digit")
    # Caps codes, in the style of the older Claude tokenizer: a title-case word is emitted
    # as a shift code plus its lowercased form, an all-caps word as a caps-lock code plus
    # its lowercased form, so 'The'/'the' and 'NASA'/'nasa' share vocabulary entries. Whole
    # spans only; mixed case ('GaN', 'WiFi') is left literal.
    caps_codes: bool = False
    model_config = ConfigDict(extra="forbid")


class BoundaryScriptPretokenizer(ScriptPretokenizer, config_type=BoundaryScriptPretokenizerConfig):
    MARKER_TEXT = "<|>"
    SHIFT_TEXT = "<^>"   # title case: next span is Xxxx
    CAPS_TEXT = "<^^>"   # caps lock: next span is XXXX

    # unit kinds
    WORD, PUNCT, DIGIT, SPACE, OTHER = "word", "punct", "digit", "space", "other"

    def _build_atomic_tokens(self):
        super()._build_atomic_tokens()
        self.marker_token_id = self._register_token(self.MARKER_TEXT)
        self.is_initial_char_tokens.add(self.marker_token_id)
        if self.config.caps_codes:
            self.shift_token_id = self._register_token(self.SHIFT_TEXT)
            self.caps_token_id = self._register_token(self.CAPS_TEXT)
            self.is_initial_char_tokens.add(self.shift_token_id)
            self.is_initial_char_tokens.add(self.caps_token_id)
        else:
            self.shift_token_id = self.caps_token_id = None
        self.caps_code_ids = {self.shift_token_id, self.caps_token_id} - {None}
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

    def _place_caps_code(self, out, code_id):
        """Put the code inside the span, before the markers are applied: <|><^>the<|>.

        This layout keeps the code between the opening marker and the word, so the atomic
        sequence of `<|>the<|>` does not occur anywhere inside it and no piece can ever
        cover the span alone. The word entry therefore cannot be shared with the
        lowercase form -- see ExtCapsBoundaryScriptPretokenizer.
        """
        out[0] = [CodeCharEnc(code_id)] + out[0]
        return out

    def _place_caps_code_outside(self, out, code_id):
        """Applied after the markers. A no-op unless the code belongs outside them."""
        return out

    def bpe_merge_allowed(self, a, b) -> bool:
        # No learned token may span an elided-space point. Without this, BPE learns tokens
        # like '<|>the<|><|>' that swallow the dangling half of the next span's opening
        # marker, reintroducing per-word duplication keyed on what follows.
        if a[-1] == self.marker_token_id and b[0] == self.marker_token_id:
            return False
        return super().bpe_merge_allowed(a, b)

    def decode(self, tokenization, errors="replace") -> str:
        decoded = ""
        pending = None   # caps code awaiting its span
        buf = ""
        i = 0
        n = len(tokenization)

        def emit(text):
            nonlocal decoded, pending, buf
            if pending is None:
                decoded += text
            else:
                buf += text

        def flush():
            """Close a caps-coded span at its terminating marker."""
            nonlocal decoded, pending, buf
            if pending is None:
                return
            decoded += (buf[:1].upper() + buf[1:]) if pending == "shift" else buf.upper()
            pending, buf = None, ""

        while i < n:
            if self.config.caps_codes and tokenization[i] in (self.shift_token_id, self.caps_token_id):
                flush()  # a code immediately after another closes the previous span
                pending = "shift" if tokenization[i] == self.shift_token_id else "caps"
                buf = ""
                i += 1
                continue
            if tokenization[i] == self.marker_token_id:
                flush()
                if i + 1 < n and tokenization[i + 1] == self.marker_token_id:
                    decoded += " "  # two markers touching == one elided space
                    i += 2
                else:
                    i += 1  # lone marker: structural boundary, no character
                continue
            if tokenization[i] in self.digit_token_ids:
                emit(self.atomic_tokens[tokenization[i]])  # digit group, single token
                i += 1
                continue
            script_tok = tokenization[i]
            ix_tok = tokenization[i + 1] if i + 1 < n else None
            if (script_tok, ix_tok) in self.detokenize_map:
                emit(self.detokenize_map[(script_tok, ix_tok)])
                i += 2
            else:
                if errors == "backslashreplace":
                    emit(self.atomic_tokens[script_tok])
                elif errors == "replace":
                    decoded += "�"
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok}) is not a valid token pair!")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        flush()  # span running to end of stream
        return decoded

    @staticmethod
    def _caps_form(text: str):
        """Return (code_kind, lowercased) if text is title or all caps AND the transform is
        exactly invertible, else None.

        Invertibility cannot be assumed. Unicode case mapping is not a bijection: 'I'.lower()
        is 'i' but Turkish dotless/dotted i break the pair, '\u0130'.lower() is two
        characters, and '\u1e9e'.lower() is '\u00df' whose upper is 'SS'. Every candidate is
        therefore verified by re-applying the transform and comparing, and anything that does
        not reproduce the source exactly is left literal.
        """
        if not text or text.islower():
            return None
        low = text.lower()
        if len(low) != len(text):
            return None
        if low[0].upper() + low[1:] == text:
            return "shift", low
        if len(text) > 1 and low.upper() == text:
            return "caps", low
        return None

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
                code_id = None
            else:
                out = [[e for _, e in run] for run in runs]
                code_id = None
                if kind == self.WORD and self.config.caps_codes:
                    text = "".join(c for run in runs for c, _ in run)
                    form = self._caps_form(text)
                    if form is not None:
                        code, low = form
                        # Re-encode the lowercased span and regroup, so a span that crosses
                        # scripts keeps the same internal split it would have had untouched.
                        low_pairs = list(zip(low, self.encode_text(low)))
                        low_runs = [list(g) for _, g in itertools.groupby(low_pairs, key=lambda x: x[1].script_id)]
                        out = [[e for _, e in r] for r in low_runs]
                        code_id = self.shift_token_id if code == "shift" else self.caps_token_id
                        out = self._place_caps_code(out, code_id)
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
            if code_id is not None:
                out = self._place_caps_code_outside(out, code_id)
            chunks.extend(out)
        return chunks

    def split_encoded(self, encoding: Sequence[CharEncT]) -> list[Sequence[CharEncT]]:
        # All chunking already happened in split_unencoded_and_encode.
        return [encoding]


class ExtCapsBoundaryScriptPretokenizerConfig(BoundaryScriptPretokenizerConfig):
    cls: str = "ExtCapsBoundaryScriptPretokenizer"


class ExtCapsBoundaryScriptPretokenizer(
    BoundaryScriptPretokenizer, config_type=ExtCapsBoundaryScriptPretokenizerConfig
):
    """Caps codes outside the span's markers, in the same pre-token: `<^><|>the<|>`.

    The default layout puts the code inside the delimited span, `<|><^>the<|>`, which is
    a different chunk from `<|>the<|>`. The trainer therefore learns a separate token for
    the title-case form and the case duplication survives -- measured on the 250M English
    cell, 4,837 of 21,819 alphabetic entries were the same word in two or three cased
    forms. The scheme cost nothing and bought nothing because it did nothing.

    Here the code sits outside the markers, so the atomic sequence of `<|>the<|>` occurs
    inside `<^><|>the<|>` as a suffix and the trainer CAN cover the span with the same
    piece the lowercase form uses:

        the   ->  [<|>the<|>]
        The   ->  [<^>] [<|>the<|>]    when the merged form is not worth a slot

    It is not forced to. For a word whose title-case form is frequent enough, BPE will
    still merge the code in and spend an entry on `<^><|>the<|>`, which is the right call
    when the frequency justifies it. What the layout buys is the option, over the band of
    words common enough to hold an entry but not common enough in title case to earn a
    second one. The inside layout does not offer that option at any frequency.

    Space elision still applies across the code. Two delimited spans separated only by
    caps codes had a space between them, so decode reads `<|> <^> <|>` as one elided
    space with the code applying to the span that follows.
    """

    def _place_caps_code(self, out, code_id):
        return out  # not inside; see _place_caps_code_outside

    def _place_caps_code_outside(self, out, code_id):
        out[0] = [CodeCharEnc(code_id)] + out[0]
        return out

    def decode(self, tokenization, errors="replace") -> str:
        decoded = ""
        pending = None
        buf = ""
        i = 0
        n = len(tokenization)
        codes = {self.shift_token_id, self.caps_token_id} - {None}

        def emit(text):
            nonlocal decoded, buf
            if pending is None:
                decoded += text
            else:
                buf += text

        def flush():
            nonlocal decoded, pending, buf
            if pending is None:
                return
            decoded += (buf[:1].upper() + buf[1:]) if pending == "shift" else buf.upper()
            pending, buf = None, ""

        def kind(tid):
            return "shift" if tid == self.shift_token_id else "caps"

        while i < n:
            tok = tokenization[i]
            if tok in codes:
                flush()
                pending, buf = kind(tok), ""
                i += 1
                continue
            if tok == self.marker_token_id:
                # A non-empty buffer means this marker closes a coded span. An empty one
                # means it opens the span the preceding code applies to, so it must not
                # flush -- that is the whole difference from the inside-the-span layout.
                if buf:
                    flush()
                j = i + 1
                between = []
                while j < n and tokenization[j] in codes:
                    between.append(tokenization[j])
                    j += 1
                if j < n and tokenization[j] == self.marker_token_id:
                    decoded += " "  # two markers, codes between them or not, == one space
                    i = j + 1
                    if between:
                        pending, buf = kind(between[-1]), ""
                    continue
                i += 1
                continue
            if tok in self.digit_token_ids:
                emit(self.atomic_tokens[tok])
                i += 1
                continue
            ix_tok = tokenization[i + 1] if i + 1 < n else None
            if (tok, ix_tok) in self.detokenize_map:
                emit(self.detokenize_map[(tok, ix_tok)])
                i += 2
            else:
                if errors == "backslashreplace":
                    emit(self.atomic_tokens[tok])
                elif errors == "replace":
                    emit("\ufffd")
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({tok}, {ix_tok})")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        flush()
        return decoded



class ExtCapsFixBoundaryScriptPretokenizerConfig(ExtCapsBoundaryScriptPretokenizerConfig):
    cls: str = "ExtCapsFixBoundaryScriptPretokenizer"


class ExtCapsFixBoundaryScriptPretokenizer(
    ExtCapsBoundaryScriptPretokenizer, config_type=ExtCapsFixBoundaryScriptPretokenizerConfig
):
    """`_extcaps` with the case test the scheme is supposed to apply: title or upper case.

    `str.islower()` needs a cased character to be True, so the inherited test lets uncased
    scripts straight through: lower-casing Arabic or Hangul is a no-op, the round trip
    therefore succeeds, and a `<^>` lands in front of every word span in those languages.
    On the Goldfish slices that is 33,183 of 33,579 Arabic word spans and 46,137 of 46,657
    Korean ones, against 6,072 of 34,707 for English, which is real title case. Decoding
    applies `capitalize()`, another no-op on uncased text, so every round trip still
    passes and nothing about it fails loudly. What it costs is tokens: Arabic spends 2.3%
    more of them under `bnd_wpd_extcaps` than under `bnd_wpd`, and that is the whole of
    the Arabic case-code regression in the compression tables.

    Testing `istitle() or isupper()` first leaves cased scripts untouched -- the two tests
    agree on every word span of the English, German, Finnish and Russian slices -- and
    drops the uncased ones to the Latin words those documents happen to contain, 260 for
    Arabic and 855 for Korean.
    """

    @staticmethod
    def _caps_form(text: str):
        if not text or not (text.istitle() or text.isupper()):
            return None
        return BoundaryScriptPretokenizer._caps_form(text)


# Named variants used by the experiments. All are ScriptEncodingV3 with
# enforce_char_boundaries=True; they differ only in which units get a boundary.
BOUNDARY_VARIANTS = {
    "bnd_w": ("word",),
    "bnd_wp": ("word", "punct"),
    "bnd_wpd": ("word", "punct", "digit"),
}


# Caps-code variants: same boundary targets, plus <^>/<^^> case codes on word spans.
# Kept in a separate table because the grids iterate BOUNDARY_VARIANTS to enumerate cells.
CAPS_VARIANTS = {f"{name}_caps": targets for name, targets in BOUNDARY_VARIANTS.items()}

# Same boundary targets, caps codes placed outside the span so the word token is shared.
EXTCAPS_VARIANTS = {f"{name}_extcaps": targets for name, targets in BOUNDARY_VARIANTS.items()}

# Same placement, but the case test rejects uncased scripts instead of coding every span.
EXTCAPSFIX_VARIANTS = {f"{name}_extcapsfix": targets for name, targets in BOUNDARY_VARIANTS.items()}

ALL_VARIANTS = {**BOUNDARY_VARIANTS, **CAPS_VARIANTS, **EXTCAPS_VARIANTS, **EXTCAPSFIX_VARIANTS}


def get_boundary_pretokenizer(name: str, **overrides) -> BoundaryScriptPretokenizer:
    """Build a named variant. `overrides` are passed to the config (e.g. digit_handling)."""
    from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3

    if name not in ALL_VARIANTS:
        raise ValueError(f"unknown boundary variant {name!r}; have {sorted(ALL_VARIANTS)}")
    fixed = name in EXTCAPSFIX_VARIANTS
    external = fixed or name in EXTCAPS_VARIANTS
    if fixed:
        cls, cfg = ExtCapsFixBoundaryScriptPretokenizer, ExtCapsFixBoundaryScriptPretokenizerConfig
    elif external:
        cls, cfg = ExtCapsBoundaryScriptPretokenizer, ExtCapsBoundaryScriptPretokenizerConfig
    else:
        cls, cfg = BoundaryScriptPretokenizer, BoundaryScriptPretokenizerConfig
    return cls(
        cfg(
            script_config=ScriptEncodingV3,
            boundary_targets=ALL_VARIANTS[name],
            caps_codes=external or name in CAPS_VARIANTS,
            **overrides,
        )
    )
