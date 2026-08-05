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
Word spans are delimited on BOTH sides unconditionally, which is the point: a
word has one canonical form regardless of what precedes it. That is the scheme
rather than an option, so `"word"` is not a member of `boundary_targets`; the
tuple selects only what is delimited *besides* words:

    ()                    - word spans only            (bnd_w)
    ("punct",)            - and punctuation            (bnd_wp)
    ("punct", "digit")    - and digits                 (bnd_wpd)

Punctuation and
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

Case codes
----------
`shift_code` and `caps_code` enable the two codes of the older Claude
tokenizer's scheme: a title-case span is emitted as `<^>` plus its lowercased
form, an all-caps span as `<^^>` plus its lowercased form, so that `The`/`the`
and `NASA`/`nasa` share vocabulary entries. Whole spans only; mixed case (`GaN`,
`WiFi`) is left literal, as is any span whose lowercasing is not exactly
invertible.

The code is placed OUTSIDE the span's markers, in the same pre-token:
`<^><|>the<|>`. Placing it inside, `<|><^>the<|>`, is a different chunk from
`<|>the<|>`, so the trainer learns a separate token for the title-case form and
the case duplication survives -- measured on a 250M English cell, 4,837 of
21,819 alphabetic entries were the same word in two or three cased forms.
Outside, the atomic sequence of `<|>the<|>` occurs inside `<^><|>the<|>` as a
suffix, so the trainer *can* cover the span with the same piece the lowercase
form uses:

    the   ->  [<|>the<|>]
    The   ->  [<^>] [<|>the<|>]    when the merged form is not worth a slot

It is not forced to. For a word whose title-case form is frequent enough, BPE
will still merge the code in and spend an entry on `<^><|>the<|>`, which is the
right call when the frequency justifies it. What the layout buys is the option,
over the band of words common enough to hold an entry but not common enough in
title case to earn a second one.

A span is coded only if it actually carries case (`istitle() or isupper()`).
That test is not redundant: `str.islower()` needs a cased character to be True,
so testing only for "not lowercase" lets uncased scripts through -- lowercasing
Arabic or Hangul is a no-op, the round trip therefore succeeds, and a `<^>`
lands in front of every word span in those languages. Measured on the Goldfish
slices that was 33,183 of 33,579 Arabic word spans and 46,137 of 46,657 Korean
ones, against 6,072 of 34,707 for English, which is real title case. It costs
tokens and nothing about it fails loudly: Arabic spent 2.3% more of them.

`min_caps_length` is the shortest span `<^^>` may cover and `single_char_shift`
decides whether a one-character span is shift-coded at all -- `<^>a` is two
tokens where `A` is one, so it is not obviously worth it, but it is what the
reported numbers were trained with and so it is the default.

Space elision still applies across a code. Two delimited spans separated only by
case codes had a space between them, so decode reads `<|> <^> <|>` as one elided
space with the code applying to the span that follows.

Scope: this is a SCRIPT-v3 scheme
---------------------------------
`get_boundary_pretokenizer` pins `ScriptEncodingV3`, and the dependency is
deeper than that parameter. Span identification reads v3's category folding --
word spans are `category == "LM" and combines_with_spaces`, digits are
`category == "N"` -- so under V1 or V2 both sets come out empty and the scheme
silently marks nothing. It also subclasses `ScriptPretokenizer`, putting the
`bytes_gpt4*` family out of reach (a `UTF8CharEnc` carries no script block to
classify a span from), and it rejects `regex_pattern`, which excludes the hybrid
`scriptenc_gpt4o_cb` as well.

Marker insertion, space elision and decode are all generic; span identification
is not. Applying the scheme to any pretokenizer means factoring
`split_unencoded_and_encode` so it composes with the base chunker rather than
replacing it, behind a per-character "space / word char / digit" classification
that the byte path would answer from `unicodedata.category`. That is a design
change to `script_bpe/` core and has not been done.
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

# "word" is not a member: word spans are always delimited. See the module docstring.
BoundaryTarget = Literal["punct", "digit"]


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
    # What is delimited besides word spans, which always are. Defaults to punctuation
    # alone: not the paper's headline arm, so a bare config cannot be mistaken for it, and
    # digits stay off because delimiting them interacts with `digit_handling`.
    boundary_targets: tuple[BoundaryTarget, ...] = ("punct",)
    # Case codes on word spans; see the module docstring. Off by default, so the plain
    # boundary scheme is what a config with no case options set gives you.
    shift_code: bool = False   # <^>  title case
    caps_code: bool = False    # <^^> all caps
    # Shortest span <^^> may cover. A single upper-case character is title case as well as
    # upper case and is picked up by the shift branch, so 2 is the lowest value that means
    # anything; higher leaves short acronyms literal.
    min_caps_length: int = 2
    # Whether a one-character span is shift-coded. `<^>a` is two tokens where `A` is one.
    single_char_shift: bool = True
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
        # Each code is registered only when it is enabled, so an unused code never occupies
        # an atomic slot.
        self.shift_token_id = self._register_token(self.SHIFT_TEXT) if self.config.shift_code else None
        self.caps_token_id = self._register_token(self.CAPS_TEXT) if self.config.caps_code else None
        self.caps_code_ids = {self.shift_token_id, self.caps_token_id} - {None}
        self.is_initial_char_tokens.update(self.caps_code_ids)
        blocks = self.config.script_config.blocks
        # the ~20 space-using writing systems, as letters
        self.word_script_ids = {b.script_id for b in blocks if b.category == "LM" and b.combines_with_spaces}
        self.digit_script_ids = {b.script_id for b in blocks if b.category == "N"}
        # kinds that carry a boundary and may therefore participate in space elision. WORD
        # is unconditional; the config only says what joins it.
        targets = set(self.config.boundary_targets)
        self.marked_kinds = frozenset(
            {self.WORD} | {k for k, name in ((self.PUNCT, "punct"), (self.DIGIT, "digit")) if name in targets}
        )

    def __init__(self, config: BoundaryScriptPretokenizerConfig) -> None:
        super().__init__(config)
        # Digit-group tokens are registered by the base _build_digit_tokens AFTER
        # _build_atomic_tokens runs, and ScriptPretokenizer.decode has no path for them
        # (digit_handling was only ever exercised with UTF8Pretokenizer). Collect their
        # ids here so decode can emit them directly; they are the only atomic tokens
        # whose text is all digits.
        self.digit_token_ids = {tid for tid, txt in self.atomic_tokens.items() if txt.isdigit()}

    def decode(self, tokenization, errors="replace") -> str:
        decoded = ""
        pending = None   # case code awaiting its span
        buf = ""
        i = 0
        n = len(tokenization)
        codes = self.caps_code_ids

        def emit(text):
            nonlocal decoded, buf
            if pending is None:
                decoded += text
            else:
                buf += text

        def flush():
            """Close a case-coded span at its terminating marker."""
            nonlocal decoded, pending, buf
            if pending is None:
                return
            decoded += (buf[:1].upper() + buf[1:]) if pending == "shift" else buf.upper()
            pending, buf = None, ""

        def kind(tid):
            return "shift" if tid == self.shift_token_id else "caps"

        while i < n:
            script_tok = tokenization[i]
            if script_tok in codes:
                flush()  # a code immediately after another closes the previous span
                pending, buf = kind(script_tok), ""
                i += 1
                continue
            if script_tok == self.marker_token_id:
                # A non-empty buffer means this marker closes a coded span. An empty one
                # means it opens the span the preceding code applies to, so it must not
                # flush -- the code sits outside the markers, not inside them.
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
                i += 1  # lone marker: structural boundary, no character
                continue
            if script_tok in self.digit_token_ids:
                emit(self.atomic_tokens[script_tok])  # digit group, single token
                i += 1
                continue
            ix_tok = tokenization[i + 1] if i + 1 < n else None
            if (script_tok, ix_tok) in self.detokenize_map:
                emit(self.detokenize_map[(script_tok, ix_tok)])
                i += 2
            else:
                if errors == "backslashreplace":
                    emit(self.atomic_tokens[script_tok])
                elif errors == "replace":
                    # emit, not `decoded +=`: inside a coded span the replacement character
                    # belongs in the buffer, or it would jump ahead of the span's own text.
                    emit("�")
                elif errors == "strict":
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok}) is not a valid token pair!")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        flush()  # span running to end of stream
        return decoded

    def _caps_form(self, text: str):
        """Return (code_kind, lowercased) for a span an enabled code applies to, else None.

        Two independent conditions. The span must actually carry case -- `istitle() or
        isupper()`, and not merely "is not lowercase", because `islower()` needs a cased
        character to be True and so lets every uncased script through; the module docstring
        has what that cost. And the lowercasing must be exactly invertible, because Unicode
        case mapping is not a bijection: 'I'.lower() is 'i' but Turkish dotless/dotted i
        break the pair, '\u0130'.lower() is two characters, and '\u1e9e'.lower() is '\u00df'
        whose upper is 'SS'. Every candidate is verified by re-applying the transform, and
        anything that does not reproduce the source exactly is left literal.

        Title case is tested before all caps, so a single upper-case character reaches the
        shift branch rather than the caps one.
        """
        if not text or not (text.istitle() or text.isupper()):
            return None
        low = text.lower()
        if len(low) != len(text):
            return None
        if low[0].upper() + low[1:] == text:
            if not self.config.shift_code:
                return None
            return None if len(text) == 1 and not self.config.single_char_shift else ("shift", low)
        if self.config.caps_code and len(text) >= self.config.min_caps_length and low.upper() == text:
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
                if kind == self.WORD and self.caps_code_ids:
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
            # After the markers, so the code sits outside them: <^><|>the<|>. Inside would
            # make the span a different chunk from the lowercase form and nothing could be
            # shared -- see the module docstring.
            if code_id is not None:
                out[0] = [CodeCharEnc(code_id)] + list(out[0])
            chunks.extend(out)
        return chunks

    def split_encoded(self, encoding: Sequence[CharEncT]) -> list[Sequence[CharEncT]]:
        # All chunking already happened in split_unencoded_and_encode.
        return [encoding]


# Named variants used by the experiments. All are ScriptEncodingV3 with
# enforce_char_boundaries=True, and differ only in what carries a boundary and whether the
# case codes are on. `w`/`wp`/`wpd` name the scope: word spans always, then punctuation,
# then digits. `_caps` adds both case codes.
CAPS = dict(shift_code=True, caps_code=True)
BOUNDARY_VARIANTS = {
    "bnd_w": dict(boundary_targets=()),
    "bnd_wp": dict(boundary_targets=("punct",)),
    "bnd_wpd": dict(boundary_targets=("punct", "digit")),
    "bnd_w_caps": dict(boundary_targets=(), **CAPS),
    "bnd_wp_caps": dict(boundary_targets=("punct",), **CAPS),
    "bnd_wpd_caps": dict(boundary_targets=("punct", "digit"), **CAPS),
}


def get_boundary_pretokenizer(name: str, **overrides) -> BoundaryScriptPretokenizer:
    """Build a named variant. `overrides` are passed to the config (e.g. digit_handling)."""
    from script_bpe.pretokenize.scriptencoding import ScriptEncodingV3

    if name not in BOUNDARY_VARIANTS:
        raise ValueError(f"unknown boundary variant {name!r}; have {sorted(BOUNDARY_VARIANTS)}")
    return BoundaryScriptPretokenizer(
        BoundaryScriptPretokenizerConfig(
            script_config=ScriptEncodingV3,
            **BOUNDARY_VARIANTS[name],
            **overrides,
        )
    )
