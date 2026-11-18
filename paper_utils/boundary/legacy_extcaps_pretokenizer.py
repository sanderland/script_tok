"""The August 2026 `ExtCapsBoundaryScriptPretokenizer`, kept so the English caps tokenizers load.

The English downstream results for `bnd_wpd_caps` (paper/generated/results_caps.tsv and
results_mingram.tsv) were trained with tokenizers whose case code sits outside the span's
markers but whose case test is the pre-fix one: `not text.islower()`. `islower()` is False
for text with no cased character, so that test codes every span in a script without case.
The fix (`istitle() or isupper()`) landed as `_extcapsfix` and is the only scheme current
`boundary_pretokenizer.py` implements. There is therefore no current equivalent of these
two tokenizers, and a config rewrite like the one in
`downstream/migrate_legacy_tokenizer_config.py` cannot make them loadable. For English text
the two tests differ only on spans with no cased character, which is why the fix left
English compression unchanged at the precision reported, but a retrain of the published
models has to use the published tokenizers.

This file is the August code, copied from
`fork/claude/mingram-five-arms:marker_experiments/boundary_pretokenizer.py`
(sha256 0d70687df8c53b57a78729f3921e74a654b413bbb948317f09ace69d47d6fc16), lines 55 to 500:
the base class and the pre-fix outside-placement class, nothing after. One change was made.
`Pretokenizer.REGISTRY` is keyed on the class name and a later registration silently
replaces an earlier one, so the August base class, which has the same name as the current
one, is renamed here to `LegacyBoundaryScriptPretokenizer`. Otherwise importing this module
would replace the current class and break every current boundary tokenizer. The subclass
keeps its August name, `ExtCapsBoundaryScriptPretokenizer`, because that is the name stored
in the tokenizer files.

Do not use this for anything new.
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


class LegacyBoundaryScriptPretokenizerConfig(ScriptPretokenizerConfig):
    cls: str = "LegacyBoundaryScriptPretokenizer"
    boundary_targets: tuple[BoundaryTarget, ...] = ("word", "punct", "digit")
    # Caps codes, in the style of the older Claude tokenizer: a title-case word is emitted
    # as a shift code plus its lowercased form, an all-caps word as a caps-lock code plus
    # its lowercased form, so 'The'/'the' and 'NASA'/'nasa' share vocabulary entries. Whole
    # spans only; mixed case ('GaN', 'WiFi') is left literal.
    caps_codes: bool = False
    model_config = ConfigDict(extra="forbid")


class LegacyBoundaryScriptPretokenizer(ScriptPretokenizer, config_type=LegacyBoundaryScriptPretokenizerConfig):
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

    def __init__(self, config: LegacyBoundaryScriptPretokenizerConfig) -> None:
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
            raise NotImplementedError("LegacyBoundaryScriptPretokenizer does not support regex_pattern")
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


class ExtCapsBoundaryScriptPretokenizerConfig(LegacyBoundaryScriptPretokenizerConfig):
    cls: str = "ExtCapsBoundaryScriptPretokenizer"


class ExtCapsBoundaryScriptPretokenizer(
    LegacyBoundaryScriptPretokenizer, config_type=ExtCapsBoundaryScriptPretokenizerConfig
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
