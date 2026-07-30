"""The stock SCRIPT baseline, made able to use `digit_handling`.

`ScriptPretokenizer` does not support `digit_handling` at all -- it was only ever
exercised with `UTF8Pretokenizer` -- and fails in two independent places:

  * `split_encoded` raises ValueError on any chunk that is not entirely `ScriptCharEnc`,
    and a digit chunk is entirely `DigitsEnc`;
  * `decode` has no path for digit-group tokens, so every digit becomes U+FFFD.

Both are fixed here and nothing else differs from `scriptenc3_cb`, so the baseline and
the boundary variant are equally able to use digit splitting.

This lives in its own module rather than inside `digit_split_grid.py` because
`Pretokenizer.REGISTRY` is populated by `__init_subclass__`: a class defined in an
experiment script is only registered in a process that runs that script, so
`BPETokenizer.load` on one of the digit-axis baselines raised
`KeyError: 'DigitAwareScriptPretokenizer'` anywhere else. The registry is keyed by class
*name*, so moving the class does not invalidate tokenizers already trained with it.
"""

from script_bpe.pretokenize.pretokenizer import ScriptPretokenizer, ScriptPretokenizerConfig


class DigitAwareScriptPretokenizerConfig(ScriptPretokenizerConfig):
    cls: str = "DigitAwareScriptPretokenizer"


class DigitAwareScriptPretokenizer(ScriptPretokenizer, config_type=DigitAwareScriptPretokenizerConfig):
    """Stock baseline plus the two fixes `digit_handling` needs on ScriptPretokenizer."""

    def __init__(self, config):
        super().__init__(config)
        self.digit_token_ids = {tid for tid, txt in self.atomic_tokens.items() if txt.isdigit()}

    def split_encoded(self, encoding):
        if any(getattr(c, "script_id", 0) == -1 for c in encoding):
            return [encoding]  # a digit chunk is already its own pretoken
        return super().split_encoded(encoding)

    def decode(self, tokenization, errors="replace") -> str:
        decoded = ""
        i = 0
        n = len(tokenization)
        while i < n:
            if tokenization[i] in self.digit_token_ids:
                decoded += self.atomic_tokens[tokenization[i]]
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
                    raise ValueError(f"Invalid tokenization: ({script_tok}, {ix_tok})")
                else:
                    raise ValueError(f"Unknown error handling mode: {errors}")
                i += 1
        return decoded
