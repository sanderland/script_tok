"""Adapter: a Hugging Face `tokenizer.json` behind the interface evaluate_ngram_bpb needs.

The n-gram eval only touches four things -- encode, decode, a `tokens` mapping for
vocabulary geometry, and a loadable (path, class) pair for the worker pool -- so external
tokenizers don't need script_bpe's model classes to be scored. Encoding goes through the
Rust `tokenizers` runtime, which also applies whatever normalizer (e.g. NFC) the file
declares; that matters because a tokenizer must be scored under its own normalization.

Specials are NOT added at encode time. The TokEval release ships no post-processor and
prepended BOS manually in training; the n-gram harness adds its own BOS/EOS above the
vocabulary, so any BOS added here would be double-counted.
"""

from tokenizers import Tokenizer


class HFTokenizerAdapter:
    def __init__(self, path: str):
        self._tok = Tokenizer.from_file(path)
        self._path = path
        # Dense mapping over the full id space. get_vocab_size(with_added_tokens=True)
        # covers specials, which occupy ids whether or not the harness ever emits them --
        # geometry must span every id encode() could produce.
        self.tokens = dict.fromkeys(range(self._tok.get_vocab_size(with_added_tokens=True)))

    @classmethod
    def load(cls, path: str) -> "HFTokenizerAdapter":
        return cls(path)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)
