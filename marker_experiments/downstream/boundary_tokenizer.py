"""Loadable entry points for boundary-marker tokenizers.

`Pretokenizer.REGISTRY` is populated by `__init_subclass__`, so it only knows about
`BoundaryScriptPretokenizer` in a process that has imported the module defining it.
The downstream harness loads a tokenizer by dotted class path in a fresh subprocess:

    python -c "from script_bpe.tokenizers.bpe import BPETokenizer; \
               BPETokenizer.load('..._bnd_wpd_bpe_32k.json.gz')"
    KeyError: 'BoundaryScriptPretokenizer'

Importing *this* module registers the pretokenizer as a side effect, and it re-exports
the tokenizer classes unchanged, so pointing the harness here fixes the load with no
subclassing and no behaviour change:

    --tokenizer-class marker_experiments.downstream.boundary_tokenizer.BoundaryBPETokenizer

The aliases are the real classes, not subclasses: `BoundaryBPETokenizer is BPETokenizer`.
Nothing about encoding or decoding lives here -- only the import that makes the
registry complete. The same trick covers the baseline arm, which needs no registration
but is exported so every arm can use one uniform `--tokenizer-class` spelling.
"""

from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.tokenizers.mingram.model import MinGramModel

# The imports are the point: they run __init_subclass__ and fill REGISTRY.
# DigitAwareScriptPretokenizer is the baseline of the digit-handling axis; it is not one
# of the downstream arms, but registering it here means every checked-in tokenizer loads
# through this one module rather than only the boundary ones.
from marker_experiments.boundary_pretokenizer import BoundaryScriptPretokenizer
from marker_experiments.digit_pretokenizer import DigitAwareScriptPretokenizer  # noqa: F401

for _name in ("BoundaryScriptPretokenizer", "DigitAwareScriptPretokenizer"):
    assert _name in BoundaryScriptPretokenizer.REGISTRY, _name

BoundaryBPETokenizer = BPETokenizer
BoundaryMinGramModel = MinGramModel

__all__ = ["BoundaryBPETokenizer", "BoundaryMinGramModel", "BPETokenizer", "MinGramModel"]
