"""pynanochat — a tokenizer-agnostic eval framework on top of nanochat.

Give it any tokenizer satisfying `pynanochat.tokenizer.Tokenizer`; it pretrains a
nanochat base model on that tokenizer and returns the downstream metric
(DCLM CORE) plus bits-per-byte. It does NOT know how the tokenizer was
built — that's the caller's concern. A tokenizer-research package (scriptbpe,
etc.) depends on pynanochat and feeds its tokenizers in, not the other way round.

nanochat does the heavy lifting (budgeted depth-sized pretraining + faithful
CORE/bpb/sample eval). pynanochat is the glue:
  - tokenizer:  the contract + helpers (inject, write_token_bytes)
  - data_prep:  corpus -> parquet shards in nanochat's dataset layout
  - runner:     shell base_train + base_eval, harvest a comparable result
"""

from .runner import ExperimentResult, run_experiment
from .tokenizer import Tokenizer, inject, write_token_bytes
from .tokenizer_adapter import ScriptBPETokenizerAdapter

__all__ = [
    "run_experiment",
    "ExperimentResult",
    "Tokenizer",
    "inject",
    "write_token_bytes",
    "ScriptBPETokenizerAdapter",
]
