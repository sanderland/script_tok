# pynanochat

A **tokenizer-agnostic** eval framework on top of [nanochat](https://github.com/karpathy/nanochat).
Give it any tokenizer; it pretrains a nanochat base model on that tokenizer and
returns **DCLM CORE** plus bits-per-byte.

pynanochat does **not** know how a tokenizer was built (BPE, unigram,
minimum-token models, byte-level, ...). That's the caller's concern. A tokenizer-research
package (scriptbpe, etc.) **depends on pynanochat** and feeds its tokenizers in —
not the other way round. pynanochat stays a stable eval substrate.

## The contract

Provide anything satisfying `pynanochat.tokenizer.Tokenizer`:

```python
encode(text, prepend=None, append=None) -> list[int]
decode(ids) -> str
get_bos_token_id() -> int
get_vocab_size() -> int
get_special_tokens() -> list[str]
id_to_token(id) -> str
```

Then:

```python
import pynanochat as png

png.inject(my_tokenizer)                 # general path: any tokenizer
png.write_token_bytes(my_tokenizer, tokenizer_dir)   # for the bpb metric
res = png.run_experiment(my_tokenizer, depth=20, budget=...)
print(res.core_metric, res.val_bpb)
```

(If a tokenizer happens to be rank-based BPE, `write_tiktoken_pickle` is an
optional fast path that nanochat loads natively — but `inject` covers everything.)

## Why it's thin

nanochat already does the hard parts, so pynanochat only adds glue:

| nanochat provides | pynanochat adds |
|---|---|
| budgeted depth-sized pretraining (`base_train.py`) | tokenizer contract + `inject`/`write_token_bytes` |
| faithful eval `--eval core,bpb,sample` (`base_eval.py`) | `runner`: shell those scripts, harvest a comparable result |
| pluggable tokenizer via `get_tokenizer()` | `data_prep`: corpus → parquet shards |

This is a **base-model** pipeline — no SFT/chat (CORE is a base benchmark; the
paper doesn't chat). nanochat is used as the training/eval engine; this package
injects tokenizers at runtime, so there is no nanochat fork.

```
pynanochat/
  tokenizer.py    # the contract + helpers (inject, write_token_bytes)
  data_prep.py    # corpus -> parquet shards in nanochat's dataset layout
  runner.py       # shell base_train + base_eval, harvest a comparable result
```

## Using it as a local dep (from scriptbpe)

Drop this directory into the scriptbpe repo and point scriptbpe's deps at it:

```toml
# scriptbpe pyproject.toml
[project]
dependencies = ["pynanochat"]

[tool.uv.sources]
pynanochat = { path = "eval/py-nanochat", editable = true }
```

or simply `pip install -e ./eval/py-nanochat`. nanochat itself isn't on PyPI:

```bash
pip install "nanochat @ git+https://github.com/karpathy/nanochat"
```

Editable means changes to pynanochat are picked up live while we iterate.

## Status

- [x] `write_token_bytes` — bpb byte-length table, faithful to nanochat's
      `tok_train.py` (int32[vocab], specials → 0).
- [x] `inject` — runtime tokenizer injection (no nanochat fork).
- [x] tokenizer-agnostic `Tokenizer` contract.
- [ ] `data_prep.write_parquet_shards` — confirm nanochat's parquet column/dir.
- [ ] `runner.run_experiment` — pick the budget knob (`--target-flops` vs
      `--target-param-data-ratio`) and parse CORE/bpb out of base_eval's output.
- [ ] optional `write_tiktoken_pickle` fast path for rank-based BPE.

## Not this

pynanochat is not a tokenizer implementation. The caller owns tokenizer training,
model loading, and experimental controls; pynanochat only runs the downstream
evaluation harness.
