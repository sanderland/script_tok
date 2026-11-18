# Aligned loss evaluation

English, Korean and Russian are supported, with plain, bnd_w, bnd_wpd and
bnd_wpd_caps MinGram models. Each run compares identical raw-text prefixes
from BOS. All schemes must fit the context limit. This differs from the
original training and validation packing protocol.

## English on this Mac

```sh
NANOCHAT_DTYPE=float32 .venv/bin/python -m paper_utils.boundary.eval_aligned_spans \
  --languages en --docs 10000 --max-tokens 2048 --seeds 0,1,2 --device mps
```

The evaluator adds `results/magikarp/nanochat-source` to the Python path.
Override with `--nanochat-root` if needed. Use the checkpoint-compatible
nanochat source, commit 92d63d4, and install its dependencies.

## Korean and Russian where the validation shards are available

Run from a checkout containing this evaluator and
`paper_utils/boundary/legacy_extcaps_pretokenizer.py`:

```sh
NANOCHAT_DTYPE=float32 .venv/bin/python -m paper_utils.boundary.eval_aligned_spans \
  --languages ko,ru --docs 10000 --max-tokens 2048 --seeds 0,1,2 \
  --device cuda --download-missing
```

Default validation paths are the published holdouts on CSCS, Korean
`kor_Hang/000_00003.parquet` and Russian `rus_Cyrl/000_00010.parquet` under
`/capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots/data/output/`.
Override each using `--validation ko=/path/to/shard.parquet` and
`--validation ru=/path/to/shard.parquet`.

To run all three languages in one invocation use `--languages en,ko,ru`.
All three validation files must be available on that machine. Downloads use
`models/boundary-marker-lms` by default. Set `--models-root` and `--output-root`
to scratch storage on a cluster. Missing weights require several GB per scheme.
The evaluator checks for every requested validation file before starting.

## Outputs and interpretation

Each language writes to `results/magikarp/aligned_spans/{language}_10000_2048/`:

- `prepare_audit.json` records source/tokenizer hashes and exclusions.
- `plan.json` stores common prefixes, source spans and dense token IDs.
- `{variant}_seed{seed}.npz` stores per-token losses, with provenance in `.json`.
- `summary.json` reports BPB and grouped plain-minus-boundary loss differences.

Every source byte and every token loss, including boundary and case markers,
belongs to exactly one span. If a token crosses a proposed lexical boundary,
the adjacent units are merged in every scheme. Such spans may be classified
as mixed. Token-count groups include structural tokens.

The published tokenizers do not preserve every Unicode source string.
A document prefix that fails any tokenizer's encode/decode round trip is
excluded from every scheme and recorded. Alignment failures on otherwise
round-tripping text still raise errors.

Frequency groups use complete-pass training counts, not seed-specific training
exposure. They apply only to single-token plain words. Counts are optional;
missing counts omit these groups. Group dimensions overlap, so contributions
must not be summed across different dimensions. Positive gain means lower
boundary-model BPB. These are descriptive attributions, not causal estimates.

Use `--prepare-only` to check tokenization without inference and `--report-only`
to regenerate summaries from saved losses. Completed model/seed results are
reused after provenance checks. An interrupted model/seed is rerun.
The earlier `eval_aligned_words` remains an English-only plain-versus-bnd_w
analysis; it now also audits and excludes non-roundtripping prefixes.
