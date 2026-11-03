# Design choices: Korean and Russian downstream runs

Every choice made in setting up the Korean and Russian language-model comparison, started
2026-09-17. The English runs this extends are in `paper/generated/results*.tsv`.

## What is being run

Per language, 4 schemes x 2 trainers x 3 seeds = 24 runs, and 48 in total.

| | |
|---|---|
| languages | Korean, Russian |
| schemes | `plain`, `bnd_w`, `bnd_wpd`, `bnd_wpd_caps` |
| trainers | BPE, MinGram |
| seeds | 0, 1, 2 |
| model | nanochat, 12 layers, one GPU per run, as in the English runs |
| metric | validation bits per true byte |

## Tokenizers

The 16 tokenizers are the `_quick` cells of the six-language grid, at total vocabulary
34,685, the same files the published intrinsic tables were computed from. They are
gitignored upstream and were restored from `fork/claude/mingram-five-arms`, where the files
the grid produced are committed to git.

Two things about the restore are worth stating plainly.

**The caps arm is the file the grid calls `bnd_wpd_extcapsfix`.** Upstream renamed that arm
to `bnd_wpd_caps` when the work moved to `main`, and dropped the earlier arm of the same
name, which placed the case code inside the span's markers. The 12 intrinsic cells upstream
reports as `bnd_wpd_caps` match the `extcapsfix` cells of the grid on training time and
vocabulary sizes. The restored file is named for the upstream arm.

**The stored configuration was relabelled, and the relabelling was verified rather than
assumed.** The August files serialize `boundary_targets` with a `word` entry, a `caps_codes`
flag and, for the caps arm, a class name that current code does not define, so current code
refuses to load them. `migrate_legacy_tokenizer_config.py` rewrites those few fields and
keeps the rewrite only if the August code reading the original file and current code reading
the rewritten file produce identical token ids over 200 documents of the language. All 12
passed, with the vocabulary unchanged at 34,685. Before that, the August and current
pretokenizers were compared directly: identical atomic tokens in identical order, and
identical output on 800 Korean and Russian documents.

## Training data

FineWeb-2, one language per model, from the copy at

    /capstor/store/cscs/swissai/infra01/datasets/swiss-ai/fineweb-2_0_1-quality_10-filterrobots

Not the ClimbMix data the English runs used, which is English. The files carry a `text`
column, which is what nanochat's loader reads, so they are linked in rather than converted.

`build_language_shards.py` links each language's shards into its own base directory and
writes `shard_provenance.json` beside them. The directory nanochat reads is called
`base_data_climbmix` and holds no ClimbMix here; that name is nanochat's, and the provenance
file is what records the truth.

**How much text each language gets.** Steps per run are fixed by the model: 2,553 steps of
524,288 tokens, 1,338,507,264 tokens per run, which is 653,568 rows of 2,048 tokens. How many
passes over the training text that takes depends on how many rows nanochat's loader makes
from it. The loader packs whole documents into rows, and when no buffered document fits the
space left in a row it crops one and discards the rest of it. So a pass yields fewer tokens
than the text holds, and the loss is larger for languages with longer documents.

Rows per training file were measured by running nanochat's own loader over one file with the
real `plain` BPE tokenizer:

| | rows from one file | share of the file's tokens kept | training shards | passes |
|---|---|---|---|---|
| English (published) | | | 8 | about 3.5, from the logs |
| Korean | 68,960 | 0.80 | 3 | 3.11 in the logs (3.16 predicted from the row count) |
| Russian | 18,816 | 0.53 | 10 | 3.47 predicted |

The Korean prediction from the row count agrees with the pass count in the Korean logs to
within 2%, which is the check on the method. Russian documents average about 4,450 characters
against 1,675 for Korean, so the loader crops much more of them.

Getting here took two wrong estimates, recorded because both changed a decision. The shard
counts were first sized from characters per token alone, assuming about 1.8e9 tokens per run
and no cropping: 3 Korean and 7 Russian shards, both meant to match English. When the first
run printed the real budget of 1.34e9 tokens, I recomputed without cropping and reported
Korean at about 2.4 passes and Russian at 7 shards at about 2.8, and Russian was changed to 6
shards on that basis. Both figures were wrong in the same direction: cropping means each pass
yields fewer tokens, so there are more passes than that calculation gave. Korean's logs show
3.11, and the loader measurement puts 6 Russian shards at about 5.8 passes. Russian was then
set to 10 shards. No Russian run had started at any point in this.

The 32-shard English appendix run, at under one pass, reproduced the English result, so the
remaining difference between 3.1 and 3.5 passes is unlikely to change a conclusion.

The comparison that carries the claim is between schemes within one language, and all schemes
of a language train on identical text, so the text budget does not enter it.

## No CORE

CORE is an English benchmark and is not scored. Current `run_arms.sh` scores no arm on CORE
in any case: the arms that mark punctuation break the prompt-prefix property its
language-modeling tasks assert, and scoring only the surviving arms would compare a subset.

## Scheduling

Account `a0229`, partition `normal`, four runs per node with one GPU each, submitted as one
job per (trainer, seed) holding that seed's four arms. A node bills 4 GPU-hours per hour
whether or not the GPUs are used, and a single run leaves most of the node idle, so one run
per node would cost four times as much. A finished job is a complete paired set
for one seed, which is the unit the comparison comes in.

Estimated cost, from the August measurement of 4 runs finishing in 2h01m: about 48 GPU-hours
per language.

## Two differences from the English runs, recorded because they are not visible in the numbers

- `run_arms.sh` gained an `ENCODE_WORKERS` passthrough. Without it each run sizes its
  tokenization pool for a whole node, and four co-resident runs would ask for about 1,150
  worker processes on 288 cores. The August launcher had this; the squash to `main` dropped it.
- `env.sh` no longer overwrites a caller's `NANOCHAT_BASE`. Each language has its own base
  directory holding its own text, and the August version would have silently pointed every
  language at one of them.

## Risks specific to this setup

- The training text and the tokenizers are chosen independently, so Korean tokenizers could
  be pointed at Russian text and would produce complete, plausible numbers. The launcher
  refuses to start unless the language recorded beside the shards matches the corpus name in
  the tokenizer file names.
- capstor deletes files not accessed for 14 days, which is what destroyed this project's
  environment, data shards, checkpoints and logs between 2026-08-04 and 2026-09-17. Anything
  worth keeping past a sweep should be copied off scratch.

## English retrain, for publishing the model weights

Every checkpoint of the published English runs was deleted from scratch between 2026-08-04
and 2026-09-17, and none had been copied anywhere else. The 24 runs behind the main table
(4 schemes x 2 trainers x 3 seeds) are retrained so that the weights can be published. The
retrained models are not the published ones: GPU training is not bit-for-bit reproducible,
so their bits per byte will be close to the published values but not equal to them. Their
run tags end in `_retrain`, which puts them in the `retrain` variant of the result files and
keeps them from ever being averaged with the published rows.

**Tokenizers.** The 8 English tokenizers the published models were trained with, restored
from `fork/claude/mingram-five-arms`: the full 5 GB sample (`fineweb_en_5gb_*`), not the
quick sample the current defaults use. `bnd_w` and `bnd_wpd` were relabelled by
`migrate_legacy_tokenizer_config.py`, with identical token ids to the August code over 200
English documents per file.

The caps arm cannot be relabelled. The published English caps models were trained with the
August scheme from before the fix for scripts without case, and current code implements only
the fixed scheme. `legacy_extcaps_pretokenizer.py` holds the August code for that one class,
with its base class renamed so that importing it leaves the current class in place, and
`boundary_tokenizer.py` imports it. The two caps files are loaded unmodified, and they give
token ids identical to the August code over 200 English documents per trainer.

**Data.** The same 8 ClimbMix training shards and the same validation shard, downloaded with
the command the published runs used. The sha256 of each shard is in `shard_provenance.json`.

**Check before any GPU time.** The published result files record each tokenizer's byte factor,
measured on the validation shard. Recomputing them from the restored tokenizers and the
downloaded shard reproduces all four BPE values exactly, to 16 digits. So the tokenizers, the
validation text and the measurement code are the ones behind the published numbers.

## Keeping the checkpoints

At the end of each job, `archive_checkpoints.py` copies each finished run's final model
weights, its metadata, its log and its tokenizer to
`/capstor/store/cscs/swissai/a0229/cmeister/script_tok_boundary/downstream_models/<sweep>/<tag>/`,
which is not purged. Each copy is kept only when its sha256 matches the source, and
`archive.json` records the hashes. Optimizer state is not copied: it is 1.6 times the size of
the weights and only matters for resuming training. The Korean jobs were submitted before this
step existed, so their weights are archived by hand after they finish.

## A defect fixed on the way

`precompute_byte_factors.py` loaded every tokenizer with the BPE class, so `--trainer mingram`
failed on the first file. `run_arms.sh` treats that step as optional, so the failure was
silent, and every MinGram run measured its own factor instead: the same number, computed once
per run rather than once per arm.
