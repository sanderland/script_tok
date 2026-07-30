# Downstream LM evaluation for boundary-marker tokenizers

Boundary markers buy **+2.14 % chars/token** on average over the SCRIPT-v3 baseline
across six languages (English +3.77 %), and caps codes reclaim **29 % of the
vocabulary** at no compression cost. Neither result says anything about language
modelling. This directory runs that check: for each tokenizer, pretrain a nanochat base
model and report **DCLM CORE** and **validation bits-per-byte**.

Everything here rides on the existing harness — `eval/py-nanochat` (`pynanochat`) and
`paper_utils/hybrid/downstream/run_downstream_eval.py`, the same path the MinGram
downstream table used. What this directory adds is the four things that path needs
before it can accept a boundary tokenizer:

| file | what it does |
|---|---|
| `boundary_tokenizer.py` | makes boundary tokenizers loadable in a fresh process |
| `train_matched.py` | trains one **vocabulary-matched** tokenizer per arm |
| `smoke_test.py` | every check that does not need a GPU |
| `run_arms.sh` | train → check → run → collect, for one arm × seed set |
| `collect_results.py` | parse run logs into a TSV |

## What has been verified, and what has not

Verified locally, on CPU, on all 56 existing tokenizers plus a freshly trained matched
set (`smoke_test.py`, 0 failures):

- fresh-process load by dotted class path, for both BPE and MinGram serialisations
- the `pynanochat.Tokenizer` contract, dense-id space, synthetic BOS at `n`, `vocab = n+1`
- round-trip through the adapter on marker-, caps-, digit- and mixed-script text,
  and on `tests/data/taylorswift.txt`
- batch encode agrees with single encode (`pretokenize.py` uses the batch path)
- the `vocab <= 65535` uint16 bound
- matched vocabulary across arms, as a hard failure

**Not verified:** the GPU leg — nanochat pretrain, CORE, bpb. The development container
has neither `torch` nor a GPU, so `write_token_bytes` (which uses `torch.save`) and
everything downstream of it are untested. `SMOKE=1` below is the first thing to run on
the cluster, and it exercises exactly that leg.

## Why a separate training step

The compression grid fixed `additional_vocab_size = 32768` for every arm, so the arms
end up with *different total* vocabularies — the boundary marker and the two caps codes
are extra atomic tokens:

```
plain          1710 atomic + 32768 = 34478
bnd_wpd        1711 atomic + 32768 = 34479
bnd_wpd_caps   1713 atomic + 32768 = 34481
```

That is the right control for measuring compression: every arm gets the same number of
*learned* merges. It is the wrong control downstream, where vocabulary size sets the
embedding and unembedding shapes, hence the parameter count, hence nanochat's
compute-optimal token horizon. `train_matched.py` matches the total instead and lets the
learned budget absorb the difference:

```
additional_vocab_size = total_vocab - len(pretokenizer.atomic_tokens)
```

The default total is **34,685**, the matched vocabulary of the MinGram downstream table,
so these runs sit on the same axis as that table. A three-token difference in 34 k would
not move CORE, but it is free to get exactly right, and `train_matched.py` exits non-zero
if the arms do not agree.

Do **not** reuse `marker_experiments/tokenizers/*.json.gz` for this. They are the
compression grid's tokenizers: unmatched by construction, and trained on FineWiki rather
than web text.

## Why `boundary_tokenizer.py` exists

`Pretokenizer.REGISTRY` is filled by `__init_subclass__`, so it only contains
`BoundaryScriptPretokenizer` in a process that imported the defining module. The harness
loads tokenizers in fresh subprocesses, where:

```python
BPETokenizer.load("..._bnd_wpd_bpe.json.gz")
# KeyError: 'BoundaryScriptPretokenizer'
```

`marker_experiments/downstream/boundary_tokenizer.py` imports the module for its
registration side effect and re-exports the tokenizer classes **unchanged** —
`BoundaryBPETokenizer is BPETokenizer`, no subclass, no behaviour change. So the fix is
one flag:

```
--tokenizer-class marker_experiments.downstream.boundary_tokenizer.BoundaryBPETokenizer
--tokenizer-class marker_experiments.downstream.boundary_tokenizer.BoundaryMinGramModel
```

Use the MinGram spelling for MinGram-trained models; `BPETokenizer.load` on one raises
`KeyError: 'merge_rules'`.

For the child process to import that path, the repo must be installed **editable**
(`uv sync` does this) — the child runs with `cwd` set to the nanochat clone, and picks up
the repo root from the editable install's `.pth`, not from `cwd`.

## Setup

This work lives on a branch, not on `main`. Clone that branch:

```bash
git clone -b claude/fineweb-space-neighbors-k10ufw \
    https://github.com/sanderland/script_tok.git
cd script_tok
```

(Branch `claude/fineweb-space-neighbors-k10ufw`, draft PR
[#7](https://github.com/sanderland/script_tok/pull/7). If you already have the repo:
`git fetch origin claude/fineweb-space-neighbors-k10ufw && git checkout claude/fineweb-space-neighbors-k10ufw`.)

Then the environment. The **editable** install is not optional: the eval's child
processes run with `cwd` set to the nanochat clone and find `marker_experiments.*`
through the editable install's `.pth`, not through `cwd`.

```bash
uv sync --extra downstream          # editable script_bpe + pynanochat, torch, deps
uv pip install "nanochat @ git+https://github.com/karpathy/nanochat"

# pynanochat shells into a vendored clone; the runner errors without it
git clone https://github.com/karpathy/nanochat eval/py-nanochat/vendor/nanochat
```

Verify the setup before asking for a GPU. This needs no GPU and takes seconds:

```bash
uv run python marker_experiments/downstream/smoke_test.py
```

Expect `0 failure(s)` over the checked-in compression tokenizers. It will note that
their vocabularies are unmatched — correct, and `train_matched.py` is what fixes it.
`write_token_bytes` reports `[ok]` here and `[skip]` without torch; on the cluster it
must be `[ok]`.

Hardware and budget: one H100 80 GB per job at `--depth 12`. `--depth 24` (what the
MinGram table used) needs more; drop `--device-batch-size` to 16/8/4 on OOM. Tokenizer
training wants ~90 CPUs and is a one-off shared across seeds. Disk: ~50 GB under
`$NANOCHAT_BASE` for shards and checkpoints, plus ~15 MB per tokenizer.

## Run

Start with the pipeline check. It is minutes, CORE comes out near random, and the point
is only that download → inject → train → eval → parse works for these tokenizers:

```bash
SMOKE=1 ARMS=plain,bnd_wpd DEPTH=12 marker_experiments/downstream/run_arms.sh
```

Then the real thing:

```bash
ARMS=plain,bnd_wpd,bnd_wpd_caps \
SEEDS=0,1,2 \
DEPTH=12 \
TRAIN_WORKERS=90 \
OUT=results/marker_downstream \
marker_experiments/downstream/run_arms.sh
```

`run_arms.sh` runs three steps: train the matched tokenizers, run `smoke_test.py` against
them (it stops before spending GPU hours if anything is off), then one
`run_downstream_eval.py` per arm × seed, logging to `$OUT/logs/<arm>_<trainer>_d<depth>_s<seed>.log`.
It skips any run whose log already contains a result, so it is resumable after a
pre-emption.

### Knobs

| var | default | notes |
|---|---|---|
| `ARMS` | `plain,bnd_wpd,bnd_wpd_caps` | `plain` is the SCRIPT-v3 baseline; also `bnd_w`, `bnd_wp`, and any `*_caps` |
| `SEEDS` | `0` | the MinGram table used 20 seeds per method |
| `TRAINER` | `bpe` | or `mingram` (`f = 1.15`) |
| `CORPUS` | `fineweb_en_5gb` | tokenizer-training corpus; `finewiki_en_1gb` reproduces the compression grid's domain |
| `VOCAB` | `34685` | matched total vocabulary |
| `DEPTH` | `12` | `model_dim = depth * 64` |
| `GPUS` | `1` | `>1` launches torchrun/DDP |
| `NUM_SHARDS` | `8` | train data shards to download |
| `TRAIN_WORKERS` | `nproc` | tokenizer-trainer workers |
| `NANOCHAT_BASE` | `~/.cache/nanochat_marker` | data, token bytes, checkpoints |
| `SMOKE` | `0` | tiny end-to-end run |

### Slurm

One arm × seed per job; `run_arms.sh` is idempotent, so an array over seeds works with
the arm list held fixed:

```bash
#!/bin/bash
#SBATCH --array=0-2
#SBATCH --gpus=1 --cpus-per-task=90 --time=12:00:00
export ARMS=plain,bnd_wpd,bnd_wpd_caps
export SEEDS=$SLURM_ARRAY_TASK_ID
export TRAIN_WORKERS=90
export OUT=results/marker_downstream
srun marker_experiments/downstream/run_arms.sh
```

Tokenizer training is shared across seeds and cached by output path, so only the first
job in the array pays for it. If jobs start simultaneously they will each build the
pretokenized corpus; run step 1 alone once first to avoid that:

```bash
uv run python marker_experiments/downstream/train_matched.py \
    --arms plain,bnd_wpd,bnd_wpd_caps --workers 90 \
    --eval-texts marker_experiments/eval_texts/en.json
```

That prints chars/token on the compression grid's held-out English slice, which is the
cheapest way to confirm the tokenizers came out as expected before any GPU time.

### Many seeds: pre-tokenize once

`script_bpe.encode` is pure Python, and with 20 seeds per arm the corpus gets encoded 20
times identically. `paper_utils/hybrid/downstream/pretokenize.py` encodes it once to
uint16 shards that `pynanochat.pretok_dataloader` reads:

```bash
uv run python paper_utils/hybrid/downstream/pretokenize.py \
    --tokenizer-path marker_experiments/downstream/tokenizers/fineweb_en_5gb_bnd_wpd_bpe_v34685.json.gz \
    --tokenizer-class marker_experiments.downstream.boundary_tokenizer.BoundaryBPETokenizer \
    --base-dir $NANOCHAT_BASE --out-dir nc_runs/pretok/bnd_wpd --workers 90
```

`run_arms.sh` does not wire this in — it uses the on-the-fly path with
`encode_workers`, which is self-contained and is what the smoke run exercises.

## Results

```bash
uv run python marker_experiments/downstream/collect_results.py \
    --logs-dir results/marker_downstream/logs --out results/marker_downstream/results.tsv
```

`pynanochat.run_experiment` returns an `ExperimentResult` but persists nothing, so the
logs are the record. The TSV columns (`method`, `seed`, `val_bpb`, `train_bpb`, `core`,
plus `task_*`) match what `paper_utils/hybrid/downstream_results.py` expects, so the
existing table generators can read it. Logs without a result block — pre-empted or OOM
jobs — are listed and skipped, and re-running `run_arms.sh` picks them up.

`collect_results.py` also prints a per-arm summary, so a finished sweep is readable
without opening the TSV:

```
  arm               n    val_bpb      CORE
  bnd_wpd           3     0.9876    0.1234
  plain             3     0.9990    0.1100
```

### What to send back

`$OUT/results.tsv`, `$OUT/logs/`, and `marker_experiments/downstream/manifest.json` (the
per-arm vocabulary sizes, train times, and chars/token — it is what shows the arms were
genuinely matched). The tokenizers themselves are reproducible from the manifest and need
not travel. Committing straight to the branch is fine; `tokenizers/` and `corpora/` are
gitignored, the TSV and manifest are not large.

## What to expect

The compression side of this work found, three times over, that reclaiming vocabulary
does not convert into compression: unifying `' the'`/`'the'` and `'The'`/`'the'` frees
thousands of slots, and BPE spends them on things worth almost nothing. The open
question is whether it converts into *modelling* quality, which compression cannot
answer. Two outcomes are informative:

- **bpb improves roughly in line with the +3.77 % English compression gain** — the gain
  is real and the vocabulary reclamation was worth doing.
- **bpb is flat while compression improved** — the marker moved token boundaries without
  making the sequence more predictable, and the compression gain is bookkeeping.

CORE at depth 12 is noisy; the MinGram table needed 20 seeds per method for
significance. Treat a 3-seed run as a direction, not a result.

## Note on the checked-in compression tokenizers

`marker_experiments/tokenizers/` holds the compression grid's tokenizers. Two files there
were originally written by the 1 GB grid and then silently overwritten by the 250 M-char
caps grid, which used the same filenames; they have been renamed to `caps250_*` to match
what they actually are, and `caps_grid.py` now prefixes its outputs. The 1 GB
`en_bnd_wpd` artifact is gone as a result — the 1 GB *results* in
`finewiki1gb_result.json` are unaffected (that cell recorded `unique_chunks=2,072,665`,
against 906,491 for the 250 M corpus), and nothing here depends on the artifact, since
every downstream arm is retrained by `train_matched.py`.
