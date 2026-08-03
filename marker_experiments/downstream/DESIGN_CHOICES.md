# Design choices for the boundary-marker downstream run

Every choice made while running the flow in `README.md` on the CSCS Clariden cluster.
The first section lists the choices that depart from what the README says; the later
sections cover choices the README leaves to the operator, and defects found in the code
along the way. This file is updated as the run proceeds.

Run identity: branch `claude/downstream-lm-eval`, based on
`claude/fineweb-space-neighbors-k10ufw`, account `a0229`, partition `normal`. The vendored
nanochat clone is at `92d63d4`; it is gitignored and not a submodule, so every job records
its commit alongside the repo commit.

## Choices that differ from the README

### 1. CORE is not scored for the arms that mark punctuation

The README's headline deliverable is "DCLM CORE and validation bits-per-byte". CORE
cannot be computed for `bnd_wp`, `bnd_wpd` or `bnd_wpd_caps`.

nanochat's CORE language_modeling tasks build two prompts per example, one without the
continuation and one with it, and assert in `core_eval.batch_sequences_lm` that

    tokens_without == tokens_with[:len(tokens_without)]

A pretokenizer that marks a character according to the character following it does not
satisfy this. Under `bnd_wpd`, `':'` is token 1906 at the end of a string and token 1965
when a space follows, so appending `" Boston"` to `"Answer:"` changes a token that was
already emitted. `base_eval` raises on the first such example, and because the exception
ends the run, nothing is reported at all, bits-per-byte included.

Measured with `core_prefix_check.py` on the real CORE prompts, using nanochat's own
`render_prompts_lm` and the `core.yaml` task metadata, at 60 examples per task:

| arm | language_modeling examples that abort |
|---|---|
| plain | 0 / 512 |
| bnd_w | 0 / 512 |
| bnd_wpd | 399 / 512 |
| bnd_wpd_caps | 399 / 512 |
| bnd_wp (compression grid, not an arm here) | 311 / 512 |

The last row comes from `marker_experiments/tokenizers/en_bnd_wp_bpe_32k.json.gz`, a
32,768-additional-vocabulary FineWiki tokenizer. There is no matched `bnd_wp` arm; the row is
included because it places the punctuation-only variant between the two extremes.

The counts are identical for the Korean- and Russian-trained tokenizers. The prompts are the
English CORE set in every case, so what this shows is that the counts follow from the marker
scheme rather than from the tokenizer's training language. It is not a measurement on Korean
or Russian text.

Choice: arms in `CORE_SAFE_ARMS` (`plain`, `bnd_w`) are run with `--eval-modes core,bpb`,
and the rest with `--eval-modes bpb`. Three alternatives were considered and rejected:
dropping the 9 language_modeling tasks and reporting CORE over the remaining tasks, or
replacing the assertion with a longest-common-prefix start index, both of which produce a
number that is not the published DCLM CORE and is therefore not comparable to the MinGram
table the README compares against; and swapping the arms for CORE-safe ones, which would
not answer the question the experiment asks. Bits-per-byte is reported for every arm, and
it is the metric in which the README states both of its informative outcomes.

### 2. A fourth arm, `bnd_w`

The README specifies `plain`, `bnd_wpd`, `bnd_wpd_caps`. `bnd_w` is added because it is
the only boundary arm that keeps the CORE prefix property (0 / 512 above), so it supplies
one boundary arm with an unmodified CORE number next to `plain`. The default `ARMS` in
`run_arms.sh`, `submit_all.sh` and `train_tokenizers.sbatch` is now
`plain,bnd_w,bnd_wpd,bnd_wpd_caps`.

### 3. nanochat is not pip-installed

The README's setup step is `uv pip install "nanochat @ git+https://github.com/karpathy/nanochat"`.
That fails: the repository is flat-layout and setuptools cannot discover a package in it.
`runner.py`'s own docstring already says nanochat is vendored *rather than* pip-installed,
and the pipeline runs its scripts with `cwd` set to the vendored clone at
`eval/py-nanochat/vendor/nanochat` (commit `92d63d4`). The step is skipped. Every
dependency the vendored clone imports is present in the environment: `filelock`,
`kernels`, `psutil`, `pyarrow`, `rustbpe`, `tiktoken`, `wandb`, `numpy`.

nanochat pins `torch==2.9.1`; the environment has `torch 2.12.1+cu130`. Since nanochat is
not installed as a package the pin is not applied. Training and the bits-per-byte
evaluation both ran under 2.12.1.

### 4. Caches are on scratch, not in the home directory

The README's commands write to default locations under `$HOME`. Home on Clariden
(`/vast/users/cscs/$USER`) has a 50 GB quota and was already near full, and
`uv sync --extra downstream` needs about 10 GB on its own; the first attempt failed
partway through with a quota error. `cluster/env.sh` redirects everything:

| variable | location | why |
|---|---|---|
| `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR` | `/capstor/scratch/cscs/$USER/marker_downstream` | see the iopsstor measurement below |
| `HF_HOME`, `NANOCHAT_BASE` | `/capstor/scratch/cscs/$USER/marker_downstream` | large; holds ClimbMix shards and checkpoints |
| corpora | `results/corpora` | the repo's `results` symlink already points at capstor scratch |

capstor deletes files not accessed for 14 days. The paths are written out in `env.sh`
rather than derived from `$SCRATCH`, because `$SCRATCH` is already set to the iopsstor
path in this account's shell and inheriting it split the layout across two filesystems by
accident.

The environment was on iopsstor at first, on the theory that an IO-optimized mount suits
the many small reads of a Python import. That was wrong on 2026-07-31: a process sat in
Lustre `cl_sync_io_wait` for minutes partway through importing `wcwidth`, and a 200 MiB
`dd` with `oflag=direct` measured 84.2 MB/s write on iopsstor against 781 MB/s write and
1.1 GB/s read on capstor. A job whose workers each import this environment cannot absorb
that, so the environment moved to capstor. The same imports take 0.3 s on a compute node.

### 5. The reported bits-per-byte is not nanochat's

The README names the deliverable as "DCLM CORE and validation bits-per-byte". What is
reported as the primary figure is summed loss divided by the true UTF-8 length of the scored
text. nanochat divides instead by the summed byte length of the emitted tokens, taking each
token's length from decoding it alone.

Those differ for any scheme that elides a character between two tokens. The single space
between two delimited spans is rebuilt at decode time from the touching markers, so it
belongs to no token and the denominator is short: measured over the tokens the evaluation
actually scores, the shortfall is 13.7% for bnd_wpd and 0.06% for plain. The shortfall
tracks how much the scheme elides, which is what the scheme optimizes, so nanochat's own
figure ranks these tokenizers by how aggressively they compress rather than by model quality.
Both columns are reported; the corrected one is primary.

The correction is exact rather than approximate, and only because the byte table it uses is
the one the metric uses. `evaluate_bpb` masks the loss of any zero-byte token, and markers
decode to the empty string, so `special_aware_token_bytes` floors non-special tokens at one
byte and the same fictional byte appears in both the numerator's mask and the denominator.

### 6. Tokenizer training runs one arm per job

The README shows a single `train_matched.py` call covering all arms. Each arm needs its
own pretokenized corpus, since the corpus directory keys on the pretokenizer hash, and
those builds are independent and long. `cluster/train_one_tokenizer.sbatch` runs one arm
per node so they proceed concurrently.

This required a change to `train_matched.py`: `--manifest-path`. The manifest is rewritten
by read-modify-write, so concurrent jobs would drop each other's entries.
`merge_manifests.py` folds the per-job files back into `manifest.json` and applies the
cross-arm matched-vocabulary check that a single-process run does at the end.

### 7. `eval_texts/en.json` was regenerated

`marker_experiments/eval_texts/` is gitignored, so a fresh clone has no held-out slice and
`train_matched.py`'s chars-per-token check reports nothing without saying why. Rebuilt
from the same 1 GB FineWiki English stream the grid uses, via
`finewiki1gb_grid.ensure_eval`: 500 documents, 3,602,925 characters, from 1,002,754,052
characters streamed. This reproduces the intended input rather than changing it, but the
file is not the byte-identical artifact used previously, since it is not in version
control.

### 8. One `NANOCHAT_BASE` per run, not one per sweep

The README documents `NANOCHAT_BASE` as a single directory. Each run now gets
`<shared>/runs/<tokenizer_id>`, with the ClimbMix shards and the CORE bundle symlinked in
from the shared copy so neither is duplicated or re-downloaded. The reason is in the
discarded rounds below.

## Discarded rounds

Two complete sweeps were run and thrown away. Both are recorded because the numbers looked
plausible in each case, and because the second was caused by the fix for the first.

**Round 1 (jobs 2972879-90).** All 12 concurrent runs shared one `NANOCHAT_BASE` and
therefore one `<base>/tokenizer/token_bytes.pt`, which `bootstrap` writes and
`nanochat.tokenizer.get_token_bytes` reads. Runs were scored against whichever arm's byte
table happened to be on disk. The symptom was a bnd_wpd_caps run reporting bnd_wpd's val bpb
despite its own training loss. Logs in `results/marker_downstream/logs_invalid/`.

**Round 2 (jobs 2973952-63).** Completed cleanly and was internally consistent. The one-byte
floor introduced to stop marker loss being masked also un-masked BOS, which nanochat excludes
by design: the `special` test compares decoded strings, these tokenizers decode BOS to the
empty string, and it had been masked only incidentally by the `len("") == 0` behaviour the
floor removed. Specials are now identified by id. Logs in
`results/marker_downstream/logs_invalid_bos/`.

## Choices the README leaves open

- Seeds 0, 1, 2 at depth 12, per the README's own example. The README notes that the
  MinGram table it compares against used 20 seeds and calls 3 seeds a direction rather
  than a result. Whether to extend past 3 is still open.
- `fineweb_en_5gb`, total vocabulary 34685, trainer `bpe`: the README's defaults.
- 128 workers for corpus pretokenization and BPE training. A `PretokenizedCorpus` holds
  128 partitions, so more workers than that do not help the training phase. The worker count
  is not free in the corpus build either: every extra worker adds one more result to merge
  per block, so encode time falls and merge time rises. Merge time stopped being the
  constraint after the quadratic merge was fixed.
- One GPU per run, on a node held exclusively. `run_downstream_eval.py` sets encode
  workers to `(cpu_count - gpus) // gpus`, and `script_bpe.encode` is pure Python, so
  wall-clock is set by cores per run rather than by GPUs.
- The GPU pipeline check was run directly against a checked-in boundary tokenizer instead
  of through `SMOKE=1`, because `run_arms.sh` step 1 would have rebuilt the corpus that a
  running job was already building.

## Defects found in the code

Fixed here:

1. `train_matched.py` forwarded `corpus_base_dir=None` into `load_corpus_by_name`, which
   overrode that function's own default instead of falling back to it, so any registry
   corpus died with `TypeError: expected str, bytes or os.PathLike object, not NoneType`.
   The docstring already said the argument defaults to the repo's cache. Only the
   `--text-file` path guarded against this, which is why it had not surfaced.
2. Worker shutdown could hang forever, at `corpus/base.py` (Pool teardown) and at
   `tokenizers/bpe/trainer.py` (`p.join()`). The forkserver leaves exited workers unreaped,
   and the parent then blocks on a sentinel that never fires. Both were hit in one job: 8.6
   hours lost at the corpus site with all 16 workers already zombies, then again at the
   trainer site with 128 zombies. In both cases every result had already been collected, so
   the hang discarded finished work. This is not a rare race on this cluster: it reproduced
   on a 200 KB corpus with 4 workers.

   `join_workers` and `shutdown_pool` in `script_bpe/utils.py` bound the wait and then kill
   the forkserver. Killing the workers does not help, because they are already dead;
   killing the forkserver closes each worker's sentinel pipe, and CPython handles that case
   directly (`popen_forkserver.poll` catches the `EOFError` and sets returncode 255), so
   every outstanding join returns at once. `forkserver.ensure_running` starts a fresh
   forkserver for the next `Process()`. Both helpers report loudly rather than raising,
   since the results in hand are complete. Verified in job 2956988: `pool did not join
   within 60s` then `killed forkserver 83612`, corpus completed; then `4 worker(s) did not
   exit within 60s`, forkserver killed, `Done! 50 merge rules created`.
3. Interpreter shutdown could hang after the work finished. multiprocessing's atexit
   handling blocks on the same unreaped workers; one job sat idle for 18 minutes after
   producing its output, which on a one-arm-per-job layout holds a node until its walltime.
   `train_matched.py` now flushes and calls `os._exit(0)` once the tokenizers and manifest
   are on disk.
4. Corpus pretokenization ignored the caller's worker count: `min(os.cpu_count() or 4, 16)`
   was hardcoded, so `--workers 128` reached BPE training but not the corpus build. This is
   what made one `fineweb_en_5gb` build take 8.6 hours on a 288-core node. `num_workers` is
   now plumbed through `load_corpus_by_name` to every builder it dispatches to.
5. Corpus pretokenization was quadratic in the parent. `from_text_batches` merged each
   worker's result with `total_chunk_counts += part_chunk_counts`, and `Counter.__iadd__`
   follows its update with `_keep_positive()`, a full rescan of the accumulated counter.
   Each merge therefore cost O(len(total_chunk_counts)), which reaches 4.7M entries on
   fineweb_en_5gb, and the number of merges is blocks times workers. The parent ran at 91%
   CPU with all 128 workers idle, and py-spy showed it inside `_keep_positive`. Raising the
   worker count made this phase worse, not better: 16 workers gives 8,000 merges (measured
   8.6 hours), 128 workers gives 64,000. `Counter.update()` skips the rescan and is exactly
   equivalent here, since these counts are only ever positive. Measured 12x faster at 400
   merges over a 400,000-entry counter with identical output; the real run is 64,000 merges
   over 4.7M entries, where the gap is larger. The `plain` corpus was built with the old
   code and is still valid, because the two produce the same counter.
6. `run_arms.sh` treated a run as finished by grepping for `CORE metric`. A bits-per-byte
   run never prints that line, so those runs would have been repeated forever. The test is
   now the printed result block.

Noted, not changed:

- `README.md` still lists the `uv pip install nanochat` step described in section 3 above.

## Findings from the pre-launch audit

An independent audit of the harness was run before the second sweep. It found four things
that would have changed the published numbers, all now fixed.

1. Zero-byte tokens were dropped from the bpb numerator, for the marker arms only.
   `loss_eval.py` masks the loss of any token whose byte count is 0, and a boundary marker
   decodes to the empty string, so the cost of predicting where a boundary goes was not
   counted. Measured share of emitted tokens on ClimbMix validation text: plain 0.000%,
   bnd_w 0.034%, bnd_wpd 0.097%, bnd_wpd_caps 0.135%, so the effect scaled with the number
   of marker codes and flattered exactly the arms under test. `write_token_bytes` and
   `measure_byte_factor` now floor non-special tokens at 1 byte; the fictional byte appears
   in both the denominator and the factor and cancels.
   I had asserted the opposite to my collaborator, on a check against
   `tests/data/taylorswift.txt`, which happens to emit none of these tokens.
2. The byte factor was measured over a 32 MB prefix. Between 32 MB and 140 MB the running
   factor still moved by 1.2e-3 (plain) and 2.0e-3 (bnd_wpd), against a seed-to-seed spread
   of val bpb of 2e-4 to 5e-4, so the correction's own error exceeded the error bars. It is
   now measured over the whole validation shard (252,166,727 bytes) and cached in the shared
   directory, written through a temporary file and renamed.
3. The factor was measured after training, so any failure in a CPU-only measurement
   discarded about 5 GPU-hours. It now runs before training.
4. Every job wrote the same `results.tsv` with a plain truncate-and-write. Two jobs
   finishing together could interleave into one half-written file that the table generator
   would read without complaint. The write is now atomic.

Two reporting issues it raised, both now stated in the table caption rather than fixed in
code, because the conventions themselves are defensible:

- The three seeds vary weight initialization only. The data loader contains no RNG, and all
  three seeds of an arm end training at the identical dataloader position, so the reported
  interval understates run-to-run variance.
- Training is matched on tokens, so the amount of text each arm covers differs. Measured on
  the ClimbMix validation shard, bnd_w covers 92.2% of what plain covers, bnd_wpd 100.6% and
  bnd_wpd_caps 100.7%. The generalisation that a marker arm covers less text holds for one
  of the three. This is the equal-compute convention, and the figures are now computed by
  measure_text_stats.py rather than typed.

Smaller fixes: the shard-count check accepted any two shards, so a `--smoke` leftover could
silently reduce a declared 8-shard run to one shard; the table generator keyed the manifest
on arm alone, so a throwaway `tiny_*` entry could be reported under the real caption; the
CORE column was driven by a hardcoded arm list rather than by the data; `run_arms.sh` still
called `train_matched.py` without `--manifest-path`; and the Triton compile cache defaulted
to a home-directory path shared by all 12 nodes.

## Experiments beyond the README's scope

The README describes one English downstream comparison. Three things were added.

### Robustness checks on the downstream result

Six seeds for `bnd_wpd` rather than three, and a 32-shard single-epoch rerun of `plain` and
`bnd_wpd`. The first tests whether three seeds is enough (the mean moves 0.10 standard
deviations, so it is). The second tests whether the result depends on training 3.4 to 3.6
passes over the same 2 GB, which suppresses the mechanism better compression is supposed to
exploit (the gap holds, +0.00532 to +0.00497). Both are reported in the appendix, generated
from the TSVs by `make_tex_tables.py`.

The 32-shard runs used their own `NANOCHAT_BASE` and `OUT`. The loader consumes every shard
except the last, so adding shards to the shared directory would silently have changed what
an 8-shard run trains on.

### The unified multilingual grid

`train_multilang.py`, written by the project owner, trains one grid that the compression
tables and the downstream tables can both read: 5 arms per language, FineWeb 5 GB, total
vocabulary 34,685. Until now the compression grids used FineWiki 1 GB at fixed
`additional_vocab_size=32768` while the downstream arms used FineWeb 5 GB at fixed total
vocabulary, so the two sets of numbers in the paper were not about the same tokenizers.

Its concurrency safety is unique filenames per cell, which protects several machines each
with its own clone but not several jobs sharing one checkout, as on a cluster. Every cell
here therefore ran with `--no-commit` and one commit collects the results. `commit_cell`
also pushes to `origin`, where this account has no write access, so the in-job push would
have failed in any case; `--no-commit` sidesteps that rather than fixing it, and the fix is
not part of this branch.

21 cells were trained: `de`, `fi`, `ru`, `ar` at 5 arms each, plus the last English cell
`bnd_wp`. Jobs 2989992-95 on four nodes, five or six cells concurrently per node at 46
workers each, all COMPLETED with `fail=0` in 2h54m to 3h20m. Each cell was invoked as

    train_multilang.py --lang <lang> --arms <arm> --trainers bpe --workers 46 --no-commit

`--trainers bpe` is passed explicitly. The script's default has since become both trainers,
so a rerun without that flag would not reproduce these cells.

With the four English arms already present this makes 25: 5 languages by 5 arms, every one
at total vocabulary 34,685 with 0 round-trip failures, checked from the manifest fragments
rather than asserted.

Korean was trained by the project owner separately and is not part of this run.

### MinGram downstream

The downstream comparison was BPE only. `plain` and `bnd_wpd` were repeated with the MinGram
trainer to test whether the conclusion is specific to greedy merge training. Two arms rather
than four, because `plain` against `bnd_wpd` is the comparison that carries the claim:
`bnd_wpd` is the arm whose gain cannot come from spending more forward passes per byte.

It reproduces. Validation bits per true byte, seeds 0 to 2, mean and sample standard
deviation, with the BPE runs restricted to the same three seeds:

| trainer | plain | bnd_wpd |
|---|---|---|
| bpe | 0.885315 (0.000312) | 0.879999 (0.000487) |
| mingram | 0.883694 (0.000808) | 0.880507 (0.000625) |

Paired by seed, `plain` minus `bnd_wpd`, positive meaning `bnd_wpd` is lower: BPE +0.005317
with SE 0.000319, t(2) = 16.7; MinGram +0.003187 with SE 0.000209, t(2) = 15.2. The
direction holds under MinGram and the size is 60% of the BPE difference. Both sweeps are
paired the same way, one training-data-order permutation per seed shared across arms.

Jobs 2990349 and 2990350, four runs and two runs on one node each, COMPLETED in 2h01m35s
and 1h57m59s. The rows are in `paper/generated/results_mingram.tsv`, kept out of
`results.tsv` because `arm` is the grouping key everywhere downstream and does not carry the
trainer, so one file holding both would average a BPE and a MinGram tokenizer into a single
mean per arm. `collect_results.py` now refuses a log directory that mixes trainers.

CORE was scored for `plain` only, as in the BPE sweep. `plain` MinGram CORE is 0.1461,
0.1340 and 0.1388 against 0.1364, 0.1407 and 0.1380 for `plain` BPE. CORE resolves no
direction at this seed count in either sweep.

MinGram does not take an existing vocabulary. It trains its own BPE at
`additional_vocab_size * overshoot_factor` and prunes down with EM, so these are full
retrains rather than a transformation of the BPE tokenizers. The pretokenized corpus is
reused, since it keys on the pretokenizer hash and MinGram uses the same pretokenizer for a
given arm.

`mingram_preflight.py` must pass before any GPU time. It checks the vocabulary is exactly
34,685 (MinGram prunes down to a target, so stopping early is a real failure mode that BPE
does not have), round-trip on the held-out slice, chars/token within 8% of the BPE tokenizer
for the same arm, that only BOS carries zero bytes, and the CORE prefix property. That last
one is measured rather than inherited: `run_arms.sh`'s `CORE_SAFE_ARMS` default was measured
on BPE tokenizers, and MinGram segments with a dynamic program instead of greedy merge
replay, so an arm CORE-safe under BPE need not be under MinGram. The preflight is itself
validated against the BPE tokenizers, whose answers are known: `plain` 0/180 aborts and
`bnd_wpd` 142/180, matching the earlier measurement.

## Scheduling on this account

There are no CPU-only nodes on a0229. Every node carries 4 GPUs and a node-hour bills as 4
GPU-hours whether or not a GPU is touched, so a layout that looks like full use of a node can
be pure waste. Tokenizer training here is entirely CPU.

The first multilingual grid layout ran one cell per node across 26 nodes, spending 104
GPU-hours per wall-clock hour to run no GPU work. `cluster/pack_cells.sbatch` now runs
several cells concurrently on one node at `(288-8)/N` workers each, which cut that to 4
nodes.

Packing is not automatically cheaper. If the work scaled perfectly with cores, N cells on one
node would take N times as long and cost identical node-hours. It pays here only because a
single cell cannot saturate 288 cores: corpus building alternates parallel encode with a
serial merge in the parent, measured at 91% of one core with every worker idle.

### The downstream LM runs are GPU-bound

`cluster/run_one.sbatch` gave a whole node to each downstream run, on the reasoning that
`script_bpe.encode` is pure Python and so wall-clock is set by how many cores one run gets.
The completed BPE runs measure otherwise. `plain` and `bnd_wpd` both run 2553 steps; mean
per-step time is 1406 ms and 1774 ms, mean `bf16_mfu` is 34.4% and 27.5%, and per-step time
rises and falls with MFU. No line in either log reports a dataloader stall. The 30-minute
wall-clock difference between the two arms is the eval configuration, not the data path:
`plain` is scored on CORE and bits-per-byte and spends 53.7 minutes on the 22 CORE tasks,
`bnd_wpd` is scored on bits-per-byte alone and spends 3.0 minutes.

`cluster/pack_runs.sbatch` therefore runs up to 4 downstream runs on one node, one GPU each
through `CUDA_VISIBLE_DEVICES`, splitting the cores through a new `ENCODE_WORKERS` variable
that becomes `run_downstream_eval.py --encode-workers`. Left unset that flag defaults to
`cpu_count - 1`, which four co-resident runs would turn into 1148 worker processes on 288
cores.

Whether 70 workers per run still supply tokens fast enough to leave per-step time unchanged
is not measured. The per-step `dt` line makes it visible within minutes of a job starting;
the solo figures to compare against are 1406 ms for `plain` and 1774 ms for `bnd_wpd`.

## Constants introduced

- `script_bpe/utils.py`: `WORKER_JOIN_TIMEOUT_S = 60`. Seconds to wait for workers to exit
  before killing the forkserver. 60 rather than longer because both call sites collect
  every result before joining, so the workers have nothing left to do, and because the
  stall fires on most builds here.
- `eval/py-nanochat/pynanochat/runner.py`: `BYTE_FACTOR_SAMPLE_BYTES = 0`, meaning the
  whole validation shard. See audit finding 2 for why a prefix is not enough.
- `script_bpe/corpus/registry.py`: `CORPUS_BUILD_DEFAULT_WORKERS = 16`. The previous
  hardcoded cap, kept as the default so callers that pass no worker count behave exactly
  as before.

## Verification performed

- `smoke_test.py` over the 56 checked-in tokenizers: 0 failures, `write_token_bytes` `[ok]`
  on all 20 to which it applies. Log at `results/marker_downstream/logs/smoke_test_checkedin.log`.
- GPU leg on a checked-in `bnd_wpd` tokenizer: training and bits-per-byte completed before
  CORE raised, which is what identified the CORE incompatibility. The bits-per-byte values
  from that run predate the byte-accounting fixes and are not comparable to anything
  reported, so they are not quoted here.
- `plain` arm trained: vocabulary 34,685 (1710 atomic plus 32,975), 0 roundtrip failures,
  3.6360 characters per token on the held-out slice.
