# Design choices for the boundary-marker downstream run

Every choice made while running the flow in `README.md` on the CSCS Clariden cluster.
The first section lists the choices that depart from what the README says; the later
sections cover choices the README leaves to the operator, and defects found in the code
along the way. This file is updated as the run proceeds.

Run identity: branch `claude/fineweb-space-neighbors-k10ufw`, base commit `b957b76`,
account `a0229`, partition `normal`.

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
| bnd_wp | 311 / 512 |
| bnd_wpd | 399 / 512 |
| bnd_wpd_caps | 399 / 512 |

The same pattern holds for the checked-in Korean and Russian tokenizers, so it follows
from the marker scheme rather than from English text.

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

### 5. Tokenizer training runs one arm per job

The README shows a single `train_matched.py` call covering all arms. Each arm needs its
own pretokenized corpus, since the corpus directory keys on the pretokenizer hash, and
those builds are independent and long. `cluster/train_one_tokenizer.sbatch` runs one arm
per node so they proceed concurrently.

This required a change to `train_matched.py`: `--manifest-path`. The manifest is rewritten
by read-modify-write, so concurrent jobs would drop each other's entries.
`merge_manifests.py` folds the per-job files back into `manifest.json` and applies the
cross-arm matched-vocabulary check that a single-process run does at the end.

### 6. `eval_texts/en.json` was regenerated

`marker_experiments/eval_texts/` is gitignored, so a fresh clone has no held-out slice and
`train_matched.py`'s chars-per-token check reports nothing without saying why. Rebuilt
from the same 1 GB FineWiki English stream the grid uses, via
`finewiki1gb_grid.ensure_eval`: 500 documents, 3,602,925 characters, from 1,002,754,052
characters streamed. This reproduces the intended input rather than changing it, but the
file is not the byte-identical artifact used previously, since it is not in version
control.

## Choices the README leaves open

- Seeds 0, 1, 2 at depth 12, per the README's own example. The README notes that the
  MinGram table it compares against used 20 seeds and calls 3 seeds a direction rather
  than a result. Whether to extend past 3 is still open.
- `fineweb_en_5gb`, total vocabulary 34685, trainer `bpe`: the README's defaults.
- 128 workers for corpus pretokenization and BPE training. A `PretokenizedCorpus` holds
  128 partitions, so more workers than that do not help the training phase. Worth noting
  that the worker count is not free in the corpus build: every extra worker adds one more
  result to merge per block, so it trades encode time against merge time. That trade only
  became favourable after defect 5 below.
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

## Constants introduced

- `script_bpe/utils.py`: `WORKER_JOIN_TIMEOUT_S = 60`. Seconds to wait for workers to exit
  before killing the forkserver. 60 rather than longer because both call sites collect
  every result before joining, so the workers have nothing left to do, and because the
  stall fires on most builds here.
- `script_bpe/corpus/registry.py`: `CORPUS_BUILD_DEFAULT_WORKERS = 16`. The previous
  hardcoded cap, kept as the default so callers that pass no worker count behave exactly
  as before.

## Verification performed

- `smoke_test.py` over the 56 checked-in tokenizers: 0 failures, `write_token_bytes` `[ok]`
  on all 20 to which it applies. Log at `results/marker_downstream/logs/smoke_test_checkedin.log`.
- GPU leg on a checked-in `bnd_wpd` tokenizer: training and bits-per-byte completed
  (train bpb 3.196317, val bpb 3.208251) before CORE raised. This is what identified the
  CORE incompatibility.
- `plain` arm trained: vocabulary 34,685 (1710 atomic plus 32,975), 0 roundtrip failures,
  3.6360 characters per token on the held-out slice.
