# WIP Branch Inventory

This branch contains work that is intentionally not integrated into the live package yet.

Current contents:

- `paper_utils/super/`: supertoken experiment scripts.
- `wip/cpp_tokenizer_fast_path/`: C++ tokenizer fast-path source snapshot.
- `wip/unigram_multiprocessing/`: unigram multiprocessing source snapshot.
- `WIP.md`: this inventory.

The source snapshots under `wip/` are reference material. They should be ported into fresh implementation branches before becoming package code.

## `paper_utils/hybrid`

Expected status: no WIP changes.

The hybrid/Mingram paper utilities should be represented by `main`. If `git diff main...wip -- paper_utils/hybrid` shows files, those are out of scope for this branch unless there is a specific new goal.

## Supertoken Work

`paper_utils/super/` contains:

- `train_supertokens.py`: Trains Unigram models with supertoken initialization, using line-level pretokenization for final training.
- `utils.py`: Registers the line pretokenizer and defines supertoken filters/cache helpers.
- `generate_results.py`: Loads supertoken models and generates scatter/top-token outputs.
- `run_experiments.sh`: Shell orchestration for supertoken runs.
- `EXPERIMENT_LOG.md`: Experiment notes and observations.

This is the most relevant saved work if supertokens are revived on top of Mingram, but it has not been refactored into a clean Mingram-native implementation.

## C++ Tokenizer Fast Path

`wip/cpp_tokenizer_fast_path/` contains:

- `script_bpe/fast/`: C++ BPE and Unigram prototype sources plus Python wrapper.
- `script_bpe/fast/heap/min_max_heap.hpp`: Heap helper used by the C++ BPE code.
- `CMakeLists.txt` and `pyproject.toml`: Build/dependency context for the prototype.
- `tests/bpe/test_fast_tokenizer.py`: Fast-tokenizer smoke tests.

This code is not imported by `script_bpe` today.

## Unigram Multiprocessing

`wip/unigram_multiprocessing/` contains:

- `script_bpe/tokenizers/unigram/trainer.py`: Unigram trainer version with multiprocessing changes.
- `paper_utils/benchmark_multiprocessing.py`: Benchmark harness for comparing multiprocessing settings.
- `analyze_timing.py`: Timing-analysis helper.

This code is not imported by `script_bpe` today.

## Explicitly Not Included

- Old `py-nanogpt` files; current `eval/py-nanochat` stays canonical.
- ToaST/caps code and tests.
- Generated outputs such as parquet files, notebooks, zips, result blobs, and scratch investigation files.
- `paper_utils/hybrid` follow-up scripts and plotting/table changes.
- Live integration of C++ fast-path code or unigram multiprocessing work.

## Suggested Cleanup Before PR

If this branch should become reviewable, prune it further:

1. Keep only the supertoken story.
2. Refactor supertoken code against the current Mingram/main APIs if it is still worth pursuing.
3. Move `wip/cpp_tokenizer_fast_path/` and `wip/unigram_multiprocessing/` into separate implementation branches if either speed path is revived.
4. Run the full test suite after any cleanup.

## Checks Run

- `uv run ruff check paper_utils/hybrid paper_utils/super tests/tokenizers/mingram/test_trainer_smoke.py`
- `uv run pytest tests/tokenizers/mingram/test_trainer_smoke.py -q`

The full test suite was not run for this WIP branch.
