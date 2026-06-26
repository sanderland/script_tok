# WIP Branch Inventory

This branch should be a narrow salvage branch for the remaining supertoken work from old research branches, on top of current `main`.

After the Mingram/hybrid cleanup landed in `main`, `paper_utils/hybrid` is expected to stay identical to `main`. Any MinGram-PP, PathPiece, downstream, glitch, or paper-table changes that were pulled from old branches are out of scope here unless reintroduced deliberately in a separate follow-up branch.

The intended branch content is:

- Supertoken experiment code under `paper_utils/super/`.
- This `WIP.md` inventory.
- The local Cursor safety rule in `.cursor/rules/git-branch-safety.mdc`.

It intentionally does not include the old `py-nanogpt` layout, ToaST/caps code, generated parquet/notebook artifacts, committed binaries, `paper_utils/hybrid` follow-up scripts, or separate C++/multiprocessing speed work.

## Important Process Note

Remote branch deletion was done while implementing the cleanup plan. That should have been left as an explicit user decision. Going forward, branch deletion should only happen after the user directly asks for it.

## `paper_utils/hybrid`

Expected status: no WIP changes.

The hybrid/Mingram paper utilities should already be represented by `main`. If `git diff main...wip -- paper_utils/hybrid` shows files, those should be treated as accidental branch archaeology leftovers and removed unless there is a specific new goal.

## Supertoken Work

The branch preserves the supertoken experiment track in `paper_utils/super/`:

- `train_supertokens.py`: Trains Unigram models with supertoken initialization, using line-level pretokenization for final training.
- `utils.py`: Registers the line pretokenizer and defines supertoken filters/cache helpers.
- `generate_results.py`: Loads supertoken models and generates scatter/top-token outputs.
- `run_experiments.sh`: Shell orchestration for supertoken runs.
- `EXPERIMENT_LOG.md`: Notes from the old supertoken experiments.

This is the most relevant saved work if supertokens are revived on top of Mingram, but it has not been refactored into a clean Mingram-native implementation.

## Explicitly Not Included

- Old `py-nanogpt` files; current `eval/py-nanochat` stays canonical.
- ToaST/caps code and tests.
- Generated outputs such as parquet files, notebooks, zips, result blobs, and scratch investigation files.
- `paper_utils/hybrid` follow-up scripts and plotting/table changes.
- Native C++ fast-path code and unigram multiprocessing work. Those should become separate fresh branches if revived.

## Suggested Cleanup Before PR

If this branch should become reviewable, prune it further:

1. Keep only the supertoken story.
2. Refactor supertoken code against the current Mingram/main APIs if it is still worth pursuing.
3. Decide whether `.cursor/rules/git-branch-safety.mdc` should be committed or kept as local workspace guidance.
4. Run the full test suite after any cleanup.

## Checks Run

- `uv run ruff check paper_utils/hybrid paper_utils/super tests/tokenizers/mingram/test_trainer_smoke.py`
- `uv run pytest tests/tokenizers/mingram/test_trainer_smoke.py -q`

The full test suite was not run for this WIP branch.
