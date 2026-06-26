# `refactor_mp` Multiprocessing Snapshot

Source commit: `df33ccae62ba17001547f87476436ad94bf4b20a`

This directory is an archival snapshot of the unigram multiprocessing work from the deleted `refactor_mp` branch.

Included:

- `script_bpe/tokenizers/unigram/trainer.py`: old trainer implementation with multiprocessing changes.
- `paper_utils/benchmark_multiprocessing.py`: benchmark harness for comparing multiprocessing settings.
- `analyze_timing.py`: timing-analysis helper from the old branch.

This code is not wired into the live package. Treat it as source material for a future clean `wip/unigram-mp` branch.
