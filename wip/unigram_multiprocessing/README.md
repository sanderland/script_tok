# Unigram Multiprocessing

This directory contains prototype multiprocessing code for Unigram training.

Included:

- `script_bpe/tokenizers/unigram/trainer.py`: Trainer implementation with multiprocessing changes.
- `paper_utils/benchmark_multiprocessing.py`: benchmark harness for comparing multiprocessing settings.
- `analyze_timing.py`: Timing-analysis helper.

This code is not wired into the live package. Treat it as source material for a future Unigram multiprocessing implementation branch.
