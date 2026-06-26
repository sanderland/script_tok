# `hybrid_fast` Speed Snapshot

Source commit: `ebda70c52815c03de708f2eacfb91445731717f7`

This directory is an archival snapshot of the C++ fast-path work from the deleted `hybrid_fast` branch.

Included:

- `script_bpe/fast/`: C++ BPE and Unigram prototype sources plus Python wrapper.
- `script_bpe/fast/heap/min_max_heap.hpp`: heap helper used by the C++ BPE code.
- `CMakeLists.txt` and `pyproject.toml`: build/dependency context from the old branch.
- `tests/bpe/test_fast_tokenizer.py`: old fast-tokenizer smoke tests.

This code is not wired into the live package. Treat it as source material for a future clean `wip/fast-cpp` branch.
