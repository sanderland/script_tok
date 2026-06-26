# C++ Tokenizer Fast Path

This directory contains prototype C++ tokenizer acceleration code.

Included:

- `script_bpe/fast/`: C++ BPE and Unigram sources plus Python wrapper.
- `script_bpe/fast/heap/min_max_heap.hpp`: Heap helper used by the C++ BPE code.
- `CMakeLists.txt` and `pyproject.toml`: Build/dependency context for the prototype.
- `tests/bpe/test_fast_tokenizer.py`: Fast-tokenizer smoke tests.

This code is not wired into the live package. Treat it as source material for a future C++ fast-path implementation branch.
