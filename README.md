# SCRIPT: Script/Category Representation In (Pre-)Tokenization

This repository provides tools for tokenization, focused on SCRIPT encoding, but also supporting UTF-8.
It contains implementations for **BPE**, **Unigram**, **MinGram**, **PathPiece**, and **ConvexTok** tokenization algorithms.

For details of the methods, see the papers:
* [Explicit Boundary Markers for Subword Vocabularies](https://arxiv.org/abs/2608.08847)
* [MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignment](https://arxiv.org/abs/2606.27019)
* [Which Pieces Does Unigram Tokenization Really Need?](https://arxiv.org/abs/2512.12641)
* [BPE Stays on SCRIPT: Structured Encoding for Robust Multilingual Pretokenization](https://arxiv.org/abs/2505.24689)

## Overview

This repository provides tools for SCRIPT encoding-based pre-tokenization with BPE and Unigram, as well as regular byte-based tokenization.

### Core Modules (`script_bpe/`)

- **`pretokenize/`**: Pre-tokenizers that handle both chunking and encoding to 'atomic' or 'base' tokens (bytes or script/index pairs)
  - `bytes_gpt4`/`bytes_gpt4o`: Classic regex + UTF-8 based tokenizer
  - `bytes_gpt4o_cb`: With character boundaries enforcement
  - `scriptenc_cb`: SCRIPT encoding with character boundaries (proposed BPE algorithm)
  - `scriptenc_cbi`: SCRIPT encoding with inherited script enforcement
  - `scriptenc_gpt4o_cb`: Hybrid (regex chunking + script encoding)

- **`tokenizers/`**: Tokenization algorithms
  - `bpe/`: Byte Pair Encoding implementation with multi-worker training
  - `unigram/`: Unigram language model with EM training, Trie, and Lattice-based Viterbi decoding
  - `mingram/`: MinGram hard-EM tokenizer and MinGram-PP via MI pruning
  - `pathpiece/`: PathPiece-style minimum-token inference baseline
  - `convextok/`: ConvexTok vocabulary-selection baseline

- **`corpus/`**: Pretokenized corpus management
  - `PretokenizedCorpus`: Partitioned storage for efficient parallel training

- **`analysis/`**: Evaluation utilities
  - Compression metrics, morphological scoring, experiment tracking

## Usage

### Installation

Ensure you have [uv](https://docs.astral.sh/uv/), it should take care of the rest.

For MorphAlign evaluations, you need to initialize the submodule with:

```bash
git submodule update --init eval/morph-tok-eval
```

### Training

To explore the available options for training, run:

```bash
uv run train --help
```

To train a BPE tokenizer:

```bash
uv run train --corpus kor_hang_300mb -n 64000 --pretokenizer scriptenc_cb --model bpe
```

To train a Unigram tokenizer:

```bash
uv run train --corpus kor_hang_300mb -n 64000 --pretokenizer scriptenc_cb --model unigram
```

### Reproducing Paper Results

The `paper_utils/` directory contains scripts to reproduce paper results from scratch:

- **`paper_utils/script_bpe/`**: BPE paper reproduction
  - Paper: [BPE Stays on SCRIPT](https://arxiv.org/abs/2505.24689)
  - `train_monolingual.sh` / `train_multilingual.sh`: Training scripts
  - `monolingual_compression.ipynb` / `multilingual_compression.ipynb`: Analysis notebooks

- **`paper_utils/unigram/`**: Unigram Pieces paper reproduction
  - Paper: [Which Pieces Does Unigram Tokenization Really Need?](https://arxiv.org/abs/2512.12641)
  - `run_all_experiments.sh`: Run all experiments
  - `generate_main_tables.py` / `generate_appendix_tables.py`: Generate paper tables
  - `train_hyperparameters.py`: Hyperparameter tuning experiments

- **`paper_utils/hybrid/`**: MinGram tokenizer paper reproduction
  - Paper: [MinGram: A Minimalist Unigram Tokenizer with High Compression and Competitive Morphological Alignment](https://arxiv.org/abs/2606.27019)
  - `run_all_experiments.sh`: Train the active tokenizer set and regenerate paper tables/figures.
  - `build_token_usage_counts.py`: Rebuild the rare-token count parquet used by downstream tables.

- **`paper_utils/boundary/`**: Boundary-marker paper reproduction
  - Paper: [Explicit Boundary Markers for Subword Vocabularies](https://arxiv.org/abs/2608.08847)
  - `run_all_experiments.sh`: Regenerate every paper table. The default run reads the
    measurement caches committed under `paper/generated/`, so it needs no GPU and no
    trained tokenizer; `GRID=1` retrains the tokenizer grid and `DOWNSTREAM=1` reruns the
    LM sweep.
  - `boundary_pretokenizer.py`: `BoundaryScriptPretokenizer`, which delimits word spans
    with a single `<|>` atomic token and elides the space between two delimited spans.
    Optional `<^>`/`<^^>` case codes place a title- or upper-case span's code outside its
    markers, so `The` is `the` plus one code.
  - `make_intrinsic_table.py` / `downstream/make_tex_tables.py`: the two main tables.
  - The grid trains on the `quick` corpus sample (`fineweb_<lang>_5gb_quick`), which reads
    until its character budget is full instead of reservoir-sampling the whole source.

## Sources

* An interesting explanation of UTF-8 is given by [Computerphile](https://www.youtube.com/watch?v=MijmeoH9LT4)
* For more information on Unicode character properties, refer to the [Wikipedia article](https://en.wikipedia.org/wiki/Unicode_character_property#General_Category).
