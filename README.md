# SCRIPT: Script/Category Representation In (Pre-)Tokenization

This repository provides tools for tokenization, focused on SCRIPT encoding, but also supporting UTF-8.
It contains tokenizers for BPE and Unigram.

For details of the methods, see our paper:
* [BPE Stays on SCRIPT: Structured Encoding for Robust Multilingual Pretokenization](https://arxiv.org/abs/2505.24689)

## Overview

This repository provides tools for SCRIPT encoding-based pre-tokenization and BPE, as well as regular byte-based BPE.
It includes the following components:

- **script_bpe**: Core modules for SCRIPT encoding and tokenization.
  - `pretokenize/`: Pre-tokenizers: These handle both chunking and encoding to 'base tokens' (i.e. bytes or script/index)
    - `bytes_gpt4`/`bytes_gpt4o`: Classic regex + UTF8 based tokenizer, most obvious point of reference.
      - `bytes_gpt4o_cb` Variant with character boundaries merge limitations which prevents partial+full character merges, and enforces left-to-right merging within characters.
      - `bytes_nosplit_cb` Variant with no pre-tokenization chunking. Very slow, mainly for limited ablations.
    - `scriptenc` SCRIPT Encoding based encoding and pre-tokenization 
       - `scriptenc_cb` Variant with character boundaries merge limitations (no partial+full merges, enforce merging into a full character first). This is the proposed algorithm.
       - `scriptenc_gpt4o_cb` Variant which does pre-tokenization chunking with regex, but then uses script encoding. For ablation testing.
       - `scriptenc_nosplit_cb` Variant with no pre-tokenization chunking. Very slow, mainly for limited ablations.
    - Regex chunked, script encoded.
  - `encoding/`: SCRIPT Encoding utilities.
  - `bpe/`: Byte Pair Encoding (BPE) implementation.
    - `stats` for tokenizer performance metrics.
  - `corpus/`
    - `PretokenizedCorpus` represents a pretokenized sharded training dataset, as `base token encoded chunk -> count`

## Usage

### Installation

Ensure you have [uv](https://docs.astral.sh/uv/), it should take care of the rest.

### Training

To explore the available options for training, run:

```bash
uv run train --help
```

To train a tokenizer using a specific corpus, use:

```bash
uv run train --corpus <kor_hang_300mb> -n <number of merge rules> --pretokenizer <pretokenizer_name> 
```

### Reproducing results

The directory `paper_utils` contains scripts to train and reproduce the paper's results from scratch.

## Sources

* An interesting explanation of UTF-8 is given by [Computerphile](https://www.youtube.com/watch?v=MijmeoH9LT4)
* For more information on Unicode character properties, refer to the [Wikipedia article](https://en.wikipedia.org/wiki/Unicode_character_property#General_Category).

