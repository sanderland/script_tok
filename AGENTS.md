# AGENTS.md - Codebase Architecture Guide

This guide provides a high-level architectural overview of the script_bpe repository to help AI Coding Assistants work effectively in this codebase.

## Project Overview

This repository implements **SCRIPT encoding-based pre-tokenization and BPE/Unigram tokenization** for multilingual text processing. The core innovation is using Unicode script properties to create more efficient and robust tokenizers for multilingual data.

**Paper**: [BPE Stays on SCRIPT: Structured Encoding for Robust Multilingual Pretokenization](https://arxiv.org/abs/2505.24689)

## High-Level Architecture

The codebase follows a **layered architecture** with clear separation of concerns:

```
Text Input
    ↓
[Pretokenization Layer] - Chunks text and encodes to base tokens
    ↓
[Corpus Layer] - Pretokenized, partitioned training data
    ↓
[Training Layer] - BPE or Unigram algorithm
    ↓
[Tokenizer Layer] - Trained model for encode/decode
```

## Module Structure

### 1. Pretokenization (`script_bpe/pretokenize/`)

**Purpose**: Transform raw text into sequences of "base tokens" (atomic units for tokenization)

**Key Concepts**:
- **Base tokens**: Atomic units - either bytes (UTF-8) or script/index pairs
- **Pretokenization**: Chunking text into segments before encoding
- **Encoding**: Converting characters to base token sequences

**Two Main Approaches**:

1. **UTF-8 Pretokenizer** (`UTF8Pretokenizer`):
   - Encodes each byte as a base token
   - Uses regex patterns (GPT-4, GPT-4o) for chunking
   - Base token = single byte

2. **Script Pretokenizer** (`ScriptPretokenizer`):
   - Uses Unicode script properties to encode characters
   - Each character → (script_block_id, index_within_block)
   - Base token = pair of script tokens
   - Can optionally chunk by script boundaries instead of regex

**Configuration System**:
- `PretokenizerConfig`: Base configuration (normalization, regex, etc.)
- `ScriptPretokenizerConfig`: Script-specific settings
- `UTF8PretokenizerConfig`: UTF-8-specific settings
- Registry in `pretokenize/__init__.py` maps names → configs

**Key Variants** (see `PRETOKENIZER_REGISTRY`):
- `bytes_gpt4`: Classic UTF-8 + GPT-4 regex
- `bytes_gpt4o_cb`: UTF-8 + GPT-4o regex + character boundaries
- `scriptenc_cb`: Script encoding with character boundaries (PROPOSED METHOD)
- `scriptenc_gpt4o_cb`: Hybrid (regex chunking + script encoding)
- `scriptenc_nosplit_cb`: No regex chunking (very slow, for ablations)

**Character Boundary Enforcement** (`enforce_char_boundaries`):
- Prevents merges that would create invalid UTF-8 sequences
- Three levels:
  - `False`: No restrictions (can create partial characters)
  - `True` (cb): Only complete characters allowed
  - `enforce_inherited=True` (cbi): Also prevents inherited scripts at start

**Script Encoding** (`scriptencoding.py`):
- `ScriptConfig`: Defines how Unicode scripts/categories map to blocks
- `ScriptBlock`: A group of characters sharing script+category
- `ScriptEncodingV1` vs `ScriptEncodingV2`: Different categorization strategies
  - V1: More granular script categories
  - V2: High-resource scripts get categories, low-resource collapse to ALL
- Special handling: Hiragana merged with Han, space-combining scripts

### 2. Corpus Management (`script_bpe/corpus/`)

**Purpose**: Efficiently store and access pretokenized training data

**Architecture**:
- `PretokenizedCorpus`: Represents a pretokenized dataset
  - Stored as partitioned Parquet files for parallel access
  - Format: `(chunk_bytes, count)` pairs
  - Metadata includes pretokenizer hash for validation
  
**Storage Structure**:
```
results/corpora/
  └── {corpus_name}/
      └── {pretokenizer_hash}/
          ├── metadata.json
          ├── part_0000.parquet
          ├── part_0001.parquet
          └── ...
```

**Workflow**:
1. Raw text → `from_texts()` → pretokenize with workers
2. Group identical chunks, count frequencies
3. Partition into ~128 files for parallel loading
4. Workers can iterate their partition independently

**Registry** (`registry.py`):
- `load_corpus_by_name()`: Lazy-loads or creates corpora
- Supports HuggingFace datasets (OSCAR, CulturaX, finewiki)
- Monolingual datasets: `eng_latn_300mb`, `kor_hang_300mb`, etc.

### 3. Tokenizers (`script_bpe/tokenizers/`)

Two tokenization algorithms, sharing a common base:

#### Base Layer (`base.py`)
- `BaseToken`: Token with id and atomic_tokens sequence
- `BaseTokenizer`: Abstract interface for encode/decode/save/load
- `BaseTrainer`: Training configuration and logging
- Common stats: compression metrics, Renyi entropy, etc.

#### BPE Implementation (`tokenizers/bpe/`)

**Training** (`trainer.py`):
- **Multi-worker architecture** using fork-server for parallel processing
- Each worker maintains local chunks and pair counts
- Main loop:
  1. Find most frequent token pair
  2. Create new token, broadcast merge to workers
  3. Workers update local counts, send deltas back
  4. Aggregate deltas, update global pair counts
- Uses heaps for efficient top-pair selection
- Enforces merge policies from pretokenizer (char boundaries, etc.)

**Model** (`tokenizer.py`):
- `MergeRule`: (token_a, token_b) → token_c mapping
- `Token`: Tracks original_count and current_count
- Encoding: Greedy heap-based merge application
- Decoding: Concatenate atomic tokens, pass to pretokenizer

**Key Insight**: BPE training is compute-intensive because every merge updates counts across the entire corpus. Multi-worker design parallelizes this.

#### Unigram Implementation (`tokenizers/unigram/`)

**Training** (`trainer.py`):
- **EM algorithm** (Expectation-Maximization):
  - E-step: Calculate expected token counts via forward-backward
  - M-step: Update token probabilities, prune low-count tokens
  - Iterate until vocabulary shrinks to target size
- **Initialization** (`init_algorithms.py`):
  - Extract frequent substrings from corpus
  - Three strategies: "simple", "corpus_long", "corpus_intermediate", "corpus_fallback"
  - Uses suffix arrays and LCP for efficient pattern mining
- **Pruning strategies**:
  - Training: Viterbi-based (considers alternative segmentations)
  - Final: Score-based (faster, just keeps top-N by probability)
  - Defensive pruning: Keeps tokens if their alternatives would also be removed

**Model** (`model.py`):
- `UnigramToken`: Token with log_prob and required flag
- `Trie`: Fast prefix matching for lattice construction
- `Lattice`: Dynamic programming for Viterbi/marginal calculation
  - Forward-backward algorithm for E-step
  - Viterbi for best path
- Encoding: Viterbi decoding on lattice
- Decoding: Concatenate atomic tokens

**Key Insight**: Unigram is probabilistic - same text can tokenize differently. Training uses forward-backward to estimate marginal probabilities, but encoding uses Viterbi for deterministic output.

### 4. Training Entry Point (`script_bpe/train.py`)

**Command-line interface**:
```bash
uv run train --corpus <name> -n <vocab_size> --pretokenizer <name> --model bpe|unigram
```

**Flow**:
1. Get pretokenizer from registry
2. Load/create corpus (cached by pretokenizer hash)
3. Create trainer with config
4. Train and save to `results/{bpe|unigram}_tokenizers/{corpus}/n{size}/{pretokenizer}.json.gz`

**Utilities**:
- `tokenizer_save_path()`: Consistent path naming
- `load_tokenizers_for_dataset()`: Load all pretokenizer variants for comparison

### 5. Utils (`script_bpe/utils.py`)

**Type Definitions**:
- `TokenSeq = array.array`: Efficient integer sequences
- `PretokenizedT = list[TokenSeq]`: List of chunks
- `InputTokenSeq`: Flexible input type

**Multiprocessing**:
- Uses `forkserver` context to avoid copy-on-write issues
- `gc.freeze()` before forking for memory efficiency

**Logging**:
- Custom logger with uptime tracking
- Used consistently across trainers

## Testing Structure

Tests organized by component:
```
tests/
├── pretokenize/     # Pretokenizer correctness (encoding, merge rules, digits)
├── bpe/             # BPE training and merge logic
├── tokenizers/
│   ├── bpe/         # BPE tokenizer functionality
│   ├── unigram/     # Unigram model, lattice, training
│   └── base/        # Base tokenizer interface
├── corpus/          # Corpus creation and loading
├── cli/             # Command-line interface smoke tests
└── conftest.py      # Shared fixtures (taylorswift.txt, pretokenizers)
```

**Testing Patterns**:
- Use `taylorswift.txt` as small test corpus
- Fixtures for common pretokenizers
- Tests for both correctness and edge cases
- Separate tests for unit components (trie, lattice) vs integration

## Important Design Patterns

### 1. Pretokenizer-Corpus Binding
- Corpus is tied to a specific pretokenizer via hash
- Prevents accidentally mixing incompatible base token encodings
- Hash includes all config: normalization, regex, script config, etc.

### 2. Registry Pattern
- Pretokenizers: Name → Config → Class instantiation
- Corpora: Name → Load from cache or create from HuggingFace
- Allows easy experimentation with variants

### 3. Serialization
- Both tokenizers save full pretokenizer config
- Enables standalone loading without external dependencies
- JSON + gzip for compression

### 4. Multi-worker Training
- BPE: Workers process corpus partitions, communicate deltas
- Unigram: Single-process (EM is sequential)
- Corpus designed for efficient worker iteration

### 5. Token Policies
- `token_allowed()`: Checks if merge is valid per pretokenizer rules
- `bpe_merge_allowed()`: Specifically for BPE merge decisions
- Enforces char boundaries, inherited script rules, etc.

## Common Workflows

### Training a New Tokenizer
```python
from script_bpe import get_pretokenizer, BPETrainer, BPETrainerConfig
from script_bpe.corpus import load_corpus_by_name

pretokenizer = get_pretokenizer("scriptenc_cb")
corpus = load_corpus_by_name("eng_latn_300mb", pretokenizer)
config = BPETrainerConfig(additional_vocab_size=64000, num_workers=8)
trainer = BPETrainer(pretokenizer, corpus, config)
tokenizer = trainer.train()
tokenizer.save("path/to/tokenizer.json.gz")
```

### Comparing Pretokenizers
Paper utilities in `paper_utils/` batch-train all variants:
```bash
bash paper_utils/train_monolingual.sh  # Uses GNU parallel
```

### Adding a New Pretokenizer Variant
1. Define config in `pretokenize/__init__.py` (e.g., new regex pattern)
2. Add to `PRETOKENIZER_REGISTRY`
3. Existing `Pretokenizer` classes handle instantiation via registry

### Evaluating Performance
```python
tokenizer = BPETokenizer.load("path/to/tokenizer.json.gz")
corpus = load_corpus_by_name("test_corpus", tokenizer.pretokenizer)
stats = tokenizer.corpus_performance(corpus)
# stats includes: chars_per_token, bytes_per_token, shannon_bits, etc.
```

## Key Files to Know

**Core Logic** (read these to understand algorithms):
- `pretokenize/pretokenizer.py` (340 lines): Pretokenization logic
- `tokenizers/bpe/trainer.py` (250 lines): BPE training loop
- `tokenizers/unigram/trainer.py` (375 lines): Unigram EM algorithm
- `tokenizers/unigram/model.py` (255 lines): Lattice and Viterbi

**Configuration** (read these to understand variants):
- `pretokenize/__init__.py`: All pretokenizer variants
- `pretokenize/scriptencoding.py`: Script encoding schemes
- `corpus/registry.py`: Available corpora

**Entry Points**:
- `train.py`: CLI training script
- `__init__.py`: Public API exports

## Vocabulary

- **Base token**: Atomic unit for BPE/Unigram (byte or script/index pair)
- **Atomic tokens**: Synonym for base tokens (see `Token.atomic_tokens`)
- **Pretokenization**: Chunking and encoding to base tokens
- **Chunk**: A contiguous sequence of base tokens (one pretokenization unit)
- **Merge rule**: BPE operation combining two tokens → one token
- **Script**: Unicode script property (Latin, Arabic, Han, etc.)
- **Category**: Unicode category (Letter, Mark, Punctuation, etc.)
- **Character boundaries**: Merge policy preventing partial UTF-8 characters
- **Inherited script**: Unicode combining marks that inherit script from base char
- **Lattice**: Graph of possible token segmentations (Unigram)
- **Viterbi**: Algorithm to find best path through lattice
- **Forward-backward**: Algorithm to compute marginal probabilities (EM E-step)

## Development Tips

1. **Adding tests**: Use existing fixtures in `conftest.py`, add to relevant subdirectory
2. **Debugging training**: Set `verbose=True` in trainer config for detailed logs
3. **Memory issues**: BPE training is memory-intensive; reduce `num_workers` or corpus size
4. **Pretokenizer experiments**: Clone existing config in registry, modify parameters
5. **Results are cached**: Tokenizers saved to `results/`, delete to retrain
6. **Corpus is cached**: Pretokenized corpora in `results/corpora/`, reused across runs

## Common Gotchas

- **Pretokenizer hash mismatch**: Corpus created with different pretokenizer config
- **Slow training**: `nosplit` variants don't use regex, process entire docs as one chunk
- **Invalid merges**: Check `enforce_char_boundaries` and `enforce_inherited` settings
- **Empty vocabulary**: `additional_vocab_size=0` skips training (useful for corpus prep)
- **Import errors**: Use `from script_bpe import X` not `from script_bpe.tokenizers.bpe import X`
- **Forgetting uv**: Use `uv run` for all Python commands to automatically run in the correct venv.

## Performance Characteristics

- **BPE training**: O(n_merges × corpus_size), parallelizable, memory-intensive
- **Unigram training**: O(iterations × corpus_size), sequential EM, more I/O bound
- **Script encoding**: ~2x more base tokens than UTF-8, but better multilingual compression
- **Corpus loading**: Lazy-loaded from partitions, minimal memory footprint
- **Serialization**: gzip compression ~10x reduction in file size

## Related Files

- `paper_utils/`: Scripts to reproduce paper results
- `nb_*.py`: Jupyter notebooks for analysis (converted to .py format)
- `results/`: Cached tokenizers and corpora (not in git)
- `tests/data/taylorswift.txt`: Small test corpus
