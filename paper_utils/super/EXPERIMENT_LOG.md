# Supertoken Experiments Log

## Overview

Training unigram models with "supertokens" - n-gram sequences extracted from a pre-trained tokenizer that span multiple pretokens. This allows transferring tokenization patterns across corpora.

## Experiment Design

### Flow
1. Train initial model A on corpus A (word-level pretokenization, vocab size VA)
2. For corpus B:
   - Load with line pretokenizer (each entry = full line)
   - Re-pretokenize each line with word pretokenizer
   - Tokenize each word chunk with model A
   - Find chunks that tokenize to exactly 1 token (single-token pretokens)
   - Extract n-grams from consecutive spans of single-token pretokens
3. Filter and score supertokens
4. Train final model on LINE-pretokenized corpus B (supertokens can span words)

### Pretokenizers
- **Word pretokenizer** (`scriptenc_cb`): Standard word-level chunking for initial model
- **Line pretokenizer** (`scriptenc_line`): Line-based chunking for final model (allows supertokens to span words)

### Parameters
- `vocab_a`: Vocabulary size for initial model
- `vocab_b`: Vocabulary size for final model
- `max_ngram`: Maximum n-gram size to extract (2, 4, 8)
- `filter_name`: Supertoken filter (all, words, len_8c)
- `fsp`: Flat-score pruning (True/False)

---

## Experiment 1: Baseline Test

**Date**: 2026-01-11

**Config**: smol_eng_latn_300mb, vocab=16k, max_ngram=4, filter=all

### N-gram Extraction
- 6,113,247 word chunks processed (89.4% single-token pretokens)
- **6,938,005 n-grams extracted**

**Top 10 N-grams**:
| Rank | Token | Count |
|------|-------|-------|
| 1 | ` of the` | 27,396 |
| 2 | `, and` | 21,790 |
| 3 | `'s` | 19,746 |
| 4 | ` in the` | 19,538 |
| 5 | `. The` | 13,904 |
| 6 | ` to the` | 11,769 |
| 7 | `, the` | 11,008 |
| 8 | `'s` | 10,528 |
| 9 | ` on the` | 9,268 |
| 10 | `. I` | 8,129 |

### Vocab Composition

| Stage | Supertokens | Regular | Base |
|-------|-------------|---------|------|
| Initial (3x vocab) | 38,169 | 7,915 | 1,916 |
| Final (1x vocab) | 8,451 (22%) | 7,549 | 1,916 |

### Model Metrics
- **Objective**: 0.9112
- **Total tokens**: 7,783,742
- **Tokens/pretoken**: 45.98

### Top 10 Surviving Supertokens (by log-prob)
| Rank | Token | #PT | Init LP | Final LP | Δ LP |
|------|-------|-----|---------|----------|------|
| 1 | `, and` | 2 | -5.69 | -6.38 | -0.69 |
| 2 | `s,` | 2 | -9.81 | -6.43 | +3.38 |
| 3 | ` of the` | 2 | -5.46 | -6.46 | -1.00 |
| 4 | `. The` | 2 | -6.14 | -6.48 | -0.34 |
| 5 | `'s` | 2 | -5.79 | -6.63 | -0.84 |
| 6 | ` in the` | 2 | -5.80 | -6.63 | -0.83 |
| 7 | `s and` | 2 | -10.11 | -6.67 | +3.45 |
| 8 | `s.` | 2 | -9.89 | -7.03 | +2.86 |
| 9 | `'s` | 2 | -6.42 | -7.08 | -0.66 |
| 10 | `, the` | 2 | -6.37 | -7.12 | -0.75 |

### Top 10 Longest Supertokens (by #pretokens)
| Rank | Token | #PT | Final LP |
|------|-------|-----|----------|
| 1 | `. It's` | 4 | -8.93 |
| 2 | ` U.S.` | 4 | -9.13 |
| 3 | `, as well as` | 4 | -9.41 |
| 4 | `". It's"` | 4 | -9.64 |
| 5 | `, it's` | 4 | -9.68 |
| 6 | ` one of the most` | 4 | -10.06 |
| 7 | `-year-old` | 4 | -10.15 |
| 8 | ` when it comes to` | 4 | -10.19 |
| 9 | `. I'm` | 4 | -10.27 |
| 10 | `. For example,` | 4 | -10.29 |

### Key Observations

1. **Supertokens survive at 22% rate**: Not all pruned, meaningful patterns persist.

2. **Log-prob shifts reveal patterns**: 
   - `s,` gains +3.38 (punctuation after plurals)
   - `s and` gains +3.45 (plural + conjunction)
   - EM "discovers" these patterns despite low initial estimates

3. **Most top supertokens are 2-pretoken bigrams**: Function word combinations like ` of the`, ` in the`, `, and`.

4. **Longest supertokens are 4-pretoken phrases**: Common idioms and constructions like ` one of the most`, ` when it comes to`, `, as well as`.

---

## Supertoken Filters

### Individual Patterns

| Filter | Pattern | Examples |
|--------|---------|----------|
| `space_phrase` | Multi-word with space start | ` of the`, ` in the`, ` one of the most` |
| `punct_trans` | Punctuation + word | `, and`, `. The`, `; but` |
| `contraction` | Apostrophe words | `'s`, `it's`, `don't`, `I'm` |
| `suffix` | Suffix + punct/word | `s,`, `s.`, `s and`, `s are` |
| `abbrev` | Abbreviations | `U.S.`, `e.g.`, `i.e.` |
| `hyphen` | Hyphenated compounds | `-year-old`, `well-known` |
| `domain` | URLs/domains | `.com`, `https://`, `www.` |

### Combinations

| Filter | Includes | Goal |
|--------|----------|------|
| `semantic` | space_phrase + contraction + abbrev + hyphen | Clean semantic units |
| `all_patterns` | All recognized patterns | Maximum coverage |
| `all` | Everything | Baseline |
| `len_8c` | Max 8 chars | Short tokens only |

---

## Hypotheses

| # | Hypothesis | Status |
|---|------------|--------|
| 1 | VA = 2×VB extracts more useful supertokens | ❓ Needs test |
| 2 | Higher max_ngram captures longer phrases | ❓ Needs test |
| 3 | `semantic` filter produces nicer tokens vs `all` | ❓ Needs test |
| 4 | Individual filters (contraction, abbrev) help specific use cases | ❓ Needs test |
| 5 | FSP improves compression, hurts objective | ✓ Confirmed (unigram paper) |

---

## Planned Experiments (run_experiments.sh)

1. **Main filter comparison** (VA=2×VB, N=4,8): `all`, `semantic`, `all_patterns`
2. **Individual pattern ablation** (VA=2×VB, N=4): Each pattern filter in isolation
3. **VA comparison**: VA=VB vs VA=2×VB for main filters
4. **FSP comparison**: With/without FSP for `all` and `semantic`

**Outputs**:
- `supertoken_scatter.png` - Supertoken count vs compression
- `filter_comparison.png` - Bar chart comparing filters
- `comparison_table.md` - Full results table
- `top_supertokens.md` - Most common supertokens across models

---

## Files

- `paper_utils/super/train_supertokens.py` - Training script (`--test` for quick debug)
- `paper_utils/super/utils.py` - Filters and utilities
- `paper_utils/super/generate_results.py` - Results visualization
- `paper_utils/super/run_experiments.sh` - Experiment grid runner
