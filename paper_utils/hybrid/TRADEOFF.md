# Optimizing the Compression-Quality Tradeoff

Investigation of techniques that combine strengths of different tokenization approaches or variations on unigram to achieve better compression than standard methods alone.

## Methods

### BPE-Init

Train BPE first, then use its vocabulary to initialize Unigram training.

**Hypothesis:** BPE's greedy merges find compression-efficient tokens; Unigram's probabilistic framework can refine these.

**Implementation:** `train_hybrid.py` with `bpe_init_factor` parameter

**Factors tested:** 1.0, 1.1, 1.25, 1.5, 2.0 (ratio of BPE vocab size to final vocab size)

### Token Bias (Training Time)

Reduce all token log-probs during training to encourage shorter segmentations.

**Hypothesis:** Penalizing token count pushes EM toward compression-efficient vocabularies.

**Implementation:** `token_bias` parameter in UnigramTrainerConfig

**Values tested:** -1, 0, 1, 2, 5

## TODO: Inference-Time Token Bias

Apply token bias at inference time only (no retraining needed).

**Approach options:**
1. Adjust log_probs when creating tokenizer copy for inference
2. Add `bias` parameter to `tokenize()` / `encode()` methods passed through to Viterbi

**Experiments to run:**
- Apply inference bias to: Default Unigram, FSP, BPE-Init
- Evaluate on training corpus and held-out FineWiki
- Compare training-time vs inference-time bias effectiveness

## Generated Files

- `results/hybrid/` - Model outputs and cache files
- `table_bpe_init_comparison.tex` - BPE-init vs baselines
- `table_bias_sweep.tex` - Effect of different bias values
