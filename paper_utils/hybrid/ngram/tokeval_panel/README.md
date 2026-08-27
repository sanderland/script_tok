# Scoring the TokEval release with n-gram bpb

External validation of the n-gram bits-per-byte metric against TokEval (Meister 2026,
arXiv:2608.18062): 51 std-1B nanochat runs at ~128K vocabulary with published val bpb,
from `cmeister/tokenizer-lm-ablations`. Nothing here trains a model; the downstream
anchors are hers.

Pipeline (all scripts read/write `$TOKEVAL_WORKDIR`, default `.`):

1. `build_mixture.py eval.jsonl train.jsonl` — reconstructs her std-1B training mixture
   from the paper's Tables 7-8 byte shares: 36.9% FineWeb-Edu, 33.4% FineWeb2 across 30
   languages at their exact shares (Russian 18.1% of total bytes down to Tamil 0.1%),
   16.1% FineMath-4plus, 13.6% Python+JS. StarCoderData gates file reads, so code comes
   from codeparrot/github-code filtered by extension — an approximation, stated as such.
2. `get_tokenizers.py` — downloads every std-1B tokeval run's tokenizer (48 HF files, 2
   off-the-shelf references from their own repos; the 3 SCRIPT-encoding ones are native
   script_bpe files fetched separately).
3. `run_panel.py out.tsv` — scores each tokenizer at n=1,2,3 via `evaluate_ngram_bpb`,
   HF ones through `script_bpe.ngram.hf_adapter`. Crash-safe (TSV rewritten per
   tokenizer) and resumable. The nl-split SCRIPT variant is skipped loudly: its config
   carries `split_line_breaks`, which this script_bpe version does not implement, and
   stripping it would score a different pretokenizer than the one that trained the model.
4. `panel_corr.py` / `panel_corr2.py` — Spearman vs her published val bpb, with the
   fragmentation-confound split.

Results (committed as `results/ngram/tokeval_ngram.tsv`), Spearman vs 1.27B val bpb:

    panel                          compression   n=1     n=2       n=3
    all 48 custom tokenizers          +0.47     +0.59   +0.80***  +0.05
    39 mixture-matched only           +0.20ns   +0.41   +0.74***  +0.39*
    her Table 2 best (n=29): Renyi 0.57, trigram entropy 0.44(ns), compression 0.32

Readings: (1) held-out smoothed bigram bpb is the strongest known val-bpb predictor on
her panel, well past Renyi efficiency, and her plug-in trigram entropy is not
significant while our held-out n=2 is at p=7e-12 — held-out estimation and byte
normalization are doing real work, not decoration. (2) On the mixture-matched subset,
compression drops to chance while n-gram holds — the fine-discrimination result from the
d24 study, reproduced externally at 4x the panel size. (3) The order interacts with
fertility spread: deliberately mismatched (english/code-only) tokenizers fragment the
multilingual text into dense low-order statistics and win unfairly at n=3, which is the
"context is measured in tokens" confound in its purest form. On heterogeneous panels use
n=2; on matched panels the d24 study's n>=3 guidance stands. Roughly, the usable order
scales with training tokens per vocabulary entry.

Screening-vs-selection profile matches her predictor fits: bottom-10 overlap with val
bpb is 9/10 (every bad tokenizer flagged), top-10 overlap 3/10.

Caveats: our corpus is a reconstruction of her mixture, not her val shard; code source
substituted; docs capped at 100K chars; the apertus seed replicate (val bpb 0.7197 vs
0.7212) is averaged; references excluded from headline rows.
