# Experiment Log

## 2026-06-09: f=1.15 Downstream Bundle Integration

Integrated the user-provided `f115_caches.tar.gz` and `f115_results.tar.gz` bundles into the downstream LM results.

Inputs:

- `f115_bundle/f115_results.tsv`: depth-24 nanochat downstream results for `mingram_f1.15`, `bpe_init_f1.15`, and `fsp_bpe_init_f1.15`, seeds 42--45, 8-GPU bpb evaluation.
- `f115_caches/cache_glitch_f115.json`: under-trained-token diagnostics for the same three `f=1.15` arms.
- `f115_caches/cache_renyi_entropy.json`: intrinsic Renyi entropy cache with `MAIN_F=1.15`.

Actions:

- Merged the three `f=1.15` downstream arms into `results/downstream/d24_results.tsv`, preserving the previous MinGram `f=1.25` row as a sensitivity comparison.
- Combined the `f=1.15` glitch diagnostics into `results/downstream/cache_glitch.json`.
- Regenerated `table_downstream.tex`, `app_table_downstream_seeds.tex`, `table_glitch.tex`, and `extra/table_renyi_entropy.tex`.
- Updated `paper.tex` to remove the stale downstream caveat about BPE-initialized hybrids not being run.

Headline downstream means:

- BPE: bpb 0.713894, CORE 0.2597.
- MinGram `f=1.15`: bpb 0.712270, CORE 0.2612, UT multi 0.
- MinGram `f=1.25`: bpb 0.712400, CORE 0.2725, UT multi 0.
- Unigram-BPE-Init `f=1.15`: bpb 0.712516, CORE 0.2655, UT multi 18.25.
- FSP-BPE-Init `f=1.15`: bpb 0.712578, CORE 0.2636, UT multi 90.

Notes:

- MinGram `f=1.15` has the best mean bpb among downstream rows. Its bpb difference from MinGram `f=1.25` is not significant across matched seeds (`p=0.319`).
- MinGram `f=1.25` has the highest mean CORE, but the CORE difference from BPE is not significant (`p=0.191`).
- All non-BPE downstream rows significantly improve bpb over BPE in paired seed tests.
- The downstream experiment still covers only English, one depth, one vocabulary size, and four seeds.

## 2026-06-09: f=1.15 EM Ablation Bundle Integration

Integrated the user-provided `em115_bundle.tar.gz` bundle for the MinGram EM-iteration ablation.

Inputs:

- `em115_bundle/models/{fineweb_en_5gb,fineweb_de_5gb,fineweb_fi_5gb}/mingram_f1.15_em{0,1,2,3,4}_p0.0_n32768_*.model.json.gz`
- `em115_bundle/caches/cache_morphalign_scatter.json`
- `em115_bundle/caches/cache_train_eval_compression_grid.json`
- `em115_bundle/tables/app_table_em_ablation.tex`

Actions:

- Installed the `f=1.15` MinGram EM 0--4 models for English, German, and Finnish into `results/mingram`.
- Replaced the hybrid MorphAlign and train/eval compression caches with the bundle's additive supersets.
- Changed `paper_utils/hybrid/generate_em_ablation_table.py` back to `MAIN_F = 1.15`.
- Regenerated `results/mingram_paper/tables/app_table_em_ablation.tex`; it exactly matches the bundled table.
- Updated the paper appendix text/caption to describe the EM ablation at `f=1.15`.

Headline table:

- MorphAlign mean rises from 0.77 at `N_em=0` to 0.84 at `N_em=1`, then plateaus at 0.85 for `N_em=2--4`.
- Compression mean improves from `-0.86%` at `N_em=0` to `-1.34%` at `N_em=2`, with only small additional gains at `N_em=3` (`-1.35%`) and `N_em=4` (`-1.36%`).
- This supports retaining `N_em=2` as the default at the paper-default `f=1.15`.
