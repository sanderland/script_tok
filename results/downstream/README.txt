downstream paper caches — port-over bundle (2026-06-14)
=======================================================
Render the downstream + glitch + renyi tables on any box WITHOUT the 4.3GB
checkpoints. Drop each file at the path below, then run its generator.

cache_glitch.json        -> results/downstream/cache_glitch.json
    Under-trained ("glitch") token analysis, 6 main methods, NOW n=20 seeds (42-61)
    to match the downstream table. multi-char ("wasted vocab") mean:
    MinGram 0, PathPiece 4, Unigram 4.4, ConvexTok 8.4, FSP 41, BPE 0.8.
    Near-deterministic across seeds (tracks corpus rarity, not init), so n=20 == n=4.
    rendered by: paper_utils/hybrid/generate_glitch_table.py   (reads cache, no torch)

d24_master_results.tsv   -> results/downstream/d24_master_results.tsv
    Master per-seed downstream LM results with bpb, CORE, and the 22 centered
    task accuracies. Paper tables use the 20 lowest seed ids for each included
    method so all rows have matched n=20.
    rendered by:
      paper_utils/hybrid/generate_downstream_table.py
      paper_utils/hybrid/generate_downstream_seed_table.py
      paper_utils/hybrid/generate_downstream_task_grid.py

cache_glitch_f115.json   -> results/downstream/cache_glitch_f115.json
    f=1.15 EM-ablation glitch arms (mingram / Unigram-BPE-Init / FSP-BPE-Init), n=4 (42-45).
    Appendix ablation; left at n=4 (separate table). multi-char: MinGram 0,
    Unigram-BPE-Init 18.25, FSP-BPE-Init 90.
    rendered by: paper_utils/hybrid/generate_glitch_table.py (f=1.15 variant)

cache_renyi_entropy.json -> (path per generate_renyi_entropy_table.py)
    Renyi entropy/efficiency cache, MAIN_F=1.15, intrinsic (no seeds).
    rendered by: paper_utils/hybrid/generate_renyi_entropy_table.py

token_usage_counts.parquet -> results/downstream/token_usage_counts.parquet
    Full-corpus tokenizer usage counts for the downstream rare-token column and
    under-trained-token appendix examples.
    rebuild with: paper_utils/hybrid/build_token_usage_counts.py
