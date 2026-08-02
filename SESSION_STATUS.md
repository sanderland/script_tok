# Session Status

## Deliverable
- PR sanderland/script_tok#8, branch claude/downstream-lm-eval on the cimeister fork, into claude/fineweb-space-neighbors-k10ufw. Tables regenerate from artifacts with make_tex_tables.py. The bpb, CORE, chars/token and text-coverage figures are read from artifacts; the eval-slice size and the corpus name in the captions are still literals in the generator.

## Ongoing experiments
- Extra seeds for bnd_wpd (jobs 2979301-03, seeds 3,4,5, 8 shards): tests whether the within-arm spread estimated from 3 seeds holds at 6. These cannot sharpen the paired comparison against plain, which needs the same seeds in both arms.
- 32-shard regime (jobs 2979305-10): plain and bnd_wpd at seeds 0,1,2, single-epoch instead of 3.4-3.6 passes. plain is included because bnd_wpd alone would have nothing to compare against once the data regime changes. Separate NANOCHAT_BASE (nanochat_base_n32) and OUT (results/marker_downstream_n32), since adding shards to the shared directory would change what an 8-shard run trains on.
- Matched tokenizers: all four trained and merged into manifest.json. plain 3.6360 chars/token, bnd_w 3.0960, bnd_wpd 3.7403, bnd_wpd_caps 3.7456, zero roundtrip failures.

## Completed
- Round 3 (jobs 2976849-60): 12/12 clean, results in marker_experiments/downstream/results_round3.tsv and in the paper tables. plain 0.8853, bnd_w 0.8768, bnd_wpd 0.8800, bnd_wpd_caps 0.8795 bpb per true byte.

## Discarded rounds, and why (full detail in DESIGN_CHOICES.md)
- Round 1 (jobs 2972879-90): all 12 runs shared one NANOCHAT_BASE and therefore one token_bytes.pt, so runs were scored against another arm's byte table. Logs in results/marker_downstream/logs_invalid/.
- Round 2 (jobs 2973952-63): completed cleanly, but the 1-byte floor added to make marker loss count also un-masked BOS, which nanochat's bpb excludes by design. Logs in results/marker_downstream/logs_invalid_bos/.

## Open decisions
- Whether the 3-seed spread is trustworthy, and whether the result survives a single-epoch regime. Both are being tested by the runs above.

## Notes
- The EXPERIMENTS_*.md document set from the global working agreements is deliberately NOT adopted on this branch, per the project owner. Experiments are recorded as per-grid result JSONs plus the paper, with divergences in DESIGN_CHOICES.md.
- marker_experiments/downstream/DESIGN_CHOICES.md records every deviation from marker_experiments/downstream/README.md and is kept current.
