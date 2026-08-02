# Session Status

## Deliverable
- PR sanderland/script_tok#8, branch claude/downstream-lm-eval on the cimeister fork, into claude/fineweb-space-neighbors-k10ufw. Tables regenerate from artifacts with make_tex_tables.py. The bpb, CORE, chars/token and text-coverage figures are read from artifacts; the eval-slice size and the corpus name in the captions are still literals in the generator.

## Ongoing experiments
- Depth-12 sweep, round 3 (jobs 2976849-60): 4 arms x seeds 0,1,2, launched after four parallel audits. Collect with collect_results.py, then regenerate tables.
- Matched tokenizers: all four trained and merged into manifest.json. plain 3.6360 chars/token, bnd_w 3.0960, bnd_wpd 3.7403, bnd_wpd_caps 3.7456, zero roundtrip failures.

## Discarded rounds, and why (full detail in DESIGN_CHOICES.md)
- Round 1 (jobs 2972879-90): all 12 runs shared one NANOCHAT_BASE and therefore one token_bytes.pt, so runs were scored against another arm's byte table. Logs in results/marker_downstream/logs_invalid/.
- Round 2 (jobs 2973952-63): completed cleanly, but the 1-byte floor added to make marker loss count also un-masked BOS, which nanochat's bpb excludes by design. Logs in results/marker_downstream/logs_invalid_bos/.

## Open decisions
- Whether to extend past 3 seeds. Round 3 varies training data order per seed as well as weight initialization, using one permutation per seed shared across arms so the arms stay paired, so the spread now covers both. Deferred until round 3 finishes, so the arm-to-arm differences can be compared against the seed-to-seed standard deviation.
- Whether to move from 8 shards to about 32. At 8 shards the run makes 3.4 to 3.6 passes over the same 2.0 GB, which suppresses the mechanism by which better compression is meant to help. Single-epoch would test it directly and costs only a download.

## Notes
- The EXPERIMENTS_*.md document set from the global working agreements is deliberately NOT adopted on this branch, per the project owner. Experiments are recorded as per-grid result JSONs plus the paper, with divergences in DESIGN_CHOICES.md.
- marker_experiments/downstream/DESIGN_CHOICES.md records every deviation from marker_experiments/downstream/README.md and is kept current.
