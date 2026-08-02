# Session Status

## Deliverable
- PR sanderland/script_tok#8, branch claude/downstream-lm-eval on the cimeister fork, into claude/fineweb-space-neighbors-k10ufw. Tables regenerate from artifacts with make_tex_tables.py; no number is hand-entered.

## Ongoing experiments
- Byte-factor recomputation (job 2976714): the four factors under the corrected byte convention, cached in the shared dir for the sweep to read.
- Depth-12 sweep, round 3 (not yet submitted): 4 arms x seeds 0,1,2, held until four parallel audits report, so a fourth round is not discarded for a fixable reason.
- Matched tokenizers: all four trained and merged into manifest.json. plain 3.6360 chars/token, bnd_w 3.0960, bnd_wpd 3.7403, bnd_wpd_caps 3.7456, zero roundtrip failures.

## Discarded rounds, and why
- Round 1 (jobs 2972879-90): all 12 runs shared one NANOCHAT_BASE and therefore one token_bytes.pt, so runs were scored against another arm's byte table. Logs in results/marker_downstream/logs_invalid/.
- Round 2 (jobs 2973952-63): completed cleanly, but the 1-byte floor added to make marker loss count also un-masked BOS, which nanochat's bpb excludes by design. Logs in results/marker_downstream/logs_invalid_bos/.

## Open decisions
- Whether to extend past 3 seeds, and whether to vary data order rather than initialization alone. The three seeds currently differ only in weight init, so the error bars understate run-to-run variance. Deferred until round 3 lands and the gaps can be compared against the spread.

## Notes
- The EXPERIMENTS_*.md document set from the global working agreements is deliberately NOT adopted on this branch, per the project owner. Experiments are recorded as per-grid result JSONs plus the paper, with divergences in DESIGN_CHOICES.md.
- marker_experiments/downstream/DESIGN_CHOICES.md records every deviation from marker_experiments/downstream/README.md and is kept current.
