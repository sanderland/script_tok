# Session Status

## Ongoing experiments
- Matched-tokenizer training, arms bnd_w / bnd_wpd / bnd_wpd_caps (jobs 2964899, 2964900, 2964901): one arm per node, each building its own fineweb_en_5gb corpus at 128 workers. Merge with merge_manifests.py when all three finish. First attempt (2957177-9) was cancelled after 6.4h: the parent's Counter merge was quadratic, so those jobs would have needed roughly 68h.
- plain arm: finished (vocab 34,685, 0 roundtrip failures, 3.6360 chars/token). Corpus cached at results/corpora/fineweb_en_5gb/PT-e690609c.
- Depth-12 sweep (not yet submitted): 4 arms x seeds 0,1,2 via cluster/submit_all.sh, blocked on the three tokenizer jobs.

## Open decisions
- Whether to extend past 3 seeds: agreed to run 3 first and revisit, since the README calls 3 seeds a direction rather than a result and the MinGram table it compares against used 20.
- Whether this branch should adopt the EXPERIMENTS_*.md documents from the global working agreements: the branch records experiments as per-grid result JSONs plus marker_experiments/paper.md instead, and I have not introduced the parallel system unasked.

## Notes
- All design choices and every deviation from the README are recorded in marker_experiments/downstream/DESIGN_CHOICES.md. The load-bearing ones: CORE cannot be computed for the punctuation-marking arms (bpb only for those), bnd_w added as a fourth arm, nanochat not pip-installed, environment on capstor scratch.
- Four defects fixed in this session: a None corpus base dir, two worker-shutdown hangs, an interpreter-shutdown hang, and a completion test that assumed every run prints a CORE metric. Details in DESIGN_CHOICES.md.
