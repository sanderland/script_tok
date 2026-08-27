"""Test the fragmentation-confound reading: does n=3 recover once the deliberately
mismatched-vocabulary tokenizers (english-only, code-only, highres/highmid) are removed?"""
import csv
import os
import collections
from scipy.stats import spearmanr

SP = os.environ.get("TOKEVAL_WORKDIR", ".")
grid = {r['run']: r for r in csv.DictReader(open(f"{SP}/tokeval/ablation_grid.csv"))}
ng = collections.defaultdict(dict)
comp = {}
for r in csv.DictReader(open(f"{SP}/tokeval_ngram.tsv"), delimiter='\t'):
    ng[r['run']][int(r['order'])] = float(r['bpb'])
    comp[r['run']] = float(r['tokens_per_byte'])

REFS = {'full-128k-apertus','full-128k-apertus-seed42','full-128k-llama3'}
MISMATCHED = {'english','code','highres','highmid'}
pts = {}
for run, o in ng.items():
    g = grid[run]
    if run in REFS:
        continue
    pts[run] = {"val": float(g['val_bpb']), "data": g['tok_training_data'] or '?',
                "comp": comp[run], **{f"n{k}": v for k, v in o.items()}}

def corr(sub, label):
    runs = sorted(sub)
    val = [sub[r]['val'] for r in runs]
    out = [label, f"n={len(runs)}"]
    for key in ("comp","n1","n2","n3"):
        rho = spearmanr([sub[r][key] for r in runs], val)
        out.append(f"{key}={rho.statistic:+.2f}({rho.pvalue:.0e})")
    print("  ".join(out))

corr(pts, "all custom          ")
matched = {r: p for r, p in pts.items() if p['data'] not in MISMATCHED and 'english' not in r and 'code' not in r and 'highres' not in r and 'highmid' not in r}
corr(matched, "mixture-matched only")
mism = {r: p for r, p in pts.items() if r not in matched}
corr(mism, "mismatched only     ")
# fertility spread in each group
for label, sub in (("matched", matched), ("mismatched", mism)):
    cs = [sub[r]['comp'] for r in sub]
    print(f"{label}: tokens/byte range {min(cs):.3f}-{max(cs):.3f}")
