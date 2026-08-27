import csv
import os
import collections
import statistics
from scipy.stats import spearmanr

SP = os.environ.get("TOKEVAL_WORKDIR", ".")
grid = {r['run']: r for r in csv.DictReader(open(f"{SP}/tokeval/ablation_grid.csv"))}
ng = collections.defaultdict(dict)
comp = {}
for r in csv.DictReader(open(f"{SP}/tokeval_ngram.tsv"), delimiter='\t'):
    ng[r['run']][int(r['order'])] = float(r['bpb'])
    comp[r['run']] = float(r['tokens_per_byte'])

REFS = {'full-128k-apertus', 'full-128k-apertus-seed42', 'full-128k-llama3'}
# dedupe apertus seed replicate: one point, mean val_bpb
points = {}
for run, orders in ng.items():
    g = grid[run]
    points[run] = {"val": float(g['val_bpb']), "flores": float(g['flores_mean_bpb'] or 0),
                   "ref": run in REFS, **{f"n{o}": v for o, v in orders.items()}, "comp": comp[run]}
ap = [p for r, p in points.items() if r.startswith('full-128k-apertus')]
if len(ap) == 2:
    merged = dict(ap[0])
    merged['val'] = statistics.fmean(x['val'] for x in ap)
    del points['full-128k-apertus-seed42']
    points['full-128k-apertus'] = merged
    print(f"apertus seed replicate: val_bpb delta = {abs(ap[0]['val']-ap[1]['val']):.4f} (seed noise at 1.27B)")

def table(pts, label):
    runs = sorted(pts)
    val = [pts[r]['val'] for r in runs]
    print(f"\n=== {label} (n={len(runs)}) — Spearman vs val_bpb ===")
    for name, key in [("compression (tokens/byte)", "comp"), ("n=1 bpb", "n1"), ("n=2 bpb", "n2"), ("n=3 bpb", "n3")]:
        x = [pts[r][key] for r in runs]
        rho = spearmanr(x, val)
        # compression: more tokens/byte = worse expected -> positive rho = agreement; bpb: positive = agreement
        print(f"  {name:26s} rho={rho.statistic:+.3f}  p={rho.pvalue:.2e}")
    print("  her Table 2 (n=29 primary): Renyi -0.57*, trigram entropy -0.44 (ns), bigram -0.35, compression -0.32")

custom = {r: p for r, p in points.items() if not p['ref']}
table(custom, "custom tokenizers, references excluded")
table(points, "sensitivity: references included")

# top/bottom by n=3 vs val ranking
runs = sorted(custom, key=lambda r: custom[r]['n3'])
vals = sorted(custom, key=lambda r: custom[r]['val'])
print("\nngram-n3 top 5:", [r.replace('full-128k-','') for r in runs[:5]])
print("val_bpb  top 5:", [r.replace('full-128k-','') for r in vals[:5]])
print("ngram-n3 bottom 5:", [r.replace('full-128k-','') for r in runs[-5:]])
print("val_bpb  bottom 5:", [r.replace('full-128k-','') for r in vals[-5:]])
