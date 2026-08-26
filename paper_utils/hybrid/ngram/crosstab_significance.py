#!/usr/bin/env python3
"""Where both metrics claim confidence, do they agree? val_bpb vs n-gram bpb, per pair.

The one-sided analysis in correlate_ngram_downstream.py conditions only on the
*downstream* sweep resolving a pair, then scores the n-gram metric's point estimate on it
-- including pairs the n-gram bootstrap itself declines to call. That undercharges neither
side but blurs the question. This restricts to pairs where BOTH tests are individually
significant (Welch over downstream seeds; paired bootstrap over held-out documents for the
n-gram side) and cross-tabulates:

    both resolved -> agreement rate, against the 50% coin-flip baseline
    ngram-only    -> claims the cheap metric makes that 20 GPU seeds cannot check
    downstream-only -> pairs 5 MB of eval text cannot yet order (more text shrinks this)
    neither       -> pairs nobody can order

On FineWeb English at n=3 this gives 17/20 = 85% (p=0.0013) and 18/25 = 72% (p=0.022) on
two disjoint text slices. Every disagreement involves unigram or fsp being rated better by
the n-gram model than the transformer agrees with -- the systematic miss, in sharper form.

    uv run python paper_utils/hybrid/ngram/crosstab_significance.py \
        results/ngram/ngram_bpb_fineweb_en_5gb_v10a_docbits.npz --order 3
"""

import csv
import itertools
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from cyclopts import App
from scipy.stats import binomtest, pearsonr, spearmanr, ttest_ind
from tabulate import tabulate

from paper_utils.hybrid.downstream_results import DOWNSTREAM_RESULTS_TSV
from paper_utils.hybrid.ngram.arm_significance import paired_bootstrap
from paper_utils.hybrid.ngram.correlate_ngram_downstream import DOWNSTREAM_METHOD, SUPERSEDED_METHOD

app = App()


def _seed_values(path: Path, col: str, seeds_per_method: int) -> dict[str, list[float]]:
    per: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"):
            per[row["method"]].append((int(row["seed"]), float(row[col])))
    return {m: [v for _, v in sorted(rows)[:seeds_per_method]] for m, rows in per.items()}


@app.default
def cli(docbits_npz: str, order: int = 3, downstream_tsv: str = str(DOWNSTREAM_RESULTS_TSV),
        col: str = "val_bpb", alpha: float = 0.05, resamples: int = 10_000,
        seeds_per_method: int = 20, include_superseded: bool = False) -> None:
    """Cross-tabulate per-pair significance of n-gram bpb against a downstream column.

    Args:
        docbits_npz: Per-document bits written by run_ngram_eval.py.
        order: N-gram order to test.
        downstream_tsv: Master per-seed downstream results.
        col: Downstream column to compare against (val_bpb; core also works but is
            noise-bound -- see correlate_ngram_downstream.py).
        alpha: Welch significance level for the downstream side.
        resamples: Bootstrap resamples for the n-gram side's 95% CIs.
        seeds_per_method: Matched seed count per arm, lowest seed ids first (the paper's
            convention).
        include_superseded: Also pair arms against downstream rows the paper no longer
            reports (the bare `pathpiece` block, whose pb setting is unrecorded). Same
            gating and same reasoning as correlate_ngram_downstream.py: sensitivity
            check, not headline.
    """
    values = _seed_values(Path(downstream_tsv), col, seeds_per_method)
    lower_better = col != "core"

    data = np.load(docbits_npz)
    doc_bytes = data["doc_bytes"].astype(np.float64)
    arms = sorted(a for a in (k.split("|")[0] for k in data.files if k.endswith(f"|{order}"))
                  if DOWNSTREAM_METHOD.get(a) in values
                  and (include_superseded or DOWNSTREAM_METHOD[a] not in SUPERSEDED_METHOD.values()))
    if len(arms) < 3:
        raise SystemExit(f"only {len(arms)} arms have both docbits at order {order} and "
                         f"downstream rows; check names against {downstream_tsv}")
    bits = {a: data[f"{a}|{order}"].astype(np.float64) for a in arms}

    buckets: dict[str, list[str]] = {"both": [], "ngram_only": [], "downstream_only": [], "neither": []}
    agree = 0
    disagreements = []
    for a, b in itertools.combinations(arms, 2):
        ma, mb = DOWNSTREAM_METHOD[a], DOWNSTREAM_METHOD[b]
        ds_sig = ttest_ind(values[ma], values[mb], equal_var=False).pvalue < alpha
        obs, lo, hi = paired_bootstrap(bits[a], bits[b], doc_bytes, resamples, 0)
        ng_sig = not (lo <= 0 <= hi)
        key = ("both" if ng_sig and ds_sig else "ngram_only" if ng_sig
               else "downstream_only" if ds_sig else "neither")
        buckets[key].append(f"{a} vs {b}")
        if key == "both":
            mean_a, mean_b = statistics.fmean(values[ma]), statistics.fmean(values[mb])
            ds_a_better = (mean_a < mean_b) if lower_better else (mean_a > mean_b)
            if ds_a_better == (obs < 0):
                agree += 1
            else:
                disagreements.append(f"{a} vs {b}")

    both = len(buckets["both"])
    total = sum(len(v) for v in buckets.values())
    print(f"\n{col} vs n-gram bpb, n={order}, {len(arms)} arms, {total} pairs "
          f"({Path(docbits_npz).name})\n")
    rows = [{"bucket": k, "pairs": len(v)} for k, v in buckets.items()]
    print(tabulate(rows, headers="keys", tablefmt="github"))
    if both:
        p = binomtest(agree, both, 0.5, alternative="greater").pvalue
        print(f"\nwhere both are confident: {agree}/{both} agree = {agree / both * 100:.0f}%, "
              f"binomial p={p:.4f} vs coin flip")
    if disagreements:
        print(f"disagreements: {'; '.join(disagreements)}")
    print("\nngram-only pairs are claims the downstream seed count cannot check; "
          "downstream-only pairs would tighten with more n-gram eval text (CPU-cheap).")

    x = [bits[a].sum() / doc_bytes.sum() for a in arms]
    y = [statistics.fmean(values[DOWNSTREAM_METHOD[a]]) for a in arms]
    pr, sr = pearsonr(x, y), spearmanr(x, y)
    print(f"\narm-level magnitude ({len(arms)} points): pearson r={pr.statistic:.3f} "
          f"(p={pr.pvalue:.3f}), spearman rho={sr.statistic:.3f} (p={sr.pvalue:.3f})")


if __name__ == "__main__":
    sys.exit(app())
