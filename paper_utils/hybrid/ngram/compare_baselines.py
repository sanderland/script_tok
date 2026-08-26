#!/usr/bin/env python3
"""The decisive table: n-gram bpb against the intrinsic baselines, same test, same pairs.

If plain compression predicted downstream val_bpb as well as the n-gram model does, this
whole module would be a complicated way to compute tokens/byte. It does not: on the
downstream-resolved pairs, compression sits at chance (53-57%) -- consistent with the
literature's repeated finding that compression alone is unreliable on closely matched
vocabularies -- while n>=3 bpb is significantly above it on both text slices (67-70%,
p<=0.049). n=1 bpb (token entropy per byte, the intrinsic-style endpoint of the family)
matches n=3 on one slice and falls to insignificance on the other, which is the
correlation-side echo of the rank-stability finding: sequential structure buys
reproducibility more than raw accuracy.

    uv run python paper_utils/hybrid/ngram/compare_baselines.py \
        results/ngram/ngram_bpb_fineweb_en_5gb_v10a.tsv
"""

import csv
import itertools
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from cyclopts import App
from scipy.stats import binomtest, ttest_ind
from tabulate import tabulate

from paper_utils.hybrid.downstream_results import DOWNSTREAM_RESULTS_TSV
from paper_utils.hybrid.ngram.correlate_ngram_downstream import DOWNSTREAM_METHOD, SUPERSEDED_METHOD

app = App()


@app.default
def cli(ngram_tsv: str, downstream_tsv: str = str(DOWNSTREAM_RESULTS_TSV), col: str = "val_bpb",
        alpha: float = 0.05, seeds_per_method: int = 20, include_superseded: bool = False) -> None:
    """Sign-agreement of each candidate metric with downstream, on the resolved pairs.

    Args:
        ngram_tsv: Output of run_ngram_eval.py (all orders it contains are compared).
        downstream_tsv: Master per-seed downstream results.
        col: Downstream column to compare against.
        alpha: Welch significance level defining the resolved pairs.
        seeds_per_method: Matched seed count per arm, lowest ids first.
        include_superseded: Same gating as the other tools in this directory.
    """
    per: dict[str, list] = defaultdict(list)
    with open(downstream_tsv) as f:
        for row in csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"):
            per[row["method"]].append((int(row["seed"]), float(row[col])))
    seeds = {m: [v for _, v in sorted(rows)[:seeds_per_method]] for m, rows in per.items()}
    lower_better = col != "core"

    tsv = list(csv.DictReader(open(ngram_tsv), delimiter="\t"))
    orders = sorted({int(r["order"]) for r in tsv})
    metrics: dict[str, dict[str, float]] = {
        "compression (tokens/byte)": {r["tokenizer_id"]: float(r["tokens_per_byte"])
                                      for r in tsv if int(r["order"]) == orders[0]},
    }
    for o in orders:
        metrics[f"n={o} bpb"] = {r["tokenizer_id"]: float(r["bpb"]) for r in tsv if int(r["order"]) == o}

    arms = sorted(a for a in metrics[f"n={orders[0]} bpb"]
                  if DOWNSTREAM_METHOD.get(a) in seeds
                  and (include_superseded or DOWNSTREAM_METHOD[a] not in SUPERSEDED_METHOD.values()))
    resolved = []
    for a, b in itertools.combinations(arms, 2):
        ma, mb = DOWNSTREAM_METHOD[a], DOWNSTREAM_METHOD[b]
        if ttest_ind(seeds[ma], seeds[mb], equal_var=False).pvalue < alpha:
            mean_a, mean_b = statistics.fmean(seeds[ma]), statistics.fmean(seeds[mb])
            resolved.append((a, b, (mean_a < mean_b) if lower_better else (mean_a > mean_b)))
    if not resolved:
        raise SystemExit("no downstream-resolved pairs at this alpha")

    rows_out = []
    for name, vals in metrics.items():
        hits = sum((vals[a] < vals[b]) == a_better for a, b, a_better in resolved)
        rows_out.append({"metric": name, "correct": f"{hits}/{len(resolved)}",
                         "pct": 100 * hits / len(resolved),
                         "binomial_p": binomtest(hits, len(resolved), 0.5, alternative="greater").pvalue})
    print(f"\n{col}-resolved pairs: {len(resolved)} ({len(arms)} arms, {Path(ngram_tsv).name})\n")
    print(tabulate(rows_out, headers="keys", tablefmt="github", floatfmt=".3f"))


if __name__ == "__main__":
    sys.exit(app())
