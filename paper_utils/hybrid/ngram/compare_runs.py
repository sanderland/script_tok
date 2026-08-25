#!/usr/bin/env python3
"""Is a ranking from `run_ngram_eval.py` reproducible, or is it sampling noise?

At higher orders the arms sit very close together -- on FineWeb English the seven
available arms span 2.0% of a bit at n=1 but only 0.13% at n>=3. A spread that small is
worthless on its own: the question is whether it is a stable property of the tokenizers or
an artifact of which text happened to be sampled. This answers that by comparing two runs
over disjoint text (`run_ngram_eval.py --skip-chars ...`) and reporting the rank
correlation between them at each order.

Read tau as the ceiling on what the metric could possibly predict downstream. Where tau is
1.0 the ranking is a real property of the tokenizers at that order; where it is below 1.0
the metric is not resolving the arms and no downstream correlation should be expected.

    uv run python paper_utils/hybrid/ngram/compare_runs.py \
        results/ngram/ngram_bpb_fineweb_en_5gb.tsv \
        results/ngram/ngram_bpb_fineweb_en_5gb_rep2.tsv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

from cyclopts import App
from scipy.stats import kendalltau
from tabulate import tabulate

app = App()


def _by_order(path: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[int(row["order"])][row["tokenizer_id"]] = float(row["bpb"])
    return out


@app.default
def cli(run_a: str, run_b: str, detail: bool = False) -> None:
    """Compare two n-gram eval runs and report rank stability per order.

    Args:
        run_a: First run's TSV.
        run_b: Second run's TSV, ideally over disjoint text.
        detail: Also print each arm's bpb in both runs and the shift between them.
    """
    a, b = _by_order(Path(run_a)), _by_order(Path(run_b))
    orders = sorted(set(a) & set(b))
    if not orders:
        raise SystemExit("the two runs share no orders")

    table = []
    for order in orders:
        arms = sorted(set(a[order]) & set(b[order]))
        if len(arms) < 3:
            continue
        va, vb = [a[order][m] for m in arms], [b[order][m] for m in arms]
        table.append({
            "order": order,
            "arms": len(arms),
            "spread_a": max(va) - min(va),
            "spread_b": max(vb) - min(vb),
            # A near-uniform shift between samples is expected -- the two slices differ in
            # difficulty. It is harmless as long as it does not reorder the arms, which is
            # exactly what tau measures.
            "mean_shift": sum(y - x for x, y in zip(va, vb)) / len(arms),
            "tau": kendalltau(va, vb).statistic,
            "best": min(arms, key=lambda m: a[order][m]),
        })
    print(f"\nrank stability across two samples\n  A: {run_a}\n  B: {run_b}\n")
    print(tabulate(table, headers="keys", tablefmt="github", floatfmt=".4f"))
    print("\ntau = 1.0 means the ranking survived a different sample; below that, the metric\n"
          "is not resolving the arms at that order and no downstream correlation is expected.")

    if detail:
        for order in orders:
            arms = sorted(set(a[order]) & set(b[order]), key=lambda m: a[order][m])
            rows = [{"arm": m, "bpb_a": a[order][m], "bpb_b": b[order][m],
                     "delta": b[order][m] - a[order][m]} for m in arms]
            print(f"\nn={order}")
            print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".4f"))


if __name__ == "__main__":
    sys.exit(app())
