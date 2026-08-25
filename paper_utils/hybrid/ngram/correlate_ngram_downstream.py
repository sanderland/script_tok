#!/usr/bin/env python3
"""Does n-gram bits-per-byte predict what the GPU runs found? Per order.

This is the experiment that decides whether the n-gram metric is worth anything. The
downstream sweep already scored the same nine arms with 20 seeds each of nanochat
pretraining; this ranks the CPU-only n-gram numbers against those and reports, for every
order, how well the cheap ranking reproduces the expensive one.

Read the output as follows. Kendall tau against `core` is the headline -- CORE is the
downstream benchmark the paper reports, and rank agreement is what matters for a metric
whose job is to choose between tokenizers. Tau against `val_bpb` is the easier target,
since it compares a bits-per-byte number to another bits-per-byte number; a metric that
tracks val_bpb but not CORE is measuring compression, which is what the existing intrinsic
metrics already do and is exactly the correlation known to be unreliable.

`--seed-spread` prints the seed-to-seed standard deviation of the downstream numbers, which
bounds how well *anything* could correlate: where two arms differ by less than the seed
noise, their true order is not established by the downstream runs either, and a
disagreement there is not evidence against the n-gram metric.

    uv run python paper_utils/hybrid/ngram/correlate_ngram_downstream.py \
        --ngram-tsv results/ngram/ngram_bpb_fineweb_en_5gb.tsv
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from cyclopts import App
from scipy.stats import kendalltau, spearmanr
from tabulate import tabulate

from paper_utils.hybrid.downstream_results import DOWNSTREAM_RESULTS_TSV, METHOD_ORDER, load_rows
from paper_utils.hybrid.token_usage_counts import TOKENIZER_TO_METHOD

app = App()


def _downstream_by_method(path: Path) -> dict[str, dict[str, float]]:
    """Mean CORE and val_bpb per arm, plus the seed spread of each."""
    per_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in load_rows(path):
        for col in ("core", "val_bpb"):
            if row.get(col) not in (None, ""):
                per_method[row["method"]][col].append(float(row[col]))
    out = {}
    for method, cols in per_method.items():
        entry = {}
        for col, values in cols.items():
            entry[col] = statistics.fmean(values)
            entry[f"{col}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            entry[f"{col}_n"] = len(values)
        out[method] = entry
    return out


@app.default
def cli(
    ngram_tsv: str,
    downstream_tsv: str = str(DOWNSTREAM_RESULTS_TSV),
    seed_spread: bool = False,
    out: str | None = None,
) -> None:
    """Rank-correlate n-gram bpb against downstream CORE and val_bpb, one row per order.

    Args:
        ngram_tsv: Output of run_ngram_eval.py.
        downstream_tsv: The GPU sweep's master results (method, seed, val_bpb, core).
        seed_spread: Also print each arm's downstream mean and seed-to-seed spread.
        out: Write the correlation table to this TSV as well as printing it.
    """
    downstream = _downstream_by_method(Path(downstream_tsv))
    if not downstream:
        raise SystemExit(f"no downstream rows in {downstream_tsv}")

    with open(ngram_tsv) as f:
        ngram_rows = list(csv.DictReader(f, delimiter="\t"))

    by_order: dict[int, dict[str, float]] = defaultdict(dict)
    unmapped = set()
    for row in ngram_rows:
        method = TOKENIZER_TO_METHOD.get(row["tokenizer_id"])
        if method is None:
            unmapped.add(row["tokenizer_id"])
            continue
        by_order[int(row["order"])][method] = float(row["bpb"])
    if unmapped:
        print(f"# ignoring arms with no downstream counterpart: {', '.join(sorted(unmapped))}")

    table = []
    for order in sorted(by_order):
        scores = by_order[order]
        methods = [m for m in METHOD_ORDER if m in scores and m in downstream]
        if len(methods) < 3:
            print(f"# order {order}: only {len(methods)} arms in both sources, skipping")
            continue
        bpb = [scores[m] for m in methods]
        row = {"order": order, "arms": len(methods)}
        for col in ("core", "val_bpb"):
            target = [downstream[m][col] for m in methods if col in downstream[m]]
            if len(target) != len(methods):
                continue
            # Lower n-gram bpb should mean *higher* CORE and *lower* val_bpb, so negate the
            # CORE correlation to put both on a "positive means agreement" footing.
            sign = -1.0 if col == "core" else 1.0
            row[f"tau_{col}"] = sign * kendalltau(bpb, target).statistic
            row[f"rho_{col}"] = sign * spearmanr(bpb, target).statistic
        table.append(row)

    print(f"\nn-gram bpb vs downstream, {len(ngram_rows)} rows from {ngram_tsv}")
    print("positive = the cheap metric agrees with the expensive one\n")
    print(tabulate(table, headers="keys", tablefmt="github", floatfmt=".3f"))

    if seed_spread:
        spread = [
            {
                "method": m,
                "core": downstream[m].get("core"),
                "core_sd": downstream[m].get("core_sd"),
                "val_bpb": downstream[m].get("val_bpb"),
                "val_bpb_sd": downstream[m].get("val_bpb_sd"),
                "seeds": downstream[m].get("core_n") or downstream[m].get("val_bpb_n"),
                **{f"ngram_n{o}": by_order[o].get(m) for o in sorted(by_order)},
            }
            for m in METHOD_ORDER if m in downstream
        ]
        print("\nper-arm downstream means and seed spread "
              "(arms closer than ~1 sd are not separated by the GPU runs either)\n")
        print(tabulate(spread, headers="keys", tablefmt="github", floatfmt=".4f"))

    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in table for k in r}), delimiter="\t")
            writer.writeheader()
            writer.writerows(table)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(app())
