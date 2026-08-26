#!/usr/bin/env python3
"""Does n-gram bits-per-byte predict what the GPU runs found? Per order.

This is the experiment that decides whether the n-gram metric is worth anything. The
downstream sweep already scored the same nine arms with 20 seeds each of nanochat
pretraining; this ranks the CPU-only n-gram numbers against those and reports, for every
order, how well the cheap ranking reproduces the expensive one.

**Read the pairwise table, not the Kendall tau.** Tau over all arms is actively
misleading here, and measurably so: on the first real run it reported 0.14 against CORE,
because it averaged a handful of genuine agreements together with sixteen arm pairs that
the GPU sweep does not separate at all. The between-arm range in CORE (0.0059) is smaller
than the within-arm seed standard deviation (~0.006), so most of what tau scores is the
downstream sweep's own noise, and no metric could do better than chance on it.

So the honest question is narrower: *of the arm pairs the expensive sweep actually
resolves, how many does the cheap metric order correctly?* That is what the pairwise
section reports -- a Welch t-test per arm pair to find the resolved ones, then the n-gram
metric's hit rate on those, with a binomial p-value against the 50% chance baseline.

Tau is still printed, since it is what a reader expects, but treat it as a footnote.

`--seed-spread` prints the per-arm means and seed-to-seed spread behind all of this.

    uv run python paper_utils/hybrid/ngram/correlate_ngram_downstream.py \
        --ngram-tsv results/ngram/ngram_bpb_fineweb_en_5gb.tsv
"""

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import itertools

from cyclopts import App
from scipy.stats import binomtest, kendalltau, spearmanr, ttest_ind
from tabulate import tabulate

from paper_utils.hybrid.downstream_results import DOWNSTREAM_RESULTS_TSV, METHOD_ORDER, load_rows
from paper_utils.hybrid.token_usage_counts import TOKENIZER_TO_METHOD

app = App()


# Which downstream metrics we compare against, and whether lower is better for each.
TARGETS = {"core": False, "val_bpb": True}

# Tokenizer-spec name -> the method key that actually appears in the master TSV. This
# diverges from TOKENIZER_TO_METHOD in three places, and every divergence silently loses
# an arm rather than raising, which is why the mapping is restated here instead of reused:
#
#   mingram_pp_f8 -- the TSV calls this arm `mingram_mi_f8`, METHOD_ORDER calls it
#     `mingram_pp_f8`. load_rows filters on METHOD_ORDER and skips methods it finds no
#     rows for, so asking for 9 arms returns 8 with no warning. Verified: load_rows on the
#     committed TSV drops exactly this arm.
#   pathpiece / pathpiece_pb0.1 -- the TSV holds both a 24-seed `pathpiece` block and a
#     20-seed `pathpiece_pb01` block. The 24-seed rows are the pb0.2 model, so
#     TOKENIZER_TO_METHOD's `pathpiece_pb0.1 -> pathpiece` joins pb0.1 tokenizers onto
#     pb0.2 downstream rows. Mapping each setting to its own rows keeps them separate and,
#     as a side effect, gives two more arms to test on.
#
# Renaming the TSV or METHOD_ORDER is the paper's call, not this module's, so the
# translation lives here and `_downstream_seeds` reads the TSV unfiltered.
DOWNSTREAM_METHOD = {
    **TOKENIZER_TO_METHOD,
    "mingram_pp_f8": "mingram_mi_f8",
    "pathpiece_pb0.1": "pathpiece_pb01",
    "pathpiece_pb0.2": "pathpiece",
}


def _downstream_seeds(path: Path) -> dict[str, dict[str, list[float]]]:
    """Per-arm lists of per-seed values, which the pairwise tests need."""
    per_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # methods=None: the default filter is METHOD_ORDER, whose names do not all exist in the
    # TSV. Read everything and let DOWNSTREAM_METHOD do the matching.
    for row in load_rows(path, methods=None):
        for col in TARGETS:
            if row.get(col) not in (None, ""):
                per_method[row["method"]][col].append(float(row[col]))
    return per_method


def _arm_order(by_order: dict[int, dict[str, float]]) -> list[str]:
    """Arms to report, METHOD_ORDER first then any extras the TSV names differently."""
    seen = {m for scores in by_order.values() for m in scores}
    return [m for m in METHOD_ORDER if m in seen] + sorted(seen - set(METHOD_ORDER))


def _resolved_pairs(seed_values: dict[str, list[float]], methods: list[str], alpha: float):
    """Arm pairs the downstream sweep separates, by Welch t-test over seeds.

    Welch rather than Student: the arms do not have equal seed variance (CORE sd ranges
    from 0.0040 to 0.0087 across arms), and pooling it would overstate significance for
    the low-variance arms.
    """
    out = []
    for a, b in itertools.combinations(methods, 2):
        if a in seed_values and b in seed_values:
            p = ttest_ind(seed_values[a], seed_values[b], equal_var=False).pvalue
            if p < alpha:
                out.append((a, b, p))
    return out


def _downstream_by_method(path: Path) -> dict[str, dict[str, float]]:
    """Mean CORE and val_bpb per arm, plus the seed spread of each."""
    per_method = _downstream_seeds(path)
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
    alpha: float = 0.05,
    out: str | None = None,
) -> None:
    """Score n-gram bpb against downstream CORE and val_bpb, one row per order.

    Args:
        ngram_tsv: Output of run_ngram_eval.py.
        downstream_tsv: The GPU sweep's master results (method, seed, val_bpb, core).
        seed_spread: Also print each arm's downstream mean and seed-to-seed spread.
        alpha: Significance level for deciding which arm pairs the sweep resolves.
        out: Write the correlation table to this TSV as well as printing it.
    """
    seed_values = _downstream_seeds(Path(downstream_tsv))
    downstream = _downstream_by_method(Path(downstream_tsv))
    if not downstream:
        raise SystemExit(f"no downstream rows in {downstream_tsv}")

    with open(ngram_tsv) as f:
        ngram_rows = list(csv.DictReader(f, delimiter="\t"))

    by_order: dict[int, dict[str, float]] = defaultdict(dict)
    unmapped, no_rows = set(), set()
    for row in ngram_rows:
        method = DOWNSTREAM_METHOD.get(row["tokenizer_id"])
        if method is None:
            unmapped.add(row["tokenizer_id"])
            continue
        if method not in seed_values:
            no_rows.add(f"{row['tokenizer_id']} -> {method}")
            continue
        by_order[int(row["order"])][method] = float(row["bpb"])
    if unmapped:
        print(f"# no downstream counterpart, ignored: {', '.join(sorted(unmapped))}")
    if no_rows:
        # Loud on purpose. A scored tokenizer whose downstream rows cannot be found is the
        # exact failure that silently turned 9 arms into 8; it must never pass unremarked.
        raise SystemExit(
            "these arms were scored but have no rows in the downstream TSV:\n  "
            + "\n  ".join(sorted(no_rows))
            + "\nCheck DOWNSTREAM_METHOD against the TSV's method column."
        )

    table = []
    for order in sorted(by_order):
        scores = by_order[order]
        methods = [m for m in _arm_order(by_order) if m in scores and m in downstream]
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

    # The headline: agreement restricted to the pairs the GPU sweep actually resolves.
    print(f"\nn-gram bpb vs downstream, {len(ngram_rows)} rows from {ngram_tsv}")
    for col, better_low in TARGETS.items():
        values = {m: v[col] for m, v in seed_values.items() if col in v}
        methods = [m for m in _arm_order(by_order) if m in values and any(m in s for s in by_order.values())]
        pairs = _resolved_pairs(values, methods, alpha)
        n_all = len(methods) * (len(methods) - 1) // 2
        print(f"\n{col}: {len(pairs)}/{n_all} arm pairs separated downstream (Welch p<{alpha})")
        if not pairs:
            print("  none — the sweep does not resolve these arms on this metric, so there is "
                  "nothing here for any cheap metric to predict.")
            continue
        rows_out = []
        for order in sorted(by_order):
            scores = by_order[order]
            if not all(m in scores for m in methods):
                continue
            hits = 0
            for a, b, _ in pairs:
                mean_a = statistics.fmean(values[a])
                mean_b = statistics.fmean(values[b])
                down_a_better = (mean_a < mean_b) if better_low else (mean_a > mean_b)
                hits += down_a_better == (scores[a] < scores[b])   # lower bpb = better
            rows_out.append({
                "order": order, "correct": f"{hits}/{len(pairs)}",
                "pct": 100 * hits / len(pairs),
                "binomial_p": binomtest(hits, len(pairs), 0.5, alternative="greater").pvalue,
            })
        print(tabulate(rows_out, headers="keys", tablefmt="github", floatfmt=".3f"))
        missed = [f"{a} vs {b}" for a, b, _ in pairs
                  if ((statistics.fmean(values[a]) < statistics.fmean(values[b])) if better_low
                      else (statistics.fmean(values[a]) > statistics.fmean(values[b])))
                  != (by_order[max(by_order)][a] < by_order[max(by_order)][b])]
        if missed:
            print(f"  missed at n={max(by_order)}: {'; '.join(missed)}")

    print("\nKendall/Spearman over all arms — a footnote, not the headline: these average the\n"
          "resolved pairs together with pairs the sweep leaves as noise (see module docstring).\n")
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
            for m in _arm_order(by_order) if m in downstream
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
