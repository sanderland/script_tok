#!/usr/bin/env python3
"""Does n-gram bpb separate two tokenizers, or is the gap within sampling noise?

The metric is deterministic given the text, so unlike the downstream sweep there is no
seed-to-seed spread to quote -- re-running it produces the identical number. That is not
the same as the number being precise: it is an estimate over one sample of held-out
documents, and a different sample moves it. On FineWeb English the whole spread between
seven arms at n>=3 is 0.13% of a bit, so "is this gap real" is the question that decides
whether the metric is usable at all.

The test is a **paired bootstrap over documents**. Every arm scores the same held-out
documents, so resampling documents (not arms) and recomputing each arm's bpb on the
resample keeps the pairing: document difficulty, by far the largest source of variance,
moves both arms together and cancels. An unpaired comparison would be swamped by it.

    uv run python paper_utils/hybrid/ngram/arm_significance.py \
        results/ngram/ngram_bpb_fineweb_en_5gb_docbits.npz --order 3
"""

import itertools
import sys

import numpy as np
from cyclopts import App
from tabulate import tabulate

app = App()
DEFAULT_RESAMPLES = 10_000


def paired_bootstrap(bits_a: np.ndarray, bits_b: np.ndarray, doc_bytes: np.ndarray,
                     resamples: int, seed: int) -> tuple[float, float, float]:
    """(observed bpb difference, 2.5th, 97.5th percentile) for arm A minus arm B.

    Resampling document indices rather than the bits themselves is what makes it paired:
    each draw takes the same documents from both arms.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(doc_bytes), size=(resamples, len(doc_bytes)))
    # bpb is a ratio of sums, not a mean of ratios, so it has to be recomputed per
    # resample rather than averaged -- long documents carry more weight, as they should.
    num = bits_a[idx].sum(axis=1) - bits_b[idx].sum(axis=1)
    den = doc_bytes[idx].sum(axis=1)
    diffs = num / den
    observed = (bits_a.sum() - bits_b.sum()) / doc_bytes.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return observed, float(lo), float(hi)


@app.default
def cli(docbits_npz: str, order: int = 3, resamples: int = DEFAULT_RESAMPLES,
        seed: int = 0, quiet: bool = False) -> None:
    """Report which arm pairs are separated at this n-gram order, with 95% CIs.

    Args:
        docbits_npz: The *_docbits.npz written next to a run_ngram_eval.py TSV.
        order: N-gram order to test.
        resamples: Bootstrap resamples.
        seed: RNG seed for the resampling.
        quiet: Print only the summary counts, not the per-pair table.
    """
    data = np.load(docbits_npz)
    doc_bytes = data["doc_bytes"].astype(np.float64)
    arms = sorted(k.split("|")[0] for k in data.files if k.endswith(f"|{order}"))
    if len(arms) < 2:
        raise SystemExit(f"need >=2 arms at order {order}; found {arms}")

    bits = {a: data[f"{a}|{order}"].astype(np.float64) for a in arms}
    ranked = sorted(arms, key=lambda a: bits[a].sum() / doc_bytes.sum())

    rows, separated = [], 0
    for a, b in itertools.combinations(ranked, 2):
        obs, lo, hi = paired_bootstrap(bits[a], bits[b], doc_bytes, resamples, seed)
        sig = not (lo <= 0 <= hi)
        separated += sig
        rows.append({"better": a, "worse": b, "delta_bpb": obs,
                     "ci_lo": lo, "ci_hi": hi, "separated": "yes" if sig else "no"})

    total = len(rows)
    print(f"\npaired bootstrap over {len(doc_bytes):,} held-out documents, "
          f"n={order}, {resamples:,} resamples")
    print(f"ranking (best first): {' < '.join(ranked)}")
    print(f"\n{separated}/{total} arm pairs separated at 95% "
          f"({separated / total * 100:.0f}%)")
    if not quiet:
        print()
        print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".5f"))
    print("\n'separated' means the 95% CI for the bpb difference excludes zero. A pair that is\n"
          "not separated is one this much held-out text cannot order; more text would narrow it.")


if __name__ == "__main__":
    sys.exit(app())
