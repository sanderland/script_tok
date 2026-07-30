#!/usr/bin/env python3
"""Parse run_downstream_eval.py logs into one TSV.

`pynanochat.run_experiment` returns an `ExperimentResult` but persists nothing, and
`run_downstream_eval.py` only prints it, so the logs are the record. This reads the
result block each run prints:

    ============================================================
    Downstream eval result
    ============================================================
      tokenizer_id : bnd_wpd_bpe_d12_s0
      depth        : 12
      vocab_size   : 34,686
      CORE metric  : 0.1234
      val bpb      : 0.9876
      train bpb    : 0.9800
      artifact_dir : ...
      per-task CORE (centered):
        hellaswag_zeroshot               0.0421
        ...

Columns match paper_utils/hybrid/downstream_results.py's expectations (method, seed,
val_bpb, train_bpb, core) so the existing table generators can read this file, with
per-task CORE in the remaining columns.

    uv run python marker_experiments/downstream/collect_results.py \
        --logs-dir results/marker_downstream/logs --out results/marker_downstream/results.tsv
"""

import csv
import os
import re

import cyclopts

app = cyclopts.App()

SCALARS = {
    "tokenizer_id": "tokenizer_id",
    "depth": "depth",
    "vocab_size": "vocab_size",
    "CORE metric": "core",
    "val bpb": "val_bpb",
    "train bpb": "train_bpb",
}
# tokenizer_id is <arm>_<trainer>_d<depth>_s<seed>, e.g. bnd_wpd_caps_bpe_d12_s0
TAG_RE = re.compile(r"^(?P<arm>.+)_(?P<trainer>bpe|mingram)_d(?P<depth>\d+)_s(?P<seed>\d+)$")


def parse_log(path):
    """Return one row, or None if the run did not reach a result block."""
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    try:
        start = max(i for i, l in enumerate(lines) if l.strip() == "Downstream eval result")
    except ValueError:
        return None

    row, in_tasks = {}, False
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("per-task CORE"):
            in_tasks = True
            continue
        if in_tasks:
            parts = stripped.split()
            if len(parts) == 2:
                row[f"task_{parts[0]}"] = parts[1]
            continue
        key, _, value = stripped.partition(":")
        col = SCALARS.get(key.strip())
        if col:
            row[col] = value.strip().replace(",", "")

    if "core" not in row and "val_bpb" not in row:
        return None
    tag = row.get("tokenizer_id", os.path.basename(path).removesuffix(".log"))
    m = TAG_RE.match(tag)
    if m:
        # `method` is what the paper's table generators key on.
        row["method"] = f"{m['arm']}_{m['trainer']}"
        row["arm"] = m["arm"]
        row["trainer"] = m["trainer"]
        row["seed"] = m["seed"]
    else:
        row["method"] = tag
    row["log"] = os.path.basename(path)
    return row


@app.default
def main(logs_dir: str, out: str = "results.tsv") -> None:
    """Collect every finished run under `logs_dir` into `out`.

    Args:
        logs_dir: Directory of .log files written by run_arms.sh.
        out: TSV to write.
    """
    rows, skipped = [], []
    for name in sorted(os.listdir(logs_dir)):
        if not name.endswith(".log"):
            continue
        row = parse_log(os.path.join(logs_dir, name))
        (rows.append(row) if row else skipped.append(name))

    if not rows:
        raise SystemExit(f"no finished runs in {logs_dir} ({len(skipped)} incomplete)")

    lead = ["method", "arm", "trainer", "seed", "val_bpb", "train_bpb", "core",
            "depth", "vocab_size", "tokenizer_id", "log"]
    tasks = sorted({k for r in rows for k in r if k.startswith("task_")})
    fields = [c for c in lead if any(c in r for r in rows)] + tasks

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[collect] {len(rows)} runs -> {out}")
    if skipped:
        print(f"[collect] {len(skipped)} log(s) without a result block: {', '.join(skipped)}")

    # Per-arm summary, so a cluster run is readable without loading the TSV.
    by_arm = {}
    for r in rows:
        if r.get("val_bpb"):
            by_arm.setdefault(r.get("arm", r["method"]), []).append(
                (float(r["val_bpb"]), float(r["core"]) if r.get("core") else None)
            )
    if by_arm:
        print(f"\n  {'arm':<16} {'n':>2}  {'val_bpb':>9}  {'CORE':>8}")
        for arm, vals in sorted(by_arm.items(), key=lambda kv: sum(v[0] for v in kv[1]) / len(kv[1])):
            bpb = sum(v[0] for v in vals) / len(vals)
            cores = [v[1] for v in vals if v[1] is not None]
            core = f"{sum(cores) / len(cores):8.4f}" if cores else "       -"
            print(f"  {arm:<16} {len(vals):>2}  {bpb:9.4f}  {core}")


if __name__ == "__main__":
    app()
