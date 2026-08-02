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
    # bpb normalized by true text bytes rather than by summed token byte length; this is
    # the figure that is comparable across tokenizers. See runner.measure_byte_factor.
    "byte factor": "byte_factor",
    "val bpb/byte": "val_bpb_true",
    "train bpb/byte": "train_bpb_true",
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
            cleaned = value.strip().replace(",", "")
            # A run scored on bits-per-byte alone prints "CORE metric : None". Recording
            # the literal string would put a non-numeric value in a numeric column, which
            # float() then trips over downstream. Leave the cell empty instead.
            if cleaned != "None":
                row[col] = cleaned

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

    lead = ["method", "arm", "trainer", "seed", "val_bpb_true", "train_bpb_true",
            "val_bpb", "train_bpb", "byte_factor", "core",
            "depth", "vocab_size", "tokenizer_id", "log"]
    tasks = sorted({k for r in rows for k in r if k.startswith("task_")})
    fields = [c for c in lead if any(c in r for r in rows)] + tasks

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    # Written then renamed: every job in the sweep runs this when it finishes, so plain
    # truncate-and-write lets two finishing jobs interleave into one half-written TSV,
    # which the table generator would then read without complaint.
    # mkstemp rather than the pid: every job in the sweep writes this file, and pids are
    # not unique across nodes.
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(out)), suffix=".tmp")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    print(f"[collect] {len(rows)} runs -> {out}")
    if skipped:
        print(f"[collect] {len(skipped)} log(s) without a result block: {', '.join(skipped)}")

    # Per-arm summary, so a cluster run is readable without loading the TSV.
    by_arm = {}
    for r in rows:
        if r.get("val_bpb_true") or r.get("val_bpb"):
            by_arm.setdefault(r.get("arm", r["method"]), []).append((
                float(r["val_bpb_true"]) if r.get("val_bpb_true") else None,
                float(r["val_bpb"]) if r.get("val_bpb") else None,
                float(r["core"]) if r.get("core") else None,
            ))
    if by_arm:
        # Sorted on bpb per true byte, the comparable one. Raw is shown alongside so the
        # size of the correction stays visible.
        def mean(vals, i):
            got = [v[i] for v in vals if v[i] is not None]
            return sum(got) / len(got) if got else None
        print(f"\n  {'arm':<16} {'n':>2}  {'bpb/true byte':>13}  {'raw bpb':>9}  {'CORE':>8}")
        for arm, vals in sorted(by_arm.items(),
                                key=lambda kv: mean(kv[1], 0) if mean(kv[1], 0) is not None else 9e9):
            true_bpb, raw_bpb, core = mean(vals, 0), mean(vals, 1), mean(vals, 2)
            print(f"  {arm:<16} {len(vals):>2}  "
                  f"{f'{true_bpb:13.4f}' if true_bpb is not None else '            -'}  "
                  f"{f'{raw_bpb:9.4f}' if raw_bpb is not None else '        -'}  "
                  f"{f'{core:8.4f}' if core is not None else '       -'}")


if __name__ == "__main__":
    app()
