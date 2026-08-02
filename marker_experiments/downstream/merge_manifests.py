#!/usr/bin/env python3
"""Merge the per-job tokenizer manifests into one, and check the vocabularies match.

Training one arm per Slurm job means one manifest per job (see train_matched.py's
`manifest_path`), because that file is rewritten by read-modify-write and concurrent
jobs would drop each other's entries. This folds them back into a single manifest.json
and applies the cross-arm check that a single-process run of train_matched.py does at
the end: every arm of the same trainer and corpus must land on the same total vocabulary,
since that is the whole point of the matched training.

    uv run python marker_experiments/downstream/merge_manifests.py \\
        --parts "results/marker_downstream/manifests/*.json"
"""

import glob
import json
import os

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "manifest.json")

app = cyclopts.App()


@app.default
def main(parts: str, out: str = MANIFEST, check: bool = True) -> None:
    """Merge every manifest matching `parts` into `out`.

    Args:
        parts: Glob for the per-job manifests.
        out: Merged manifest to write.
        check: Fail if arms sharing a trainer and corpus disagree on total vocabulary.
    """
    paths = sorted(glob.glob(parts))
    if not paths:
        raise SystemExit(f"no manifests matched {parts!r}")

    merged: dict = json.load(open(out)) if os.path.exists(out) else {}
    for path in paths:
        part = json.load(open(path))
        for key, info in part.items():
            if key in merged and merged[key] != info:
                # Two jobs recorded different numbers for the same arm. Refusing here
                # rather than picking one, since the difference is the finding.
                raise SystemExit(
                    f"{key} recorded twice with different values:\n"
                    f"  existing: {json.dumps(merged[key], sort_keys=True)}\n"
                    f"  {path}: {json.dumps(info, sort_keys=True)}"
                )
            merged[key] = info

    if check:
        groups: dict[tuple[str, str], dict[str, int]] = {}
        for key, info in merged.items():
            groups.setdefault((info["trainer"], info["corpus"]), {})[info["arm"]] = info["total_vocab"]
        for (trainer, corpus), sizes in sorted(groups.items()):
            if len(set(sizes.values())) > 1:
                raise SystemExit(
                    f"vocabulary sizes are NOT matched for {trainer} on {corpus}: {sizes}"
                )

    with open(out, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)

    print(f"[merge] {len(paths)} part(s) -> {out}: {len(merged)} arm(s)")
    for key, info in sorted(merged.items()):
        cpt = info.get("eval_chars_per_token")
        print(
            f"  {info['arm']:<14} vocab={info['total_vocab']:,} "
            + (f"ch/tok={cpt:.4f} " if cpt else "")
            + f"rt_fail={info.get('roundtrip_failures', '?')}"
        )


if __name__ == "__main__":
    app()
