#!/usr/bin/env python3
"""Measure the byte-accounting factor for each arm, once, into the shared cache.

`run_experiment` measures this itself, but every seed of an arm would repeat it and the
measurement is single-threaded over the whole validation shard (about 16 minutes per
arm). Running it here first means the sweep reads a cache, and a problem with the shard
or a tokenizer surfaces before any GPU time is spent.

    uv run python paper_utils/boundary/downstream/precompute_byte_factors.py

Writes byte_factor_<hash>.json into $NANOCHAT_BASE, which is exactly where
`measure_byte_factor(..., cache_dir=shared_dir)` looks.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "py-nanochat"))

app = cyclopts.App()

DOTTED = "paper_utils.boundary.downstream.boundary_tokenizer.BoundaryBPETokenizer"


def _one(args):
    arm, path, base = args
    import importlib

    from pynanochat.runner import measure_byte_factor
    from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter

    module_name, _, cls_name = DOTTED.rpartition(".")
    cls = getattr(importlib.import_module(module_name), cls_name)
    adapter = ScriptBPETokenizerAdapter(cls.load(path))
    factor, sample = measure_byte_factor(adapter, base, path, cache_dir=base)
    return arm, factor, sample


@app.default
def main(
    arms: str = "plain,bnd_w,bnd_wpd,bnd_wpd_caps",
    corpus: str = "fineweb_en_5gb_quick",
    trainer: str = "bpe",
    vocab: int = 34685,
    base_dir: str | None = None,
) -> None:
    """Measure and cache one factor per arm.

    Args:
        arms: Comma-separated arms.
        corpus: Corpus the tokenizers were trained on (part of their filename).
        trainer: Trainer used (part of their filename).
        vocab: Matched total vocabulary (part of their filename).
        base_dir: Shared NANOCHAT_BASE. Defaults to $NANOCHAT_BASE.
    """
    base = base_dir or os.environ.get("NANOCHAT_BASE")
    if not base:
        raise SystemExit("set NANOCHAT_BASE or pass --base-dir")

    jobs = []
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        path = os.path.join(HERE, "tokenizers", f"{corpus}_{arm}_{trainer}_v{vocab}.json.gz")
        if not os.path.exists(path):
            raise SystemExit(f"missing tokenizer for arm {arm}: {path}")
        jobs.append((arm, path, base))

    print(f"[factors] {len(jobs)} arm(s), whole val shard each, in parallel", flush=True)
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(_one, jobs))

    print(f"\n  {'arm':<14} {'byte factor':>12} {'sample bytes':>14}")
    for arm, factor, sample in sorted(results):
        print(f"  {arm:<14} {factor:>12.6f} {sample:>14,}")


if __name__ == "__main__":
    app()
