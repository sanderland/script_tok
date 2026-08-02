#!/usr/bin/env python3
"""Which tokenizers can be scored on CORE, and which only on bpb.

nanochat's CORE language_modeling tasks build two prompts per example, one without the
continuation and one with it, and assert in core_eval.batch_sequences_lm that

    tokens_without == tokens_with[:len(tokens_without)]

A pretokenizer whose marker for a character depends on the character that follows does
not satisfy this. In bnd_wpd, ':' is token 1906 at the end of a string and token 1965
when a space follows, so appending " Boston" to "Answer:" rewrites a token that was
already emitted. base_eval then raises on the first such example and the run reports
nothing at all, bits-per-byte included, which is why the arms that fail here are run
with `--eval-modes bpb`.

This reuses nanochat's own render_prompts_lm and the real core.yaml task metadata
(continuation_delimiter, num_fewshot, the 1337 data shuffle and the 1234+idx few-shot
seed), so the prompts are the ones base_eval would actually tokenize.

    uv run python marker_experiments/downstream/core_prefix_check.py \
        --tokenizer-dir marker_experiments/downstream/tokenizers

Exit status is 1 if any tokenizer violates the property, so this doubles as a gate.
"""

import json
import os
import random
import sys

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VENDOR = os.path.join(REPO, "eval", "py-nanochat", "vendor", "nanochat")

app = cyclopts.App()


def _load_core_tasks(bundle: str):
    """The language_modeling tasks in core.yaml, with the metadata base_eval passes on."""
    import yaml

    config_path = os.path.join(bundle, "core.yaml")
    if not os.path.exists(config_path):
        raise SystemExit(
            f"CORE bundle not found at {bundle}. It is downloaded by the first "
            f"base_eval run; point --bundle at $NANOCHAT_BASE/eval_bundle."
        )
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return [
        {
            "label": t["label"],
            "dataset_uri": t["dataset_uri"],
            "num_fewshot": t["num_fewshot"][0],
            "continuation_delimiter": t.get("continuation_delimiter", " "),
        }
        for t in config["icl_tasks"]
        if t["icl_task_type"] == "language_modeling"
    ]


def _prompt_pairs(bundle: str, meta: dict, max_per_task: int):
    """Every (without, with) prompt pair base_eval would build for one task."""
    from nanochat.core_eval import render_prompts_lm

    path = os.path.join(bundle, "eval_data", meta["dataset_uri"])
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    random.Random(1337).shuffle(data)  # base_eval's shuffle, so we see its examples
    if max_per_task > 0:
        data = data[:max_per_task]

    pairs = []
    for idx, item in enumerate(data):
        fewshot = []
        if meta["num_fewshot"] > 0:
            rng = random.Random(1234 + idx)
            available = [i for i in range(len(data)) if i != idx]
            fewshot = [data[i] for i in rng.sample(available, meta["num_fewshot"])]
        pairs.append(render_prompts_lm(item, meta["continuation_delimiter"], fewshot))
    return pairs


@app.default
def main(
    tokenizer_dir: str = os.path.join(HERE, "tokenizers"),
    bundle: str | None = None,
    pattern: str = "",
    max_per_task: int = 60,
    trainer: str = "bpe",
) -> None:
    """Report, per tokenizer, how many CORE language_modeling examples would abort.

    Args:
        tokenizer_dir: Directory of .json.gz tokenizers to check.
        bundle: CORE eval_bundle directory (default $NANOCHAT_BASE/eval_bundle).
        pattern: Only check tokenizers whose filename contains this. A directory holding
            both bpe and mingram models needs one run per trainer, as run_arms.sh does.
        max_per_task: Examples per task, matching base_eval's --max-per-task.
        trainer: Which model class to load the files as, bpe or mingram.
    """
    sys.path.insert(0, REPO)
    sys.path.insert(0, VENDOR)

    from marker_experiments.downstream.boundary_tokenizer import (
        BoundaryBPETokenizer,
        BoundaryMinGramModel,
    )
    from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter

    if trainer not in ("bpe", "mingram"):
        raise SystemExit(f"unknown trainer {trainer!r}, expected bpe or mingram")
    model_cls = BoundaryBPETokenizer if trainer == "bpe" else BoundaryMinGramModel

    if bundle is None:
        base = os.environ.get("NANOCHAT_BASE")
        if not base:
            raise SystemExit("set NANOCHAT_BASE or pass --bundle")
        bundle = os.path.join(base, "eval_bundle")

    tasks = _load_core_tasks(bundle)
    per_task = {t["label"]: _prompt_pairs(bundle, t, max_per_task) for t in tasks}
    total = sum(len(v) for v in per_task.values())
    print(f"[core] {len(tasks)} language_modeling tasks, {total} examples at max_per_task={max_per_task}")

    names = sorted(n for n in os.listdir(tokenizer_dir) if n.endswith(".json.gz") and pattern in n)
    if not names:
        raise SystemExit(f"no tokenizers matching {pattern!r} in {tokenizer_dir}")

    failures = 0
    print(f"\n  {'tokenizer':<52} {'aborts':>12}  eval-modes")
    for name in names:
        try:
            adapter = ScriptBPETokenizerAdapter(model_cls.load(os.path.join(tokenizer_dir, name)))
        except KeyError as e:
            raise SystemExit(
                f"{name} does not load as a {trainer} model (missing {e}). "
                f"Narrow --pattern, or pass --trainer for this file's trainer."
            ) from e
        bos = adapter.get_bos_token_id()
        bad = 0
        for pairs in per_task.values():
            for without, with_ in pairs:
                a = adapter.encode(without, prepend=bos)
                b = adapter.encode(with_, prepend=bos)
                if not (len(a) < len(b) and a == b[: len(a)]):
                    bad += 1
        modes = "core,bpb" if bad == 0 else "bpb"
        failures += bad > 0
        print(f"  {name:<52} {bad:>5}/{total:<6}  {modes}")

    print(
        f"\n{failures} of {len(names)} tokenizer(s) cannot be scored on CORE. "
        f"Run those with --eval-modes bpb."
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    app()
