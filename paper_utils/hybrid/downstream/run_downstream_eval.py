#!/usr/bin/env python3
"""Downstream LM eval for a script_bpe tokenizer, on a single H100.

Loads a trained script_bpe tokenizer (.json.gz), wraps it to nanochat's tokenizer
surface, and pretrains a small nanochat base model on it, reporting DCLM CORE +
bits-per-byte.

Examples:
    # quick end-to-end sanity check (~minutes, CORE ~ random)
    uv run python paper_utils/hybrid/downstream/run_downstream_eval.py \
        --tokenizer-path results/hybrid/fineweb_fi_5gb/bpe_n36044.model.json.gz \
        --smoke

    # real depth-12 eval (single GPU; longer)
    CUDA_VISIBLE_DEVICES=0 uv run python paper_utils/hybrid/downstream/run_downstream_eval.py \
        --tokenizer-path results/hybrid/fineweb_fi_5gb/bpe_n36044.model.json.gz \
        --depth 12 --num-shards 8

Run via `uv sync --extra downstream` first (pulls torch + the nanochat deps), and
ensure the vendored nanochat clone exists at eval/py-nanochat/vendor/nanochat.
"""

from cyclopts import App

# convenience aliases -> dotted class paths the child subprocess loads with.
TOKENIZER_CLASSES = {
    "bpe": "script_bpe.tokenizers.bpe.BPETokenizer",
    "unigram": "script_bpe.tokenizers.unigram.model.UnigramModel",
    "mingram": "script_bpe.tokenizers.mingram.model.MinGramModel",
    "convextok": "script_bpe.tokenizers.convextok.model.ConvexTokModel",
    "pathpiece": "script_bpe.tokenizers.pathpiece.model.PathPieceModel",
}

app = App()


@app.default
def cli(
    tokenizer_path: str,
    tokenizer_class: str = "bpe",
    depth: int = 12,
    ratio: float | None = None,
    gpus: int = 1,
    seed: int | None = None,
    smoke: bool = False,
    num_shards: int = 8,
    device_batch_size: int | None = None,
    total_batch_size: int | None = None,
    split_tokens: int | None = None,
    encode_workers: int | None = None,
    max_per_task: int | None = None,
    tokenizer_id: str | None = None,
    base_dir: str | None = None,
    eval_modes: str = "core,bpb",
) -> None:
    """Pretrain a nanochat base model on a script_bpe tokenizer and score CORE + bpb.

    Args:
        tokenizer_path: Path to a script_bpe tokenizer .json.gz model.
        tokenizer_class: Tokenizer kind: an alias (bpe/unigram/mingram/pathpiece/convextok) or a dotted class path.
        depth: nanochat model depth (model_dim = depth*64). depth=12 fits one H100 80GB.
        ratio: target tokens : scaling-params (nanochat default 12; speedrun uses 8). Sets the compute-optimal horizon.
        gpus: GPUs to train/eval on. >1 launches via torchrun (DDP); encode workers are split per rank.
        seed: Training seed (overrides nanochat's hardcoded 42). Use distinct seeds for replicates.
        smoke: Tiny end-to-end run (20 iters, batch 8, 2 shards, 8 examples/task) to prove the pipeline.
        num_shards: Number of train data shards to download (val shard always added).
        device_batch_size: Per-device batch size override (reduce to 16/8/4 if OOM).
        total_batch_size: Total tokens per optimizer step (-1/None = nanochat auto ~524288).
        split_tokens: Tokens per split for the bpb eval (smaller = faster; default ~21M).
        encode_workers: Processes for on-the-fly tokenization (script_bpe.encode is GIL-bound; >1 parallelizes it). Default: CPU-1 for real runs, 1 for --smoke.
        max_per_task: Cap examples per CORE task (eval speed).
        tokenizer_id: Model tag / checkpoint dir name (defaults to the tokenizer file stem).
        base_dir: NANOCHAT_BASE_DIR for data, tokenizer, checkpoints (defaults to ~/.cache/nanochat).
        eval_modes: What base_eval scores: "core,bpb", "bpb", or "core". Use "bpb" for a
            tokenizer whose pretokenizer marks a character from its right neighbour
            (bnd_wp, bnd_wpd and their _caps forms). CORE's language_modeling tasks
            assert that encode(context) is a prefix of encode(context + continuation);
            those tokenizers break it, base_eval raises, and the run reports nothing at
            all, bpb included.
    """
    import importlib
    import os

    import pynanochat as png

    dotted = TOKENIZER_CLASSES.get(tokenizer_class, tokenizer_class)
    module_name, _, cls_name = dotted.rpartition(".")
    cls = getattr(importlib.import_module(module_name), cls_name)
    adapter = png.ScriptBPETokenizerAdapter(cls.load(tokenizer_path))

    if tokenizer_id is None:
        stem = tokenizer_path.split("/")[-1].split(".")[0]
        tokenizer_id = f"{tokenizer_class}_{stem}"

    print(f"[downstream] tokenizer={tokenizer_path} class={dotted}")
    print(
        f"[downstream] vocab_size={adapter.get_vocab_size():,} bos_id={adapter.get_bos_token_id()} "
        f"depth={depth} tag={tokenizer_id} smoke={smoke}"
    )

    # smoke preset keeps the full path but tiny: prove download->inject->train->eval->parse.
    # script_bpe.encode is pure-Python, so tokenized-token COUNT dominates wall-clock;
    # shrink the per-step batch and the bpb split, not just the iteration count.
    smoke_iters = 20 if smoke else None
    if smoke:
        num_shards = min(num_shards, 2)
        device_batch_size = device_batch_size or 8
        total_batch_size = total_batch_size or (device_batch_size * 2048 * gpus)
        split_tokens = split_tokens or (device_batch_size * 2048 * gpus)
        max_per_task = max_per_task or 16
        if encode_workers is None:
            encode_workers = 1
    elif encode_workers is None:
        # real run: tokenize in parallel. Workers are per rank, so split CPUs across GPUs.
        encode_workers = max(1, ((os.cpu_count() or 2) - gpus) // gpus)

    print(f"[downstream] gpus={gpus} encode_workers/rank={encode_workers}")
    result = png.run_experiment(
        adapter,
        tokenizer_path=tokenizer_path,
        tokenizer_class=dotted,
        depth=depth,
        tokenizer_id=tokenizer_id,
        num_train_shards=num_shards,
        smoke_iters=smoke_iters,
        target_param_data_ratio=ratio,
        device_batch_size=device_batch_size,
        total_batch_size=total_batch_size,
        split_tokens=split_tokens,
        max_per_task=max_per_task,
        encode_workers=encode_workers,
        nproc=gpus,
        seed=seed,
        disable_compile=smoke,
        base_dir=base_dir,
        eval_modes=eval_modes,
    )

    print("\n" + "=" * 60)
    print("Downstream eval result")
    print("=" * 60)
    print(f"  tokenizer_id : {result.tokenizer_id}")
    print(f"  depth        : {result.depth}")
    print(f"  vocab_size   : {result.vocab_size:,}")
    print(f"  CORE metric  : {result.core_metric}")
    print(f"  val bpb      : {result.val_bpb}")
    print(f"  train bpb    : {result.train_bpb}")
    print(f"  artifact_dir : {result.artifact_dir}")
    if result.core_per_task:
        print("  per-task CORE (centered):")
        for task, val in result.core_per_task.items():
            print(f"    {task:<32} {val}")


if __name__ == "__main__":
    app()
