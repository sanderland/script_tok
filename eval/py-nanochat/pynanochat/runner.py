"""Thin driver: any script_bpe tokenizer -> pretrain (nanochat) -> CORE + bpb.

A shell-out wrapper around nanochat's own scripts, not a reimplementation.
nanochat does budgeted depth-sized pretraining and the faithful CORE/bpb eval; we
orchestrate three subprocesses and harvest the numbers.

Flow per experiment (all on a SINGLE GPU — plain `python -m`, no torchrun):
  1. download a few data shards:  python -m nanochat.dataset -n N
  2. train:                       scripts.base_train --depth D ...
  3. eval:                        scripts.base_eval  --eval core,bpb ...
Steps 2 and 3 run via a bootstrap import (pynanochat.bootstrap) that rebuilds the
script_bpe tokenizer from env vars and injects it into nanochat before argparse.

nanochat is *vendored* (a git clone under eval/py-nanochat/vendor/nanochat) rather than
pip-installed: it is a flat-layout clone-and-run repo, not a package. We run its
scripts with cwd = that clone so `import nanochat` / `scripts.*` resolve.
"""

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# single-GPU child: inject the tokenizer (on import) then run the nanochat module.
_BOOT = "import pynanochat.bootstrap; import runpy; runpy.run_module({mod!r}, run_name='__main__')"
_ENTRY = str(Path(__file__).resolve().parent / "_torchrun_entry.py")


def _launch_cmd(py: str, module: str, flags: list[str], nproc: int, env: dict) -> list[str]:
    """Build the command to run a nanochat module, single-GPU or multi-GPU (torchrun).

    Single GPU: `python -c "<bootstrap>; run_module(module)" <flags>`.
    Multi GPU:  `python -m torch.distributed.run --standalone --nproc_per_node=N
                 _torchrun_entry.py <flags>` with PYNANOCHAT_TARGET=module.
    Either way the bootstrap injects the tokenizer before the module's argparse runs.
    """
    if nproc > 1:
        env["PYNANOCHAT_TARGET"] = module
        return [py, "-m", "torch.distributed.run", "--standalone",
                f"--nproc_per_node={nproc}", _ENTRY, *flags]
    return [py, "-c", _BOOT.format(mod=module), *flags]


@dataclass
class ExperimentResult:
    tokenizer_id: str
    depth: int
    vocab_size: int
    core_metric: float | None = None          # headline (DCLM CORE, centered)
    core_per_task: dict = field(default_factory=dict)
    val_bpb: float | None = None              # intrinsic companion
    train_bpb: float | None = None
    artifact_dir: str | None = None


def _default_nanochat_repo() -> Path:
    return Path(os.environ.get("NANOCHAT_REPO", Path(__file__).resolve().parent.parent / "vendor" / "nanochat"))


def _run_child(argv: list[str], *, cwd: Path, env: dict, capture: bool = False) -> str:
    """Run a subprocess, streaming its output to our stdout; optionally capture it."""
    print(f"\n$ (cwd={cwd}) {' '.join(argv)}\n", flush=True)
    chunks: list[str] = []
    proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if capture:
            chunks.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, argv)
    return "".join(chunks)


def run_experiment(
    tokenizer=None,
    *,
    tokenizer_path: str,
    tokenizer_class: str,
    depth: int = 12,
    tokenizer_id: str = "tokenizer",
    num_train_shards: int = 8,
    smoke_iters: int | None = None,
    target_param_data_ratio: float | None = None,
    device_batch_size: int | None = None,
    total_batch_size: int | None = None,
    split_tokens: int | None = None,
    max_per_task: int | None = None,
    base_dir: str | None = None,
    nanochat_repo: str | None = None,
    window_pattern: str = "L",
    disable_compile: bool = False,
    encode_workers: int | None = None,
    nproc: int = 1,
    seed: int | None = None,
    eval_modes: str = "core,bpb",
    extra_train_args: list[str] | None = None,
) -> ExperimentResult:
    """Pretrain a nanochat base model on a script_bpe tokenizer and score CORE + bpb.

    `tokenizer_path` + `tokenizer_class` (a dotted path, e.g.
    ``script_bpe.tokenizers.bpe.BPETokenizer``) are what the child subprocess uses
    to rebuild the tokenizer — the live object can't cross a process boundary.
    `tokenizer` (an adapter) is optional and used only to report vocab_size.
    """
    repo = Path(nanochat_repo) if nanochat_repo else _default_nanochat_repo()
    if not (repo / "scripts" / "base_train.py").exists():
        raise FileNotFoundError(f"vendored nanochat not found at {repo} (clone karpathy/nanochat there)")

    base_dir = base_dir or os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    os.makedirs(base_dir, exist_ok=True)

    env = dict(os.environ)
    env["NANOCHAT_BASE_DIR"] = base_dir
    # absolute: the child runs with cwd=nanochat clone, so a relative path would break.
    env["PYNANOCHAT_TOKENIZER"] = os.path.abspath(str(tokenizer_path))
    env["PYNANOCHAT_TOK_CLASS"] = tokenizer_class
    # Put the vendored nanochat clone on PYTHONPATH so `import nanochat`/`scripts.*`
    # resolve. The single-GPU `python -c` path gets this free via cwd, but torchrun runs
    # `python entry.py`, whose sys.path[0] is the script dir, not cwd.
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo), env.get("PYTHONPATH", "")) if p)
    env.setdefault("OMP_NUM_THREADS", "1")
    if encode_workers is not None:
        # parallel on-the-fly tokenization (adapter spawns a fork pool of this size)
        env["PYNANOCHAT_ENCODE_WORKERS"] = str(encode_workers)
    if seed is not None:
        env["PYNANOCHAT_SEED"] = str(seed)  # bootstrap overrides nanochat's hardcoded seed 42
    if disable_compile:
        # base_train calls torch.compile() unconditionally; a smoke run shouldn't pay
        # the multi-minute compile, so run eager.
        env["TORCHDYNAMO_DISABLE"] = "1"

    py = sys.executable

    # 1) data: download train shards (+ the always-included val shard). Skip if present.
    data_dir = Path(base_dir) / "base_data_climbmix"
    have = sorted(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    if len(have) < 2:
        _run_child([py, "-m", "nanochat.dataset", "-n", str(num_train_shards), "-w", "16"], cwd=repo, env=env)
    else:
        print(f"[pynanochat] reusing {len(have)} existing data shards in {data_dir}", flush=True)

    # 2) train (1 or N GPUs). Skip mid-run CORE/sampling/checkpoints; eval at the end.
    train_flags = [
        "--depth", str(depth),
        "--run", "dummy",
        "--model-tag", tokenizer_id,
        # This box has no FA3; nanochat's default sliding-window pattern (SSSL) falls
        # back to SDPA with terrible utilization, so use full-context attention (L).
        "--window-pattern", window_pattern,
        # Disable all in-loop evals; we harvest bpb + CORE once at the end via base_eval.
        # (eval-every defaults to 250 and fires at step 0, running a ~42M-token val bpb
        # pass before the first step — a multi-minute stall with pure-Python tokenization.)
        "--eval-every", "-1",
        "--core-metric-every", "-1",
        "--sample-every", "-1",
        "--save-every", "-1",
    ]
    if smoke_iters is not None:
        train_flags += ["--num-iterations", str(smoke_iters)]
    if target_param_data_ratio is not None:
        train_flags += ["--target-param-data-ratio", str(target_param_data_ratio)]
    if device_batch_size is not None:
        train_flags += ["--device-batch-size", str(device_batch_size)]
    if total_batch_size is not None:
        train_flags += ["--total-batch-size", str(total_batch_size)]
    if extra_train_args:
        train_flags += list(extra_train_args)
    _run_child(_launch_cmd(py, "scripts.base_train", train_flags, nproc, env), cwd=repo, env=env)

    # 3) eval on the final checkpoint (same GPU count).
    # `eval_modes` is not always "core,bpb": CORE's language_modeling tasks assert that
    # encode(context) is a prefix of encode(context + continuation), which a pretokenizer
    # whose marker depends on the following character does not satisfy. base_eval then
    # raises and the run yields nothing, bpb included. Pass "bpb" for those tokenizers.
    eval_flags = ["--eval", eval_modes, "--model-tag", tokenizer_id]
    if device_batch_size is not None:
        eval_flags += ["--device-batch-size", str(device_batch_size)]
    if split_tokens is not None:
        eval_flags += ["--split-tokens", str(split_tokens)]
    if max_per_task is not None:
        eval_flags += ["--max-per-task", str(max_per_task)]
    out = _run_child(_launch_cmd(py, "scripts.base_eval", eval_flags, nproc, env), cwd=repo, env=env, capture=True)

    # 4) harvest.
    core_metric = _search_float(r"CORE metric:\s*([-\d.]+)", out)
    val_bpb = _search_float(r"val bpb:\s*([-\d.]+)", out)
    train_bpb = _search_float(r"train bpb:\s*([-\d.]+)", out)
    core_per_task = _read_core_csv(Path(base_dir) / "base_eval")

    vocab_size = tokenizer.get_vocab_size() if tokenizer is not None else _vocab_from(tokenizer_path, tokenizer_class)

    return ExperimentResult(
        tokenizer_id=tokenizer_id,
        depth=depth,
        vocab_size=vocab_size,
        core_metric=core_metric,
        core_per_task=core_per_task,
        val_bpb=val_bpb,
        train_bpb=train_bpb,
        artifact_dir=base_dir,
    )


def _search_float(pattern: str, text: str) -> float:
    m = re.findall(pattern, text)
    if not m:
        raise ValueError(f"metric pattern {pattern!r} not found in subprocess output")
    return float(m[-1])


def _read_core_csv(eval_dir: Path) -> dict:
    """Read the newest base_model_*.csv per-task table into {task: centered}."""
    if not eval_dir.exists():
        return {}
    csvs = sorted(eval_dir.glob("base_model_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        return {}
    per_task: dict[str, float] = {}
    for line in csvs[-1].read_text().splitlines()[1:]:  # skip header
        parts = [c.strip() for c in line.split(",")]
        if len(parts) >= 3 and parts[0] not in ("", "CORE"):
            per_task[parts[0]] = float(parts[2])
    return per_task


def _vocab_from(tokenizer_path: str, tokenizer_class: str) -> int:
    import importlib
    from .tokenizer_adapter import ScriptBPETokenizerAdapter
    module_name, _, cls_name = tokenizer_class.rpartition(".")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return ScriptBPETokenizerAdapter(cls.load(tokenizer_path)).get_vocab_size()
