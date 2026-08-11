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
    val_bpb: float | None = None              # as nanochat reports it (see byte_factor)
    train_bpb: float | None = None
    # nanochat divides summed loss by the summed byte length of the target tokens, taking
    # each token's length from its own decoding. A tokenizer that elides a character
    # between two tokens has no token to charge it to, so that denominator is short and
    # bpb is inflated, in proportion to how much the scheme elides. byte_factor is
    # (summed token byte length) / (true UTF-8 length) over the same text; multiplying
    # val_bpb by it gives loss per true byte, which is comparable across tokenizers.
    byte_factor: float | None = None
    val_bpb_per_true_byte: float | None = None
    train_bpb_per_true_byte: float | None = None
    byte_factor_sample_bytes: int | None = None
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


# 0 means the whole validation shard. A 32 MB prefix was not enough: the running factor
# still moved by 1.2e-3 (plain) and 2.0e-3 (bnd_wpd) between 32 MB and 140 MB, an order of
# magnitude above the seed-to-seed spread of val bpb (2e-4 to 5e-4) that it would be
# compared against. base_eval scores 40 x 524288 tokens per split, which spans roughly
# 87-94 MB of text, so the factor has to be measured over at least that much.
BYTE_FACTOR_SAMPLE_BYTES = 0


def measure_byte_factor(adapter, base_dir: str, tokenizer_path: str,
                        sample_bytes: int = BYTE_FACTOR_SAMPLE_BYTES,
                        cache_dir: str | None = None,
                        device_batch_size: int = 32, sequence_len: int = 2048,
                        split_tokens: int = 40 * 524288, split: str = "val"):
    """(summed token byte length) / (true UTF-8 length) over the text bpb is scored on.

    nanochat's bpb divides summed loss by the summed byte length of the target tokens,
    where each token's length comes from decoding that token alone. A scheme that elides a
    character *between* two tokens (here, the single space between two delimited spans,
    rebuilt at decode time from the touching markers) has no token to charge that byte to,
    so the denominator is short and bpb is inflated. The shortfall scales with how much the
    scheme elides, which is exactly what it optimizes, so the raw number is not comparable
    across these tokenizers. Multiplying by this factor gives loss per true byte.

    Measured over the tokens nanochat's own loader emits, not over the shard. Those differ:
    the best-fit packer never places a document longer than the row capacity, which is
    21.7% of plain's bytes and 23.0% of bnd_w's, and the eval stops after a fixed token
    budget part-way through the shard. Estimating the factor on the whole shard therefore
    biased it by 0.14% (plain) to 0.27% (bnd_w), two to four times the seed spread, and
    moved the between-arm gap by 13.6%. Driving the real loader removes the estimate.

    Exactness also depends on the table below being the one the metric uses:
    `evaluate_bpb` masks the loss of any zero-byte token, so the numerator does depend on
    the byte table. Markers decode to the empty string and would be masked;
    `special_aware_token_bytes` floors them to one byte so their loss counts, and the same
    fictional byte sits in this denominator, so it cancels. Only BOS stays at zero.

    Cached under `cache_dir` keyed by the tokenizer's content, the shard, and the byte
    convention. Written to a temp file and renamed, since an arm's seeds run concurrently.
    """
    import hashlib
    import json
    import tempfile

    from .tokenizer import TOKEN_BYTES_CONVENTION, special_aware_token_bytes

    data_dir = Path(base_dir) / "base_data_climbmix"
    shards = sorted(data_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no ClimbMix shards under {data_dir} to measure the byte factor")
    shard = shards[-1]

    # Keyed on the tokenizer's CONTENT, not its path: retraining an arm writes the same
    # filename, and a path key would hand the new tokenizer the old factor, which then
    # multiplies every bpb number for that arm.
    with open(tokenizer_path, "rb") as f:
        tok_digest = hashlib.sha256(f.read()).hexdigest()
    shard_id = f"{shard.name}:{shard.stat().st_size}"
    spec = f"{split}:{device_batch_size}x{sequence_len}:{split_tokens}"
    key = hashlib.sha256(
        f"{tok_digest}:{shard_id}:{spec}:{TOKEN_BYTES_CONVENTION}".encode()
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir or base_dir, f"byte_factor_{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"[pynanochat] byte factor {cached['byte_factor']:.6f} (cached, "
              f"{cached['true_bytes']:,} true bytes, key {key})", flush=True)
        return cached["byte_factor"], cached["true_bytes"]

    # nanochat's own loader, with base_eval's defaults, so the token multiset is the one
    # the metric will be computed over rather than an approximation of it. This runs in the
    # parent, which unlike the training/eval children has neither the vendored clone on its
    # path nor NANOCHAT_BASE_DIR set, and list_parquet_files resolves through both.
    vendor = str(_default_nanochat_repo())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    prior_base = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = base_dir
    try:
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

        steps = split_tokens // (device_batch_size * sequence_len)
        table = special_aware_token_bytes(adapter)
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            adapter, device_batch_size, sequence_len, split, device="cpu"
        )
        measured = 0
        true_bytes = 0
        for _ in range(steps):
            _, targets = next(loader)
            for row in targets.tolist():
                measured += sum(table[t] for t in row)
                # decode is the inverse of encode and BOS decodes to "", so the byte length
                # of the decoded row is exactly the text those target tokens stand for.
                true_bytes += len(adapter.decode(row).encode("utf-8"))
    finally:
        if prior_base is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = prior_base
    factor = measured / true_bytes
    print(
        f"[pynanochat] byte factor {factor:.6f} "
        f"({measured:,} token bytes / {true_bytes:,} true bytes over {steps} scored steps)",
        flush=True,
    )

    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cache_path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump({"byte_factor": factor, "true_bytes": true_bytes,
                   "measured_token_bytes": measured, "tokenizer_sha256": tok_digest,
                   "shard": shard_id, "loader_spec": spec, "steps": steps,
                   "convention": TOKEN_BYTES_CONVENTION}, f, indent=2)
    os.replace(tmp, cache_path)
    return factor, true_bytes


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
    shuffle_data_order: bool = False,
    extra_train_args: list[str] | None = None,
) -> ExperimentResult:
    """Pretrain a nanochat base model on a script_bpe tokenizer and score CORE + bpb.

    `tokenizer_path` + `tokenizer_class` (a dotted path, e.g.
    ``script_bpe.tokenizers.bpe.BPETokenizer``) are what the child subprocess uses
    to rebuild the tokenizer — the live object can't cross a process boundary.
    `tokenizer` (an adapter) is optional and used only to report vocab_size.

    `shuffle_data_order` makes `seed` permute the training document order as well as the
    weight initialization; see `bootstrap._shuffle_data_order`. It defaults off because it
    changes what a given seed means, and results measured without it would not reproduce.
    """
    repo = Path(nanochat_repo) if nanochat_repo else _default_nanochat_repo()
    if not (repo / "scripts" / "base_train.py").exists():
        raise FileNotFoundError(f"vendored nanochat not found at {repo} (clone karpathy/nanochat there)")

    shared_dir = base_dir or os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    os.makedirs(shared_dir, exist_ok=True)
    # Per-run base dir. bootstrap writes the bpb byte table to
    # <base_dir>/tokenizer/token_bytes.pt and nanochat.tokenizer.get_token_bytes reads
    # that same path, so concurrent runs sharing one base_dir race on a single file and a
    # run can be scored against another tokenizer's byte table. Observed: a bnd_wpd_caps
    # run reported bnd_wpd's val bpb despite its own training loss. Checkpoints are keyed
    # by model tag and would not collide, but the byte table is not.
    base_dir = os.path.join(shared_dir, "runs", tokenizer_id)
    os.makedirs(base_dir, exist_ok=True)
    # The heavy, read-only artifacts stay shared: 700 MB of ClimbMix shards and the CORE
    # bundle should not be re-downloaded per run. Link them only when they already exist,
    # since base_eval tests `os.path.exists`, which follows a symlink and would try to
    # download over a broken one.
    for shared_name in ("base_data_climbmix", "eval_bundle"):
        src, dst = os.path.join(shared_dir, shared_name), os.path.join(base_dir, shared_name)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

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
    if shuffle_data_order:
        if seed is None:
            raise ValueError("shuffle_data_order needs a seed: the permutation is derived from it")
        env["PYNANOCHAT_SHUFFLE_DATA_ORDER"] = "1"
    if disable_compile:
        # base_train calls torch.compile() unconditionally; a smoke run shouldn't pay
        # the multi-minute compile, so run eager.
        env["TORCHDYNAMO_DISABLE"] = "1"

    py = sys.executable

    # 1) data: download train shards (+ the always-included val shard). Skip if present.
    data_dir = Path(base_dir) / "base_data_climbmix"
    have = sorted(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    # `num_train_shards` train shards plus the pinned val shard. The old test was
    # `len(have) < 2`, which a 2-shard --smoke leftover satisfies, so a run declared at 8
    # shards would have trained on 1 shard and cycled it ~37 times with nothing in the log.
    if have and len(have) < num_train_shards + 1:
        raise FileNotFoundError(
            f"{data_dir} holds {len(have)} shard(s) but this run declares "
            f"{num_train_shards} train shards plus a val shard. Refusing to train on a "
            f"smaller corpus than declared; delete the directory to re-download."
        )
    if len(have) < 2:
        # nanochat.dataset writes shard_XXXXX.parquet.tmp with no unique suffix and, on a
        # failed attempt, removes the FINAL file rather than the partial one. The data
        # directory is symlinked into every run, so 12 concurrent jobs would collide on one
        # temp name and could delete a shard another job is reading. Pre-download instead.
        raise FileNotFoundError(
            f"{data_dir} holds {len(have)} shard(s). Download them once before the sweep:\n"
            f"  NANOCHAT_BASE_DIR={shared_dir} python -m nanochat.dataset "
            f"-n {num_train_shards} -w 16\n"
            f"Refusing to download from inside a run: concurrent jobs share this directory."
        )
    else:
        print(f"[pynanochat] reusing {len(have)} existing data shards in {data_dir}", flush=True)

    # 1b) byte-accounting factor, measured before training on purpose: it needs only the
    # tokenizer and the val shard, and a failure here after training would throw away the
    # GPU hours over a CPU-only measurement. Also fails fast on an unreadable shard.
    byte_factor = byte_factor_sample = None
    if "bpb" in {m.strip() for m in eval_modes.split(",")} and tokenizer is not None:
        byte_factor, byte_factor_sample = measure_byte_factor(
            tokenizer, base_dir, tokenizer_path, cache_dir=shared_dir
        )

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
    train_out = _run_child(_launch_cmd(py, "scripts.base_train", train_flags, nproc, env),
                           cwd=repo, env=env, capture=True)

    # 3) eval on the final checkpoint (same GPU count).
    # `eval_modes` is not always "core,bpb": CORE's language_modeling tasks assert that
    # encode(context) is a prefix of encode(context + continuation), which a pretokenizer
    # whose marker depends on the following character does not satisfy. base_eval then
    # raises and the run yields nothing, bpb included. Pass "bpb" for those tokenizers.
    # Pin the step to the one just trained. base_eval otherwise resolves the checkpoint
    # with find_last_step, i.e. max(step) in the directory, so a checkpoint left by an
    # earlier round at a higher step would be evaluated instead and would print a complete,
    # plausible result block for the wrong weights.
    trained_step = _search_int(r"[Ss]aving (?:model )?checkpoint.*?(\d{4,})|model_(\d{6})\.pt", train_out)
    eval_flags = ["--eval", eval_modes, "--model-tag", tokenizer_id]
    if trained_step is not None:
        eval_flags += ["--step", str(trained_step)]
    else:
        print("[pynanochat] WARNING: could not read the trained step from base_train output; "
              "base_eval will resolve the checkpoint itself", flush=True)
    if device_batch_size is not None:
        eval_flags += ["--device-batch-size", str(device_batch_size)]
    if split_tokens is not None:
        eval_flags += ["--split-tokens", str(split_tokens)]
    if max_per_task is not None:
        eval_flags += ["--max-per-task", str(max_per_task)]
    out = _run_child(_launch_cmd(py, "scripts.base_eval", eval_flags, nproc, env), cwd=repo, env=env, capture=True)

    # 4) harvest. Only the metrics `eval_modes` asked for are present, so requiring a
    # CORE line from a bpb-only run would throw away a finished run over a metric that
    # was never requested. Anything that *was* requested is still required, so a silently
    # missing number still fails.
    modes = {m.strip() for m in eval_modes.split(",") if m.strip()}
    want_core = "core" in modes
    want_bpb = "bpb" in modes
    core_metric = _search_float(r"CORE metric:\s*([-\d.]+)", out) if want_core else None
    val_bpb = _search_float(r"val bpb:\s*([-\d.]+)", out) if want_bpb else None
    train_bpb = _search_float(r"train bpb:\s*([-\d.]+)", out) if want_bpb else None
    core_per_task = _read_core_csv(Path(base_dir) / "base_eval") if want_core else {}

    vocab_size = tokenizer.get_vocab_size() if tokenizer is not None else _vocab_from(tokenizer_path, tokenizer_class)

    # Correct the bpb denominator to true bytes, using the factor measured before training.
    val_bpb_true = train_bpb_true = None
    if byte_factor is not None:
        if val_bpb is not None:
            val_bpb_true = val_bpb * byte_factor
        if train_bpb is not None:
            train_bpb_true = train_bpb * byte_factor

    return ExperimentResult(
        tokenizer_id=tokenizer_id,
        depth=depth,
        vocab_size=vocab_size,
        core_metric=core_metric,
        core_per_task=core_per_task,
        val_bpb=val_bpb,
        train_bpb=train_bpb,
        byte_factor=byte_factor,
        val_bpb_per_true_byte=val_bpb_true,
        train_bpb_per_true_byte=train_bpb_true,
        byte_factor_sample_bytes=byte_factor_sample,
        artifact_dir=base_dir,
    )


def _search_int(pattern: str, text: str):
    """Last integer matching `pattern`, or None. Used to pin the evaluated checkpoint."""
    found = [g for m in re.findall(pattern, text) for g in (m if isinstance(m, tuple) else (m,)) if g]
    return int(found[-1]) if found else None


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
