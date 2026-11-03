"""CPU bilingual ablation with parent-side process-group memory monitoring.

Run from the repository root:
  .venv/bin/python -m paper_utils.boundary.run_bilingual
  .venv/bin/python -m paper_utils.boundary.run_bilingual --stage prepare
No network or GPU is used. Existing paper artifacts are not outputs of this run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RSS_POLL_SECONDS = 5


class MemoryLimitExceeded(RuntimeError):
    """A child process group exceeded the runner's aggregate RSS limit."""


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_process_group_rss(ps_output, pgid):
    """Return aggregate RSS in bytes for ``pgid`` from ``ps`` machine output."""
    total_kib = 0
    for line in ps_output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            _pid, row_pgid, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        if row_pgid == pgid:
            total_kib += max(rss_kib, 0)
    return total_kib * 1024


def process_group_rss_bytes(pgid, ps_runner=subprocess.run):
    """Read aggregate RSS for a process group on macOS and Linux."""
    result = ps_runner(
        ["ps", "-axo", "pid=,pgid=,rss="],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_process_group_rss(result.stdout, pgid)


def terminate_process_group(process, *, killpg=os.killpg):
    """Stop only the process group created for one launched child."""
    try:
        killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        # The leader can exit before forkserver workers. Kill the original group anyway.
        killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return True


def monitor_child(process, max_rss_bytes, on_poll, *, rss_reader=process_group_rss_bytes,
                  sleep=time.sleep):
    """Wait for a launched child while recording its process-group RSS."""
    highwater_bytes = 0
    try:
        while True:
            rss_bytes = rss_reader(process.pid)
            highwater_bytes = max(highwater_bytes, rss_bytes)
            on_poll(rss_bytes, highwater_bytes)
            if rss_bytes > max_rss_bytes:
                raise MemoryLimitExceeded(
                    f"Child process group {process.pid} exceeded --max-rss-gib "
                    f"({rss_bytes / 1024**3:.2f} GiB > {max_rss_bytes / 1024**3:.2f} GiB)."
                )
            returncode = process.poll()
            if returncode is not None:
                if returncode:
                    raise subprocess.CalledProcessError(returncode, process.args)
                return highwater_bytes
            sleep(RSS_POLL_SECONDS)
    except BaseException:
        terminate_process_group(process)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--million-chars", type=int, default=2000)
    parser.add_argument("--english-percent", type=int, default=50)
    parser.add_argument("--sample-version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--trainer", choices=["bpe", "mingram"], default="bpe")
    parser.add_argument("--vocab", type=int, default=34685)
    parser.add_argument("--eval-chars", type=int, default=500_000)
    parser.add_argument("--max-rss-gib", type=float, default=20)
    parser.add_argument("--arms", default="plain,bnd_wpd,bnd_wpd_caps")
    parser.add_argument("--stage", choices=["all", "prepare", "train", "analyze"], default="all")
    parser.add_argument("--child", choices=["prepare", "train", "analyze"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.million_chars < 1 or args.workers < 1 or args.eval_chars < 1 or args.max_rss_gib <= 0:
        parser.error("million-chars, workers, and eval-chars must be positive; max-rss-gib must be greater than zero")
    return args


def child_main(args, corpus, base, out, arms):
    """Run one stage. Corpus imports deliberately stay in the child."""
    sample = None
    if args.child in {"prepare", "train"}:
        from script_bpe.corpus.bilingual import prepare_mixture

        _, sample = prepare_mixture(corpus, base)
    if args.child == "prepare":
        print(f"Prepared: {corpus}", flush=True)
        return

    if len(arms) != 1:
        raise ValueError("child requires one arm")
    arm = arms[0]
    tokenizer_path = out / f"{arm}.json.gz"
    info_path = out / f"{arm}.train.json"
    if args.child == "train":
        from paper_utils.boundary.downstream.train_matched import train_one

        if tokenizer_path.exists() and info_path.exists():
            info = json.loads(info_path.read_text())
            if info["trainer"] != args.trainer or info["sample"] != sample or info["tokenizer_sha256"] != hashlib.sha256(tokenizer_path.read_bytes()).hexdigest():
                raise ValueError(f"Cached training provenance mismatch: {tokenizer_path}")
            print(f"Reusing {tokenizer_path}", flush=True)
            return
        started = time.monotonic()
        tokenizer, info = train_one(arm, args.trainer, corpus, args.vocab, args.workers, 1.15, str(base), None)
        if len(tokenizer.tokens) != args.vocab:
            raise RuntimeError(f"Unfilled vocabulary: {len(tokenizer.tokens)} != {args.vocab}")
        tokenizer.save(str(tokenizer_path))
        info.update(sample=sample, workers=args.workers, wall_seconds=time.monotonic() - started,
                    tokenizer_sha256=hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
        write_json(info_path, info)
        return

    from paper_utils.boundary.bilingual_analysis import analyze_tokenizer
    from script_bpe.tokenizers.load import load_tokenizer

    tokenizer = load_tokenizer(str(tokenizer_path))
    eval_texts, provenance = {}, {}
    for lang in ("en", "ko"):
        path = ROOT / f"paper_utils/boundary/eval_texts/goldfish_{lang}.json"
        remaining = args.eval_chars
        selected = []
        for text in json.loads(path.read_text()):
            text = text[:remaining]
            if text:
                selected.append(text)
                remaining -= len(text)
            if not remaining:
                break
        if remaining:
            raise ValueError(f"Insufficient evaluation characters in {path}")
        eval_texts[lang] = selected
        provenance[lang] = {"path": str(path.relative_to(ROOT)), "characters": args.eval_chars,
                            "selected_sha256": hashlib.sha256(json.dumps(selected, ensure_ascii=False).encode()).hexdigest()}
    result = analyze_tokenizer(tokenizer, eval_texts)
    write_json(out / f"{arm}.analysis.json", {"arm": arm, "trainer": args.trainer, "corpus": corpus, "evaluation": provenance, "analysis": result})


def parent_main(args, corpus, out, arms):
    max_rss_bytes = int(args.max_rss_gib * 1024**3)
    status = {"corpus": corpus, "max_rss_gib": args.max_rss_gib, "stages": []}
    highwater = {"corpus": corpus, "max_rss_gib": args.max_rss_gib, "stages": {}}
    status_path = out / "runner_status.json"
    highwater_path = out / "rss_highwater.json"
    if status_path.exists():
        status["stages"] = json.loads(status_path.read_text())["stages"]
    if highwater_path.exists():
        highwater["stages"] = json.loads(highwater_path.read_text())["stages"]
    stages = ["prepare", "train", "analyze"] if args.stage == "all" else [args.stage]
    for stage in stages:
        stage_arms = ["prepare"] if stage == "prepare" else arms
        for arm in stage_arms:
            key = stage if stage == "prepare" else f"{stage}:{arm}"
            log = out / f"{key.replace(':', '.')}.log"
            entry = {"stage": stage, "arm": None if stage == "prepare" else arm,
                     "status": "running", "started_unix": time.time(), "log": str(log.relative_to(ROOT))}
            status["stages"].append(entry)
            highwater["stages"][key] = {"process_group": None, "highwater_bytes": 0}
            write_json(status_path, status)
            write_json(highwater_path, highwater)
            command = [sys.executable, "-m", "paper_utils.boundary.run_bilingual", "--child", stage,
                       "--million-chars", str(args.million_chars), "--english-percent", str(args.english_percent),
                       "--sample-version", str(args.sample_version), "--workers", str(args.workers),
                       "--trainer", args.trainer,
                       "--vocab", str(args.vocab), "--eval-chars", str(args.eval_chars), "--arms", arm]
            print(f"{stage}: {corpus} {arm}, log {log}", flush=True)
            with log.open("w") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                highwater["stages"][key]["process_group"] = process.pid

                def record_poll(rss_bytes, highwater_bytes):
                    highwater["stages"][key].update(rss_bytes=rss_bytes, highwater_bytes=highwater_bytes,
                                                      polled_unix=time.time())
                    write_json(highwater_path, highwater)

                try:
                    monitor_child(process, max_rss_bytes, record_poll)
                except BaseException as error:
                    entry.update(status="failed", finished_unix=time.time(), error=str(error))
                    write_json(status_path, status)
                    raise
            entry.update(status="completed", finished_unix=time.time())
            write_json(status_path, status)
    print(f"Completed: {out}", flush=True)


def main():
    args = parse_args()
    os.chdir(ROOT)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[key] = "2"
    corpus = f"fineweb_enko_{args.million_chars}m_en{args.english_percent}_local_v{args.sample_version}"
    base = ROOT / "results/corpora"
    out = ROOT / "results/boundary_bilingual" / f"{corpus}_v{args.vocab}"
    if args.trainer != "bpe":
        out = out.with_name(out.name + "_" + args.trainer)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    if args.child:
        child_main(args, corpus, base, out, arms)
    else:
        parent_main(args, corpus, out, arms)


if __name__ == "__main__":
    main()
