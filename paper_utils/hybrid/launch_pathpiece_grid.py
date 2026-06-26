"""Parallel launcher: train all (lang x init) PathPiece tokenizers concurrently.

Each (lang, init) pair is run in its own subprocess so the BPE-init
worker pool is local to that pair. Logs are streamed to per-pair files
under ``results/pathpiece/logs/`` and a short status line is emitted on
stdout for each job as it finishes.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

LANGS = ["eng", "deu", "fin", "rus", "arb", "kor"]
INITS = ["ngram", "bpe"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGS, choices=LANGS)
    parser.add_argument("--inits", nargs="+", default=INITS, choices=INITS)
    parser.add_argument("--init-vocab-size", type=int, default=131072)
    parser.add_argument("--prune-batch-fraction", type=float, default=0.20)
    parser.add_argument("--max-token-width", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="BPE-init worker pool size per (lang, init) job.")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    log_dir = Path("results/pathpiece/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[str, str, subprocess.Popen, Path]] = []
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for lang in args.langs:
        for init in args.inits:
            log_path = log_dir / f"{lang}_{init}.log"
            cmd = [
                sys.executable, "-m", "paper_utils.hybrid.run_pathpiece_grid",
                "--langs", lang,
                "--inits", init,
                "--init-vocab-size", str(args.init_vocab_size),
                "--prune-batch-fraction", str(args.prune_batch_fraction),
                "--max-token-width", str(args.max_token_width),
                "--num-workers", str(args.num_workers),
            ]
            if args.retrain:
                cmd.append("--retrain")
            log_fh = open(log_path, "wb")
            print(f"[launch] {lang}/{init} -> {log_path}", flush=True)
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            procs.append((lang, init, proc, log_path))

    t0 = time.perf_counter()
    failures: list[tuple[str, str, int]] = []
    while procs:
        time.sleep(20)
        still_running = []
        for lang, init, proc, log_path in procs:
            rc = proc.poll()
            if rc is None:
                still_running.append((lang, init, proc, log_path))
                continue
            elapsed = time.perf_counter() - t0
            tag = "OK" if rc == 0 else f"FAIL({rc})"
            print(f"[{elapsed/60:6.1f}m] {tag} {lang}/{init}  log={log_path}", flush=True)
            if rc != 0:
                failures.append((lang, init, rc))
        procs = still_running

    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print(f"All {len(args.langs) * len(args.inits)} runs complete in {(time.perf_counter()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
