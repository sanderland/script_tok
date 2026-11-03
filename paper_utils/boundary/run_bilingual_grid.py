"""Sequential full bilingual grid, reusing completed runs and shared corpus caches."""

import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("plain", "bnd_wpd", "bnd_wpd_caps", "bnd_w", "bnd_w_caps", "bnd_wp", "bnd_wp_caps")
CORPUS = "fineweb_enko_2000m_en50_local_v2"


def completed(out, arm, trainer, sample, vocab=34685, corpus=CORPUS):
    try:
        info = json.loads((out / f"{arm}.train.json").read_text())
        analysis = json.loads((out / f"{arm}.analysis.json").read_text())
        return (info["trainer"] == trainer and info["sample"] == sample
                and info["total_vocab"] == vocab
                and hashlib.sha256((out / f"{arm}.json.gz").read_bytes()).hexdigest() == info["tokenizer_sha256"]
                and analysis["corpus"] == corpus
                and all(analysis["evaluation"][lg]["characters"] == 500_000 for lg in ("en", "ko")))
    except (FileNotFoundError, KeyError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--vocab", type=int, default=34685)
    budget.add_argument("--learned-vocab", type=int)
    parser.add_argument("--trainers", default="bpe,mingram")
    parser.add_argument("--million-chars", type=int, default=2000)
    parser.add_argument("--english-percent", type=int, default=50)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    trainers = args.trainers.split(",")
    if not trainers or any(t not in ("bpe", "mingram") for t in trainers):
        parser.error("trainers must be bpe, mingram, or bpe,mingram")
    if args.vocab < 1 or (args.learned_vocab is not None and args.learned_vocab < 1):
        parser.error("vocabulary budgets must be positive")
    arms = args.arms.split(",")
    if not arms or any(arm not in ARMS for arm in arms):
        parser.error("unknown ablation arm")
    if args.million_chars < 1 or not 1 <= args.english_percent <= 99:
        parser.error("require positive million-chars and english-percent between 1 and 99")
    os.chdir(ROOT)
    corpus = f"fineweb_enko_{args.million_chars}m_en{args.english_percent}_local_v2"
    mixture_args = ["--million-chars", str(args.million_chars), "--english-percent", str(args.english_percent), "--sample-version", "2"]
    subprocess.run([sys.executable, "-m", "paper_utils.boundary.run_bilingual",
                    *mixture_args, "--stage", "prepare", "--max-rss-gib", "20"], check=True)
    sample = json.loads((ROOT / f"results/corpora/_bilingual_text/{corpus}/manifest.json").read_text())
    for arm in arms:
        vocab = args.vocab
        if args.learned_vocab is not None:
            from paper_utils.boundary.downstream.train_matched import make_pretokenizer

            vocab = args.learned_vocab + len(make_pretokenizer(arm).atomic_tokens)
        for trainer in trainers:
            suffix = "" if trainer == "bpe" else "_mingram"
            out = ROOT / f"results/boundary_bilingual/{corpus}_v{vocab}{suffix}"
            if completed(out, arm, trainer, sample, vocab, corpus):
                print(f"Reuse completed {trainer} {arm}", flush=True)
                continue
            print(f"Starting {trainer} {arm}, total vocabulary {vocab}, learned budget {args.learned_vocab}", flush=True)
            subprocess.run([sys.executable, "-m", "paper_utils.boundary.run_bilingual",
                            *mixture_args,
                            "--vocab", str(vocab), "--eval-chars", "500000",
                            "--trainer", trainer, "--arms", arm, "--workers", "4", "--max-rss-gib", "20"], check=True)
            report_budget = ["--learned-vocab", str(args.learned_vocab)] if args.learned_vocab is not None else ["--vocab", str(args.vocab)]
            subprocess.run([sys.executable, "-m", "paper_utils.boundary.summarize_bilingual", *report_budget], stdout=subprocess.DEVNULL, check=True)
    print(f"Completed all {len(arms) * len(trainers)} bilingual cells", flush=True)


if __name__ == "__main__":
    main()
