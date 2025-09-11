from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter

from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.pretokenize.pretokenizer import (
    UTF8Pretokenizer, UTF8PretokenizerConfig,
    ScriptPretokenizer, ScriptPretokenizerConfig,
)
from script_bpe.tokenizers.unigram.init_algorithms import (
    compute_substring_frequencies_simple,
    compute_substring_frequencies_spm,
)
from script_bpe.tokenizers.unigram.init_corpus import (
    compute_substring_frequencies_corpus,
)


def build_candidate_counts(
    algo: str,
    pretokenizer,
    corpus,
    max_token_len: int,
) -> Counter[tuple[int, ...]]:
    if algo == "simple":
        return compute_substring_frequencies_simple(pretokenizer, corpus, max_token_len)
    elif algo == "spm":
        raw = compute_substring_frequencies_spm(pretokenizer, corpus, max_token_len, repair=False)
        # Ensure only valid tokens are counted for fair comparison
        return Counter({tok: cnt for tok, cnt in raw.items() if pretokenizer.token_allowed(tok)})
    elif algo == "spm_repair":
        raw = compute_substring_frequencies_spm(pretokenizer, corpus, max_token_len, repair=True)
        return Counter({tok: cnt for tok, cnt in raw.items() if pretokenizer.token_allowed(tok)})
    elif algo in ("corpus", "corpus_intermediate"):
        # Convert to flat corpus of (tuple[int,...], int)
        flat_corpus = [(tuple(seq), freq) for seq, freq in corpus]
        freq = compute_substring_frequencies_corpus(
            flat_corpus,
            max_token_len,
            intermediate_patterns=(algo == "corpus_intermediate"),
        )
        # Filter to only allow tokens permitted by the pretokenizer
        return Counter({tok: cnt for tok, cnt in freq.items() if pretokenizer.token_allowed(tok)})
    else:
        raise ValueError(f"Unknown algo: {algo}")


def jaccard(a: set[tuple[int, ...]], b: set[tuple[int, ...]]) -> float:
    if not a and not b:
        return 100.0
    inter = len(a & b)
    union = len(a | b)
    return 100.0 * inter / union if union else 0.0


def human_bytes(n_bytes: int) -> str:
    mib = n_bytes / (1024 * 1024)
    return f"{mib:.2f} MiB"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare unigram init-vocab strategies: counts, Jaccard, time/mem")
    p.add_argument("--corpus", default="swift", help="Corpus name from registry (default: swift)")
    p.add_argument("--max-token-len", type=int, default=16, dest="max_token_len")
    p.add_argument("--pretokenizer", choices=["script", "utf8"], default="script")
    p.add_argument(
        "--algos",
        nargs="*",
        default=["simple", "spm", "spm_repair", "corpus", "corpus_intermediate"],
        help="Subset of algos to run",
    )
    args = p.parse_args()

    if args.pretokenizer == "script":
        pt = ScriptPretokenizer(ScriptPretokenizerConfig())
    else:
        pt = UTF8Pretokenizer(UTF8PretokenizerConfig())

    corpus = load_corpus_by_name(args.corpus, pretokenizer=pt)

    algo_to_set: dict[str, set[tuple[int, ...]]] = {}
    timings: dict[str, float] = {}
    mem_peaks: dict[str, int] = {}

    print(f"Corpus: {args.corpus}")
    print(f"Pretokenizer: {pt.__class__.__name__} ({pt.hash()})")
    print(f"Total pretokens: {sum(freq for _, freq in corpus):,}")

    for algo in args.algos:
        tracemalloc.start()
        t0 = time.perf_counter()
        freq_counter = build_candidate_counts(algo, pt, corpus, args.max_token_len)
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        algo_to_set[algo] = set(freq_counter.keys())
        timings[algo] = elapsed
        mem_peaks[algo] = peak

    # Sizes and timing/mem
    print("\nPer-strategy stats:")
    print("algo               count        time (s)    peak mem")
    for algo in args.algos:
        count = len(algo_to_set[algo])
        print(f"{algo:<18} {count:10,d}   {timings[algo]:10.3f}   {human_bytes(mem_peaks[algo])}")

    # Jaccard matrix
    print("\nJaccard overlap (%), candidate sets:")
    header = ["algo"] + args.algos
    print(" ".join(f"{h:>16}" for h in header))
    for a in args.algos:
        row = [f"{a:>16}"]
        for b in args.algos:
            val = jaccard(algo_to_set[a], algo_to_set[b])
            row.append(f"{val:16.1f}")
        print(" ".join(row))


if __name__ == "__main__":
    main()


