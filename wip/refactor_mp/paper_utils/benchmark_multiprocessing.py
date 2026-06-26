#!/usr/bin/env python3
"""Benchmark Unigram training with different numbers of workers on eng_latn_300mb."""

import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CORPUS_NAME = "eng_latn_300mb"
PRETOKENIZER = "scriptenc_cb"
VOCAB_SIZE = 16384
WORKER_COUNTS = [1, 2, 4, 8, 16, 32]
OUTPUT_LOG = Path(__file__).parent / "benchmark_multiprocessing.log"


def run_training(num_workers: int) -> dict:
    """Run training with specified number of workers and return result dict."""
    cmd = [
        "uv", "run", "script_bpe/train.py",
        "--model", "unigram",
        "--corpus", CORPUS_NAME,
        "--pretokenizer", PRETOKENIZER,
        "-n", str(VOCAB_SIZE),
        "--parallel", str(num_workers),
        "--retrain",
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        return {
            "num_workers": num_workers,
            "elapsed_seconds": elapsed_time,
            "success": True,
            "output": result.stdout,
        }
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        return {
            "num_workers": num_workers,
            "elapsed_seconds": elapsed_time,
            "success": False,
            "output": e.stdout + "\n" + e.stderr,
        }


def main():
    print(f"Multiprocessing Benchmark for Unigram Training")
    print(f"Corpus: {CORPUS_NAME}")
    print(f"Pretokenizer: {PRETOKENIZER}")
    print(f"Vocab size: {VOCAB_SIZE:,}")
    print(f"Worker counts to test: {WORKER_COUNTS}")
    print(f"Output log: {OUTPUT_LOG}")
    print(f"\nRunning all {len(WORKER_COUNTS)} benchmarks in parallel...")
    print(f"Results will be printed as they complete.\n")
    
    results = {}
    
    with open(OUTPUT_LOG, 'w') as log_file:
        log_file.write(f"Multiprocessing Benchmark for Unigram Training\n")
        log_file.write(f"Corpus: {CORPUS_NAME}\n")
        log_file.write(f"Pretokenizer: {PRETOKENIZER}\n")
        log_file.write(f"Vocab size: {VOCAB_SIZE:,}\n")
        log_file.write(f"Worker counts: {WORKER_COUNTS}\n")
        log_file.write(f"{'='*80}\n\n")
        
        # Run all benchmarks in parallel
        with ProcessPoolExecutor(max_workers=len(WORKER_COUNTS)) as executor:
            futures = {executor.submit(run_training, num_workers): num_workers 
                      for num_workers in WORKER_COUNTS}
            
            for future in as_completed(futures):
                result = future.result()
                results[result["num_workers"]] = result
                
                # Print immediately as each completes
                status = "✓" if result["success"] else "✗"
                print(f"{status} Workers: {result['num_workers']:<3} | Time: {result['elapsed_seconds']:>8.2f}s")
                
                log_file.write(f"Workers: {result['num_workers']:<3} | Time: {result['elapsed_seconds']:>8.2f}s | Status: {status}\n")
                log_file.flush()
        
        # Print summary table (sorted by worker count)
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS (sorted by worker count)")
        print(f"{'='*80}")
        
        log_file.write(f"\n{'='*80}\n")
        log_file.write("BENCHMARK RESULTS (sorted by worker count)\n")
        log_file.write(f"{'='*80}\n")
        
        header = f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12} {'Status':<10}"
        separator = f"{'-'*80}"
        
        print(header)
        print(separator)
        log_file.write(header + "\n")
        log_file.write(separator + "\n")
        
        # Sort results by worker count for display
        sorted_results = [results[w] for w in sorted(results.keys())]
        baseline_time = sorted_results[0]["elapsed_seconds"] if sorted_results and sorted_results[0]["success"] else 0
        
        for stats in sorted_results:
            if baseline_time > 0 and stats["success"]:
                speedup = baseline_time / stats["elapsed_seconds"]
                efficiency = speedup / stats["num_workers"] * 100
            else:
                speedup = 0
                efficiency = 0
            
            status = "✓" if stats["success"] else "✗"
            line = (
                f"{stats['num_workers']:<10} "
                f"{stats['elapsed_seconds']:<12.2f} "
                f"{speedup:<10.2f}x "
                f"{efficiency:<12.1f}% "
                f"{status:<10}"
            )
            print(line)
            log_file.write(line + "\n")
        
        print(separator)
        log_file.write(separator + "\n")
        
    print(f"\nResults saved to: {OUTPUT_LOG}")


if __name__ == '__main__':
    main()
