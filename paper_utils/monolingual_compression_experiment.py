from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from queue import Empty, Queue
from threading import Thread

from tabulate import tabulate

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from script_bpe.utils import create_logger


def _train_and_evaluate(
    *,
    corpus_name: str,
    pretokenizer_name: str,
    additional_vocab_size: int,
    init_algorithm: str,
    parallel_workers: int,
) -> dict[str, object]:
    # Construct resources in the worker to avoid pickling issues
    pretokenizer = get_pretokenizer(pretokenizer_name)
    corpus = load_corpus_by_name(corpus_name, pretokenizer)

    cfg = UnigramTrainerConfig(
        additional_vocab_size=additional_vocab_size,
        num_workers=parallel_workers,
        init_vocab_algo=init_algorithm,
    )
    trainer = UnigramTrainer(pretokenizer, corpus, cfg)

    t0 = time.perf_counter()
    model = trainer.train()
    train_seconds = time.perf_counter() - t0

    # Use stats directly from trainer metadata
    stats = model.metadata or {}
    
    # Calculate actual UTF-8 bytes for the corpus
    total_bytes = 0
    total_chars = 0
    for token_seq, freq in corpus:
        decoded_text = pretokenizer.decode(token_seq)
        total_bytes += len(decoded_text.encode("utf-8")) * freq
        total_chars += len(decoded_text) * freq

    total_tokens = stats.get("total_tokens", 0)
    
    return {
        "corpus": corpus_name,
        "algo": init_algorithm,
        "n": additional_vocab_size,
        "seconds": train_seconds,
        "tokens": total_tokens,
        "bytes": total_bytes,
        "chars": total_chars,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "tokens_per_char": stats.get("tokens/pretoken", 0.0),
        "bytes_per_char": total_bytes / total_chars if total_chars else 0.0,
        "initial_vocab_size": stats.get("initial_vocab_size", 0),
    }


def _heartbeat_worker(event_queue: Queue, start_time: float, logger, all_jobs: list[tuple[str, str]]) -> None:
    """Background thread that prints results as they complete and periodic status updates."""
    finished_results: dict[tuple[str, str], dict] = {}  # (corpus, algo) -> result
    last_report = start_time
    report_interval = 600.0  # 10 minutes

    while True:
        # Calculate timeout until next heartbeat
        now = time.perf_counter()
        time_since_report = now - last_report
        timeout = max(0.1, report_interval - time_since_report)
        
        try:
            event = event_queue.get(timeout=timeout)
        except Empty:
            # Time for periodic heartbeat - show comprehensive table
            elapsed_min = (now - start_time) / 60
            
            # Build table with all jobs (finished + pending)
            table_rows = []
            for corpus, algo in all_jobs:
                if (corpus, algo) in finished_results:
                    # Finished job - show all stats
                    table_rows.append(finished_results[(corpus, algo)])
                else:
                    # Pending job - show empty row with just corpus/algo
                    table_rows.append({
                        "corpus": corpus,
                        "algo": algo,
                    })
            
            table = tabulate(table_rows, headers="keys", tablefmt="grid")
            finished_count = len(finished_results)
            total_count = len(all_jobs)
            logger.info(f"⏱️ {elapsed_min:.1f}m elapsed — {finished_count}/{total_count} jobs complete:\n{table}")
            last_report = now
            continue

        if event is None:  # Stop signal
            break
            
        # Handle completed result
        result = event["result"]
        corpus, algo = result["corpus"], result["algo"]
        seconds = result["seconds"]
        finished_results[(corpus, algo)] = result
        
        # Print immediate completion notice
        logger.info(f"✅ {corpus}:{algo} completed in {seconds:.1f}s")

    # Final status
    elapsed_min = (time.perf_counter() - start_time) / 60
    logger.info(f"🏁 Heartbeat finished after {elapsed_min:.1f}m")


def run_experiment(
    *,
    corpus_names: list[str],
    pretokenizer_name: str,
    additional_vocab_size: int,
    init_algorithms: list[str],
    parallel_workers: int = 4,
) -> list[dict[str, object]]:
    """Run parallel training experiments and return results with live progress updates."""
    logger = create_logger("experiment", verbose=True)
    results: list[dict[str, object]] = []
    start_time = time.perf_counter()

    # Prepare list of all jobs for heartbeat tracking
    all_jobs = [(corpus_name, algo) for corpus_name in corpus_names for algo in init_algorithms]
    
    # Start heartbeat thread
    event_queue: Queue = Queue()
    heartbeat_thread = Thread(
        target=_heartbeat_worker,
        args=(event_queue, start_time, logger, all_jobs),
        daemon=True,
    )
    heartbeat_thread.start()

    # Submit all jobs to process pool
    max_workers = len(init_algorithms) * len(corpus_names)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for corpus_name in corpus_names:
            for algo in init_algorithms:
                logger.info(f"▶️ Starting {algo} on {corpus_name} (n={additional_vocab_size})")
                future = executor.submit(
                    _train_and_evaluate,
                    corpus_name=corpus_name,
                    pretokenizer_name=pretokenizer_name,
                    additional_vocab_size=additional_vocab_size,
                    init_algorithm=algo,
                    parallel_workers=parallel_workers,
                )
                futures[future] = (corpus_name, algo)

        # Collect results as they complete
        while futures:
            done, _ = wait(futures.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                results.append(result)
                event_queue.put({"result": result})

    # Clean shutdown
    event_queue.put(None)
    heartbeat_thread.join(timeout=2.0)
    
    return results


if __name__ == "__main__":
    # Example: compare en/zh/ko with all three algorithms
    results = run_experiment(
        corpus_names=["smol_eng_latn_300mb"], #, "zho_hans_300mb", "kor_hang_300mb"],
        pretokenizer_name="scriptenc_cb",
        additional_vocab_size=8192,
       init_algorithms=["corpus", "corpus_intermediate", "simple"],
       #init_algorithms=["spm"],
    )
    # tabulate results
    print(tabulate(results, headers="keys", tablefmt="grid"))
