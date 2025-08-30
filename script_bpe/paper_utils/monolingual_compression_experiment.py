from __future__ import annotations

import os
import time
from typing import Iterable

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
from script_bpe.utils import create_logger


def run_experiment(
    *,
    corpus_names: list[str],
    pretokenizer_name: str,
    additional_vocab_size: int,
    init_algorithms: list[str],
    parallel_workers: int = 4,
) -> list[dict[str, object]]:
    logger = create_logger("experiment", verbose=True)
    results: list[dict[str, object]] = []
    pretokenizer = get_pretokenizer(pretokenizer_name)

    for corpus_name in corpus_names:
        corpus = load_corpus_by_name(corpus_name, pretokenizer)
        for algo in init_algorithms:
            cfg = UnigramTrainerConfig(
                additional_vocab_size=additional_vocab_size,
                num_workers=parallel_workers,
                init_vocab_algo=algo,
            )
            trainer = UnigramTrainer(pretokenizer, corpus, cfg)
            logger.info(f"▶️ Training {algo} on {corpus_name} with n={additional_vocab_size}")
            t0 = time.perf_counter()
            model = trainer.train()
            train_seconds = time.perf_counter() - t0

            # Compute compression metrics over the corpus
            total_bytes = 0
            total_tokens = 0
            total_chars = 0
            for token_seq, freq in corpus:
                encoded = model.encode_atomic(token_seq)
                total_tokens += len(encoded) * freq
                total_chars += sum(len(str(x)) for x in token_seq) * freq
                # bytes proxy: sum UTF-8 lengths of decoded string
                decoded_text = pretokenizer.decode(token_seq)
                total_bytes += len(decoded_text.encode("utf-8")) * freq
            bytes_per_token = total_bytes / total_tokens if total_tokens else 0.0

            result = {
                "corpus": corpus_name,
                "algo": algo,
                "n": additional_vocab_size,
                "seconds": train_seconds,
                "tokens": total_tokens,
                "bytes": total_bytes,
                "bytes_per_token": bytes_per_token,
            }
            logger.info(
                f"✅ {algo} on {corpus_name}: {train_seconds:.2f}s, bytes/token={bytes_per_token:.3f}"
            )
            results.append(result)
    return results


if __name__ == "__main__":
    # Example: compare en/zh/ko with all three algorithms
    results = run_experiment(
        corpus_names=["eng_latn_300mb", "zho_hans_300mb", "kor_hang_300mb"],
        pretokenizer_name="scriptenc_cb",
        additional_vocab_size=1000,
        init_algorithms=["simple", "spm", "spm_repair"],
    )
    for r in results:
        print(r)


