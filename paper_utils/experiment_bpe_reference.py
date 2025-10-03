from __future__ import annotations
import json
import time
from pathlib import Path

from tabulate import tabulate

from script_bpe import get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.utils import create_logger


def _train_and_evaluate(
    *,
    corpus_name: str,
    pretokenizer_name: str,
    additional_vocab_size: int,
) -> dict[str, object]:
    from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig

    pretokenizer = get_pretokenizer(pretokenizer_name)
    corpus = load_corpus_by_name(corpus_name, pretokenizer)

    # Fixed workers (speed only, not a parameter to sweep)
    cfg = BPETrainerConfig(additional_vocab_size=additional_vocab_size, num_workers=4)
    trainer = BPETrainer(pretokenizer, corpus, cfg)

    t0 = time.perf_counter()
    tokenizer = trainer.train()
    train_seconds = time.perf_counter() - t0

    # Aggregate corpus-level performance using the trained tokenizer
    perf = tokenizer.corpus_performance(corpus)

    # Compute byte and char totals for comparability with unigram experiment outputs
    total_bytes = 0
    total_chars = 0
    for token_seq, freq in corpus:
        decoded_text = pretokenizer.decode(token_seq)
        total_bytes += len(decoded_text.encode("utf-8")) * freq
        total_chars += len(decoded_text) * freq

    total_tokens = int(perf.get("total_tokens_len", 0))

    return {
        "corpus": corpus_name,
        "algo": "bpe",
        "n": additional_vocab_size,
        "seconds": train_seconds,
        "tokens": total_tokens,
        "bytes": total_bytes,
        "chars": total_chars,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "chars_per_token": perf.get("chars_per_token", 0.0),
        "tokens_per_char": perf.get("tokens_per_char", 0.0),
        "bytes_per_char": total_bytes / total_chars if total_chars else 0.0,
        "initial_vocab_size": len(pretokenizer.atomic_tokens),
        "num_merge_rules": len(getattr(tokenizer, "merge_rules", []) or []),
        # Keep an objective key for table compatibility with unigram scripts
        "objective": 0.0,
    }


def run_experiment(
    *,
    corpus_names: list[str],
    pretokenizer_name: str,
    additional_vocab_size: int,
    use_cache: bool = True,
) -> list[dict[str, object]]:
    """Train reference BPE tokenizers and return results (with caching).

    Mirrors the structure of the unigram experiment scripts but without unigram-specific knobs.
    """
    logger = create_logger("experiment", verbose=True)
    results: list[dict[str, object]] = []

    for corpus_name in corpus_names:
        cache_dir = Path("results") / "bpe_ref" / corpus_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        experiment_name = f"n{additional_vocab_size}"
        cache_file = cache_dir / f"{experiment_name}.json"

        if use_cache and cache_file.exists():
            logger.info(f"💾 Loading cached result for {corpus_name} from {cache_file}")
            with open(cache_file, "r") as f:
                result = json.load(f)
            results.append(result)
            continue

        logger.info(f"▶️  Starting BPE on {corpus_name} (n={additional_vocab_size})")

        try:
            result = _train_and_evaluate(
                corpus_name=corpus_name,
                pretokenizer_name=pretokenizer_name,
                additional_vocab_size=additional_vocab_size,
            )
            results.append(result)

            if use_cache:
                with open(cache_file, "w") as f:
                    saveable_result = {k: float(v) if hasattr(v, 'item') else v for k, v in result.items()}
                    json.dump(saveable_result, f, indent=2)
                logger.info(f"📄 Saved result to {cache_file}")

        except Exception as e:
            logger.error(f"❌ Job failed for {corpus_name}", exc_info=True)
            raise RuntimeError(f"Job failed for {corpus_name}") from e

    results.sort(key=lambda r: (str(r.get("corpus", "")), float(r.get("tokens_per_char", float("inf")))))

    return results


if __name__ == "__main__":
    results = run_experiment(
        corpus_names=[
            "zho_hans_300mb",
            "eng_latn_300mb",
            "deu_latn_300mb",
            "arb_arab_300mb",
            "hin_deva_300mb",
            "kor_hang_300mb",
        ],
        pretokenizer_name="scriptenc2_cbi",
        additional_vocab_size=16384,
    )
    print(tabulate(results, headers="keys", tablefmt="grid"))


