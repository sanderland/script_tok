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
    init_algorithm: str,
    initial_vocab_factor: int | None = None,
    max_token_len: int | None = None,
    pruning_shrinking_factor: float | None = None,
) -> dict[str, object]:
    from script_bpe.tokenizers.unigram.trainer import UnigramTrainer, UnigramTrainerConfig
    pretokenizer = get_pretokenizer(pretokenizer_name)
    corpus = load_corpus_by_name(corpus_name, pretokenizer)

    cfg_kwargs: dict[str, object] = {
        "additional_vocab_size": additional_vocab_size,
        "init_vocab_algo": init_algorithm,
    }
    if initial_vocab_factor is not None:
        cfg_kwargs["initial_vocab_factor"] = initial_vocab_factor
    if max_token_len is not None:
        cfg_kwargs["max_token_len"] = max_token_len
    if pruning_shrinking_factor is not None:
        cfg_kwargs["pruning_shrinking_factor"] = pruning_shrinking_factor
    cfg = UnigramTrainerConfig(**cfg_kwargs)
    trainer = UnigramTrainer(pretokenizer, corpus, cfg)

    t0 = time.perf_counter()
    model = trainer.train()
    train_seconds = time.perf_counter() - t0

    stats = model.metadata or {}
    
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
        "objective": stats.get("objective", 0.0),
    }


def run_experiment(
    *,
    corpus_names: list[str],
    pretokenizer_name: str,
    additional_vocab_size: int,
    init_algorithms: list[str],
    initial_vocab_factor: int | None = None,
    max_token_len: int | None = None,
    pruning_shrinking_factors: list[float] | None = None,
    use_cache: bool = True,
) -> list[dict[str, object]]:
    """Run training experiments serially with caching and return results."""
    logger = create_logger("experiment", verbose=True)
    results: list[dict[str, object]] = []

    if not pruning_shrinking_factors:
        pruning_shrinking_factors = [0.75]

    all_jobs = [
        (corpus_name, algo, sf)
        for corpus_name in corpus_names
        for algo in init_algorithms
        for sf in pruning_shrinking_factors
    ]

    for corpus_name, algo, shrink_factor in all_jobs:
        cache_dir = Path("results") / "mce" / corpus_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        experiment_parts = [algo, f"n{additional_vocab_size}", f"psf{shrink_factor}"]
        if initial_vocab_factor is not None:
            experiment_parts.append(f"f{initial_vocab_factor}")
        if max_token_len is not None:
            experiment_parts.append(f"mtl{max_token_len}")
        experiment_name = "_".join(experiment_parts)
        cache_file = cache_dir / f"{experiment_name}.json"


        if use_cache and cache_file.exists():
            logger.info(f"💾 Loading cached result for {corpus_name}:{algo} from {cache_file}")
            with open(cache_file, "r") as f:
                result = json.load(f)
            results.append(result)
            continue

        logger.info(
            f"▶️  Starting {algo} on {corpus_name} (n={additional_vocab_size}, psf={shrink_factor})"
        )
        
        try:
            result = _train_and_evaluate(
                corpus_name=corpus_name,
                pretokenizer_name=pretokenizer_name,
                additional_vocab_size=additional_vocab_size,
                init_algorithm=algo,
                initial_vocab_factor=initial_vocab_factor,
                max_token_len=max_token_len,
                pruning_shrinking_factor=shrink_factor,
            )
            result["algo"] = algo
            result["pruning_shrinking_factor"] = shrink_factor
            results.append(result)

            if use_cache:
                with open(cache_file, "w") as f:
                    # A custom serializer could handle numpy types, but for now we convert manually
                    saveable_result = {k: float(v) if hasattr(v, 'item') else v for k, v in result.items()}
                    json.dump(saveable_result, f, indent=2)
                logger.info(f"📄 Saved result to {cache_file}")

        except Exception as e:
            logger.error(f"❌ Job failed for {corpus_name}:{algo}", exc_info=True)
            raise RuntimeError(f"Job failed for {corpus_name}:{algo}") from e

    results.sort(key=lambda r: (str(r.get("corpus", "")), float(r.get("objective", float("inf")))))

    return results


if __name__ == "__main__":
    # Base algorithms only (no alias modifiers); we will sweep pruning_shrinking_factor
    init_alg = ["corpus_repair"]

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
        init_algorithms=init_alg,
        pruning_shrinking_factors=[0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
    )
    print(tabulate(results, headers="keys", tablefmt="grid"))
