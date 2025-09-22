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
    use_cache: bool = True,
) -> list[dict[str, object]]:
    """Run training experiments serially with caching and return results."""
    logger = create_logger("experiment", verbose=True)
    results: list[dict[str, object]] = []

    all_jobs = [(corpus_name, algo) for corpus_name in corpus_names for algo in init_algorithms]

    def normalize_algo(algo: str) -> tuple[str, dict[str, int]]:
        base = algo
        overrides: dict[str, int] = {}
        if algo.endswith("_many"):
            base = algo.removesuffix("_many")
            overrides = {"initial_vocab_factor": 40}
        elif algo.endswith("_few"):
            base = algo.removesuffix("_few")
            overrides = {"initial_vocab_factor": 2}
        elif algo.endswith("_short"):
            base = algo.removesuffix("_short")
            overrides = {"max_token_len": 8 if base == "simple" else 16}
        return base, overrides

    for corpus_name, alias_algo in all_jobs:
        cache_dir = Path("results") / "mce" / corpus_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        base_algo, alias_overrides = normalize_algo(alias_algo)
        eff_initial_vocab_factor = (
            alias_overrides.get("initial_vocab_factor", initial_vocab_factor)
            if initial_vocab_factor is not None or "initial_vocab_factor" in alias_overrides
            else None
        )
        eff_max_token_len = (
            alias_overrides.get("max_token_len", max_token_len)
            if max_token_len is not None or "max_token_len" in alias_overrides
            else None
        )

        experiment_parts = [base_algo, f"n{additional_vocab_size}"]
        if eff_initial_vocab_factor is not None:
            experiment_parts.append(f"f{eff_initial_vocab_factor}")
        if eff_max_token_len is not None:
            experiment_parts.append(f"mtl{eff_max_token_len}")
        experiment_name = "_".join(experiment_parts)
        cache_file = cache_dir / f"{experiment_name}.json"


        if use_cache and cache_file.exists():
            logger.info(f"💾 Loading cached result for {corpus_name}:{alias_algo} from {cache_file}")
            with open(cache_file, "r") as f:
                result = json.load(f)
            results.append(result)
            continue

        logger.info(f"▶️  Starting {alias_algo} (base={base_algo}) on {corpus_name} (n={additional_vocab_size})")
        
        try:
            result = _train_and_evaluate(
                corpus_name=corpus_name,
                pretokenizer_name=pretokenizer_name,
                additional_vocab_size=additional_vocab_size,
                init_algorithm=base_algo,
                initial_vocab_factor=eff_initial_vocab_factor,
                max_token_len=eff_max_token_len,
            )
            result["algo"] = alias_algo
            results.append(result)

            if use_cache:
                with open(cache_file, "w") as f:
                    # A custom serializer could handle numpy types, but for now we convert manually
                    saveable_result = {k: float(v) if hasattr(v, 'item') else v for k, v in result.items()}
                    json.dump(saveable_result, f, indent=2)
                logger.info(f"📄 Saved result to {cache_file}")

        except Exception as e:
            logger.error(f"❌ Job failed for {corpus_name}:{alias_algo}", exc_info=True)
            raise RuntimeError(f"Job failed for {corpus_name}:{alias_algo}") from e

    results.sort(key=lambda r: (str(r.get("corpus", "")), float(r.get("objective", float("inf")))))

    return results


if __name__ == "__main__":
    init_alg = ["corpus_repair", "corpus_repair_many", "corpus_repair_short", "corpus_repair_few","corpus_long", "corpus_intermediate", "simple", "simple_many", "simple_short", "corpus_intermediate_many", "corpus_intermediate_short", "simple_few", "corpus_intermediate_few"]
    results = run_experiment(
        corpus_names=[
        #    "smol_eng_latn_300mb",
       #     "eng_latn_300mb",
       #     "deu_latn_300mb",
      #      "arb_arab_300mb",
      #      "hin_deva_300mb",
     #       "zho_hans_300mb",
            "kor_hang_300mb",
        ],
        pretokenizer_name="scriptenc_cb",
        additional_vocab_size=16384,
        init_algorithms=init_alg[::-1],
    )
    print(tabulate(results, headers="keys", tablefmt="grid"))
