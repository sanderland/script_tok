from __future__ import annotations

from paper_utils.monolingual_compression_experiment import run_experiment


def main() -> None:
    results = run_experiment(
        corpus_names=["smol_eng_latn_300mb", "eng_latn_300mb"],
        pretokenizer_name="scriptenc_cb",
        additional_vocab_size=1024,
        init_algorithms=["simple_short"],
    )
    print(results)


if __name__ == "__main__":
    main()


