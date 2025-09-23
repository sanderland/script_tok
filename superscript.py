from script_bpe.corpus import load_corpus_by_name
from script_bpe.pretokenize import PRETOKENIZER_REGISTRY, get_pretokenizer
from script_bpe.pretokenize.pretokenizer import ScriptPretokenizerConfig
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV2
from script_bpe.tokenizers.unigram.trainer import (
    UnigramTrainer,
    UnigramTrainerConfig,
)
from script_bpe.train import train_tokenizer
from script_bpe.utils import create_logger
from multiprocessing import freeze_support
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
pretokenizer_name = "scriptenc2_cbi"
corpus_name = "eng_latn_300mb"
#corpus_name = "smol_eng_latn_300mb"
retrain = False
n_cpus = 4
# Analysis config
NGRAM_SIZES = [1, 2, 3, 4]
TOP_N = 250

# This pattern will be used by our custom pretokenizer.
line_regex = r"[\r\n]*[^\r\n]*"

# Register the custom pretokenizer so get_pretokenizer can find it

PRETOKENIZER_REGISTRY["scriptenc_line_regex"] = ScriptPretokenizerConfig(
    regex_pattern=line_regex,
    script_split=False,
    script_config=ScriptEncodingV2,
)

trainer_config_kwargs = {
    "additional_vocab_size": 100_000,
    "init_vocab_algo": "corpus_repair",
    "initial_vocab_factor": 10,
}
# --- End Configuration ---

if __name__ == '__main__':
    freeze_support()

    logger = create_logger("superscript", verbose=True)

    logger.info(f"Pretokenizer: {pretokenizer_name}, Corpus: {corpus_name}")
    logger.info(f"Trainer Config: {trainer_config_kwargs}")

    model = train_tokenizer(
        pretokenizer_name=pretokenizer_name,
        model_name="unigram",
        corpus_name=corpus_name,
        additional_vocab_size=trainer_config_kwargs["additional_vocab_size"],
        n_cpus=n_cpus,
        retrain=retrain,
        report=False,
        trainer_config_kwargs=trainer_config_kwargs,
    )

    assert model, "Could not train or load the initial 100k model."

    logger.info("Run complete.")
    logger.info(f"Final model stats: {model.metadata}")


    # --- Stage 2: Supertoken Analysis ---
    logger.info("--- Starting Stage 2: Supertoken Analysis ---")

    # 1. Load the corpus with the line-based pretokenizer.
    logger.info("Loading corpus with line-based pretokenizer...")
    analysis_pretokenizer = get_pretokenizer("scriptenc_line_regex")
    corpus = load_corpus_by_name(corpus_name, analysis_pretokenizer)

    # 2. Initialize counters for n-grams.
    ngram_counters = {n: Counter() for n in NGRAM_SIZES}

    # The following analysis is memory-intensive.
    logger.info("Tokenizing corpus and counting n-grams...")
    for pretoken_seq, freq in corpus:
        # The corpus is pre-tokenized line-by-line; now tokenize with the 100k model.
        lattice = model.make_lattice(pretoken_seq)
        viterbi_path, _ = lattice.viterbi()
        token_ids = [token.id for token in viterbi_path]

        # Extract n-grams for the configured sizes.
        for n in NGRAM_SIZES:
            if len(token_ids) >= n:
                for i in range(len(token_ids) - n + 1):
                    ngram = tuple(token_ids[i : i + n])
                    ngram_counters[n][ngram] += freq

    # 3. Report the top N n-grams for each size.
    logger.info(f"--- Top {TOP_N} N-grams ---")
    unigram_counts = ngram_counters.get(1, Counter())
    for n in NGRAM_SIZES:
        print(f"\n--- Top {TOP_N} {n}-grams ---")
        top_ngrams = ngram_counters[n].most_common(TOP_N)
        for ngram_ids, count in top_ngrams:
            # Decode the n-gram for readability.
            decoded_tokens = []
            for token_id in ngram_ids:
                token = model.tokens_by_id[token_id]
                decoded_tokens.append(model.pretokenizer.decode(token.atomic_tokens))
            readable_str = "".join(decoded_tokens)

            token_counts_str = ""
            if n > 1:
                token_counts_parts = []
                for i, token_id in enumerate(ngram_ids):
                    individual_count = unigram_counts.get((token_id,), 0)
                    token_str = decoded_tokens[i]
                    token_counts_parts.append(f"'{token_str}': {individual_count:,}")
                token_counts_str = "| Token counts: " + " ".join(token_counts_parts)

            print(f"Count: {count:<10,} | IDs: {ngram_ids!s:<25} | Decoded: '{readable_str}' {token_counts_str}")

    # 4. Output a combined histogram of counts for all n-gram sizes.
    logger.info("--- N-gram Count Histograms ---")

    all_counts = [np.array(list(ngram_counters[n].values())) for n in NGRAM_SIZES]
    all_counts = [counts for counts in all_counts if len(counts) > 0]

    if not all_counts:
        logger.warning("No n-grams found to generate a histogram.")
    else:
        plt.figure(figsize=(12, 8))
        
        # Determine the global min and max counts to create shared bins
        min_count = 1
        max_count = max(counts.max() for counts in all_counts)
        
        # Create logarithmic bins with 3 bins per order of magnitude
        num_orders_of_magnitude = int(np.ceil(np.log10(max_count)))
        num_bins = num_orders_of_magnitude * 3
        bins = np.logspace(np.log10(min_count), np.log10(max_count), num_bins)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red

        for i, n in enumerate(NGRAM_SIZES):
            counts = np.array(list(ngram_counters[n].values()))
            if len(counts) > 0:
                plt.hist(counts, bins=bins, alpha=0.6, label=f'{n}-grams', color=colors[i % len(colors)])

        plt.xscale("log")
        plt.yscale("log")
        plt.title("Histogram of N-gram Frequencies")
        plt.xlabel("Frequency (Count)")
        plt.ylabel("Number of N-grams")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        save_path = "ngram_counts_histogram_combined.png"
        plt.savefig(save_path)
        logger.info(f"Saved combined histogram to {save_path}")


