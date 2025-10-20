from script_bpe.corpus import load_corpus_by_name
from script_bpe.pretokenize import PRETOKENIZER_REGISTRY, get_pretokenizer
from script_bpe.pretokenize.pretokenizer import ScriptPretokenizerConfig
from script_bpe.pretokenize.scriptencoding import ScriptEncodingV2
from script_bpe.tokenizers.unigram.model import UnigramToken, UnigramModel
from script_bpe.tokenizers.unigram.trainer import (
    UnigramTrainer,
    UnigramTrainerConfig,
)
from script_bpe.train import train_tokenizer, tokenizer_save_path
from script_bpe.utils import create_logger
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from json import JSONDecodeError
import math
import regex as re
from typing import Callable
from dataclasses import dataclass
from cyclopts import App


def supertoken_length(pretokenizer, seq: tuple[int, ...]) -> int:
    tokenstr = pretokenizer.decode(seq)
    return len(pretokenizer.pretokenize(tokenstr))

line_regex = r"[\r\n]*[^\r\n]*" # This pattern will be used by our custom pretokenizer.
WORD_REGEX = r"[\p{L}]+(?:['’]\p{L}+)?"

def supertoken_words_filter_str(tokenstr: str) -> bool:
    if re.fullmatch(rf",? ?{WORD_REGEX}(?:,? ?{WORD_REGEX})*", tokenstr) is not None:
        return True
    if re.fullmatch(rf" ?{WORD_REGEX}(?:-?{WORD_REGEX})+", tokenstr) is not None:
        return True
    if re.fullmatch(r" ?([A-Z]\.){2,8}", tokenstr) is not None:
        return True # abbreviations
    if re.fullmatch(r"\.[a-z]{2,4}", tokenstr) is not None:
        return True # .com, .net, etc.
    if re.fullmatch(r"[a-z]{3,8}://(?:www\.)?", tokenstr) is not None:
        return True # urls, file paths, etc.
    return False

def supertoken_words_filter_nocomma(pretokenizer, seq: tuple[int, ...]) -> bool:
    tokenstr = pretokenizer.decode(seq)
    if re.fullmatch(rf", {WORD_REGEX}", tokenstr) is not None:
        return False
    else:
        return supertoken_words_filter_str(tokenstr)

def supertoken_words_filter(pretokenizer, seq: tuple[int, ...]) -> bool:
    tokenstr = pretokenizer.decode(seq)
    return supertoken_words_filter_str(tokenstr)

def supertoken_len_filter(pretokenizer, seq: tuple[int, ...], max_len: int) -> bool:
    return len(seq) <= max_len

SUPERTOKEN_FILTERS: dict[str, Callable] = {
    "all": lambda pt, seq: True,
    "words": supertoken_words_filter,
    "words_nocomma": supertoken_words_filter_nocomma,
    "len_8c": lambda pt, seq: supertoken_len_filter(pt, seq, 16),
    "len_16c": lambda pt, seq: supertoken_len_filter(pt, seq, 32),
}

def _filter_metadata_for_comparison(metadata: dict[str, object]) -> dict[str, object]:
    """Return a shallow copy of metadata without verbose keys for tabular comparison."""
    excluded_keys = {
        "ngram_init_distribution",
        "top_allowed_supertokens",
        "top_rejected_supertokens",
        "totals_removed",
    }
    return {
        key: value
        for key, value in metadata.items()
        if key not in excluded_keys and "supertoken" not in key and "ngram" not in key
    }

# --- Configuration ---
pretokenizer_name = "scriptenc2_cbi"
retrain = False
n_cpus = 4
# Analysis config
TOP_N = 250
SUPERTOKEN_INIT_SIZE = 250_000



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

@dataclass(frozen=True)
class Variant:
    filter_name: str
    max_ngram: int


INITIAL_100K_MODEL: dict[str, UnigramModel] = {}
BASELINE_64K_MODEL: dict[str, UnigramModel] = {}


def _get_initial_model(corpus_name: str) -> UnigramModel:
    global INITIAL_100K_MODEL
    if corpus_name not in INITIAL_100K_MODEL:
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
        INITIAL_100K_MODEL[corpus_name] = model
    return INITIAL_100K_MODEL[corpus_name]


def _get_baseline_model(corpus_name: str) -> UnigramModel:
    global BASELINE_64K_MODEL
    if corpus_name not in BASELINE_64K_MODEL:
        baseline = train_tokenizer(
            pretokenizer_name=pretokenizer_name,
            model_name="unigram",
            corpus_name=corpus_name,
            additional_vocab_size=64_000,
            n_cpus=n_cpus,
            retrain=retrain,
            report=False,
            trainer_config_kwargs=trainer_config_kwargs,
        )
        assert baseline, "Could not train or load the baseline 64k model."
        BASELINE_64K_MODEL[corpus_name] = baseline
    return BASELINE_64K_MODEL[corpus_name]


def run_experiment(variant: Variant, corpus_name: str, view: bool = False) -> dict[str, object] | None:
    logger = create_logger("superscript", verbose=True)
    logger.info(f"Variant: filter={variant.filter_name}, max_ngram={variant.max_ngram}")

    model = _get_initial_model(corpus_name)
    baseline_64k_model = _get_baseline_model(corpus_name)

    final_model_tag = f"scriptenc2_cbi_supertokens_{variant.filter_name}_n{variant.max_ngram}_singlespan"
    final_model_save_path = tokenizer_save_path(corpus_name, 64_000, final_model_tag, "unigram")
    final_model = None
    try:
        final_model = UnigramModel.load(final_model_save_path)
        logger.info(f"Loaded existing supertoken model from {final_model_save_path}")
        if view:
            return {
                "model_name": f"Supertoken 64k (single span)",
                "filter": variant.filter_name,
                "max_ngram": variant.max_ngram,
                **_filter_metadata_for_comparison(final_model.metadata),
            }
    except (FileNotFoundError, JSONDecodeError):
        if view:
            return None
        logger.info(f"No existing supertoken model found at {final_model_save_path}; training a new one...")
        analysis_pretokenizer = get_pretokenizer("scriptenc_line_regex")
        corpus = load_corpus_by_name(corpus_name, analysis_pretokenizer)

        ngram_sizes = list(range(2, variant.max_ngram + 1))
        unigram_counts = Counter()
        ngram_counters = {n: Counter() for n in ngram_sizes}
        logger.info(f"Tokenizing corpus and counting n-grams with single token span...")

        def tokenize_encoded(atomic_tokens: list[int]) -> list[int]:
            lattice = model.make_lattice(atomic_tokens)
            viterbi_path, _ = lattice.viterbi()
            return [token.id for token in viterbi_path]

        for pretoken_seq, freq in corpus:
            text = model.pretokenizer.decode(pretoken_seq)
            chunks = [tokenize_encoded(chunk) for chunk in model.pretokenizer.pretokenize(text)]
            for chunk in chunks:
                for token_id in chunk:
                    unigram_counts[(token_id,)] += freq

            for n in ngram_sizes:
                if len(chunks) >= n:
                    for i in range(len(chunks) - n + 1):
                        if not all(len(c)==1 for c in chunks[i:i+n]):
                            continue
                        ngram = tuple(c[0] for c in chunks[i:i+n])
                        ngram_counters[n][ngram] += freq            

        logger.info(f"Scoring and selecting top {SUPERTOKEN_INIT_SIZE:,} tokens...")
        scores = Counter()
        scores_by_n = defaultdict(Counter)
        rejected_scores = Counter()
        allowed_token_f = SUPERTOKEN_FILTERS[variant.filter_name]
        for n in ngram_sizes:
            for ngram_ids, count in ngram_counters[n].items():
                seq = tuple(x for tid in ngram_ids for x in model.tokens_by_id[tid].atomic_tokens)
                if allowed_token_f(baseline_64k_model.pretokenizer, seq):
                    scores[seq] += count * n
                    scores_by_n[n][seq] += count * n
                else:
                    rejected_scores[seq] += count * n
        logger.info(f"Rejected {len(rejected_scores):,} supertokens and allowed {len(scores):,}.")
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        sorted_rejected_scores = sorted(rejected_scores.items(), key=lambda x: -x[1])
        readable_allowed = [
            (baseline_64k_model.pretokenizer.tokens_repr(seq), score)
            for seq, score in sorted_scores[:10]
        ]
        readable_rejected = [
            (baseline_64k_model.pretokenizer.tokens_repr(seq), score)
            for seq, score in sorted_rejected_scores[:10]
        ]
        logger.info(f"Top 10 allowed supertokens: {readable_allowed}")
        logger.info(f"Top 10 rejected supertokens: {readable_rejected}")

        for t in model.tokens.values():
            seq = tuple(t.atomic_tokens)
            scores[seq] += max(1, unigram_counts.get((t.id,), 0))
            scores_by_n[1][seq] += max(1, unigram_counts.get((t.id,), 0))
        final_vocab_for_trainer = UnigramTrainer.init_vocab_normalize_scores(
            [(score, seq) for seq, score in scores.items()],
            SUPERTOKEN_INIT_SIZE,
        )
        selected_scores_by_n = defaultdict(list)
        analysis_pretokenizer = get_pretokenizer("scriptenc_line_regex")
        for token in final_vocab_for_trainer:
            seq = tuple(token.atomic_tokens)
            score = math.exp(token.log_prob)
            selected_scores_by_n[len(seq)].append((seq, score))
        init_ngram_distribution = {
            str(n): {
                "num": len(pairs),
                "top": [
                    {
                        "score": s,
                        "text": analysis_pretokenizer.tokens_repr(seq),
                        "atomic_len": len(seq),
                    }
                    for seq, s in sorted(pairs, key=lambda x: -x[1])[:10]
                ],
            }
            for n, pairs in selected_scores_by_n.items()
        }
        logger.info(f"--- Launching final training with {len(final_vocab_for_trainer):,} initial tokens ---")
        final_config = UnigramTrainerConfig(
            additional_vocab_size=64_000,
            forced_initial_vocab=final_vocab_for_trainer,
        )
        final_trainer = UnigramTrainer(analysis_pretokenizer, corpus, final_config)
        final_model = final_trainer.train()
        final_model.metadata = final_model.metadata or {}
        final_model.metadata["supertoken_filter"] = variant.filter_name
        final_model.metadata["single_token_span"] = True
        final_model.metadata["top_allowed_supertokens"] = [
            {
                "score": score,
                "text": analysis_pretokenizer.tokens_repr(seq),
                "atomic_len": len(seq),
            }
            for seq, score in sorted_scores[:10]
        ]
        final_model.metadata["top_rejected_supertokens"] = [
            {
                "score": score,
                "text": analysis_pretokenizer.tokens_repr(seq),
                "atomic_len": len(seq),
            }
            for seq, score in sorted_rejected_scores[:10]
        ]
        final_model.metadata["ngram_init_distribution"] = init_ngram_distribution
        final_model.save(final_model_save_path)

    # Write report for this variant unless in view mode
    if not view:
        supertokens_report_path = final_model_save_path.replace(".json.gz", ".md")
        with open(supertokens_report_path, "w") as f:
            init_dist = final_model.metadata.pop("ngram_init_distribution", None)
            base_report = final_model.report()
            f.write(base_report)
            if init_dist is not None:
                f.write("\n\n## Initial n-gram distribution (by score)\n\n")
                dist_rows = [
                    {"n": n, "num": info["num"]}
                    for n, info in init_dist.items()
                ]
                f.write(tabulate(dist_rows, headers="keys", tablefmt="github"))
                f.write("\n")
            supertokens = [t for t in final_model.tokens.values() if supertoken_length(baseline_64k_model.pretokenizer, t.atomic_tokens) > 1]
            supertoken_counts = Counter(supertoken_length(baseline_64k_model.pretokenizer, t.atomic_tokens) for t in final_model.tokens.values())
            f.write(f"\n\n## Supertoken Length Distribution\n\n")
            f.write(tabulate(supertoken_counts.most_common(), headers=["Length", "Count"], tablefmt="github"))
            f.write("\n")
            if "top_rejected_supertokens" in final_model.metadata and final_model.metadata["top_rejected_supertokens"]:
                f.write("\n\n## Top Rejected Supertokens\n\n")
                rej_rows = [
                    {
                        "Score": f"{item['score']:.6g}",
                        "Text": repr(item["text"]),
                        "Atomic Len": item["atomic_len"],
                    }
                    for item in final_model.metadata["top_rejected_supertokens"]
                ]
                f.write(tabulate(rej_rows, headers="keys", tablefmt="github"))
                f.write("\n")
            f.write(f"\n\n## Supertokens ({len(supertokens):,} total)\n\n")
            rows = [{
                "ID": t.id,
                "Probability": f"{math.exp(t.log_prob):.6g}",
                "Log Probability": f"{t.log_prob:.4f}", 
                "Text": repr(final_model.pretokenizer.tokens_repr(t.atomic_tokens)),
                "Atomic Len": len(t.atomic_tokens),
                "Supertoken Length": supertoken_length(baseline_64k_model.pretokenizer, t.atomic_tokens),
            } for t in supertokens ]
            f.write(tabulate(rows, headers="keys", tablefmt="github"))
            f.write("\n")
        logger.info(f"Saved supertokens report with {len(rows):,} supertokens to {supertokens_report_path}")

    return {
        "model_name": f"Supertoken 64k",
        "filter": variant.filter_name,
        "max_ngram": variant.max_ngram,
        **_filter_metadata_for_comparison(final_model.metadata),
    }


def run_all_and_tabulate(filter_names: list[str], max_ngrams: list[int], corpus_name: str, view: bool = False) -> None:
    logger = create_logger("superscript", verbose=True)
    logger.info(f"Pretokenizer: {pretokenizer_name}, Corpus: {corpus_name}")
    logger.info(f"Trainer Config: {trainer_config_kwargs}")
    logger.info(f"Filters: {filter_names}; Max n-grams: {max_ngrams}; View: {view}")

    _ = _get_initial_model(corpus_name)
    baseline_64k_model = _get_baseline_model(corpus_name)

    baseline_report_path = tokenizer_save_path(corpus_name, 64_000, pretokenizer_name, "unigram").replace(".json.gz", ".md")
    with open(baseline_report_path, "w") as f:
        f.write(baseline_64k_model.report())
    logger.info(f"Saved baseline report to {baseline_report_path}")

    comparison_rows = [
        {"model_name": "Baseline 64k", "filter":"", "max_ngram": "", **_filter_metadata_for_comparison(baseline_64k_model.metadata)}
    ]

    for filter_name in filter_names:
        for max_ngram in max_ngrams:
            row = run_experiment(Variant(filter_name=filter_name, max_ngram=max_ngram), corpus_name=corpus_name, view=view)
            if row is not None:
                comparison_rows.append(row)

    logger.info("--- Final Model Comparison ---")
    print(tabulate(comparison_rows, headers="keys", tablefmt="grid"))


app = App()

def _parse_csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]

def _parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]

@app.default
def cli(filters: str = "all,words,words_nocomma,len_8c,len_16c", max_ngrams: str = "2,4,8", corpus_name: str = "eng_latn_300mb", view: bool = False) -> None:
    filter_names = _parse_csv_strs(filters)
    max_ngram_values = _parse_csv_ints(max_ngrams)
    run_all_and_tabulate(filter_names, max_ngram_values, corpus_name=corpus_name, view=view)

if __name__ == '__main__':
    app()


