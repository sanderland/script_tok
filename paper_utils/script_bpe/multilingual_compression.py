# %%
# Multilingual Compression
#
# This notebook calculates the compression stats for large multilingual tokenizers on monolingual datasets ... very slowly.
#
# It caches results to make exploration less painful.

# %%
from script_bpe.tokenizers.bpe import BPETokenizer
from script_bpe.train import tokenizer_save_path, get_pretokenizer
from script_bpe.corpus.registry import load_corpus_by_name, MONOLINGUAL_DATASETS
from collections import Counter, defaultdict
import pandas as pd
import multiprocessing
import json

# %%
PRETOKENIZERS = ["bytes_gpt4_cb", "bytes_gpt4o_cb", "scriptenc_cb", "scriptenc_gpt4o_cb",
               "scriptenc2_cb", "scriptenc2_cbi", "scriptenc3_cb", "scriptenc3_cbi"]
TRAIN_N = 256000
NUM_PROCESSES = 16
TRAIN_CORPORA = {"culturax": "CulturaX-subsample-100-bal2"}
SPLIT_VAL_CORPORA = [
    "CulturaX-subsample-100-bal2val-1",
    "CulturaX-subsample-100-bal2val-2",
    "CulturaX-subsample-100-bal2val-3",
    "CulturaX-subsample-100-bal2val-4",
]
TEST_FILES = MONOLINGUAL_DATASETS + list(TRAIN_CORPORA.values()) + SPLIT_VAL_CORPORA

# %%
for ptok in PRETOKENIZERS:  # can not nest multiprocessing, so ensure these work before
    for file in TEST_FILES:
        corpus = load_corpus_by_name(file, get_pretokenizer(ptok))

# %%
def process_file_ptok(args):
    file, ptok, tokenizer = args
    corpus = load_corpus_by_name(file, tokenizer.pretokenizer)

    lengths = []
    for atomic_tokens, count in corpus:
        text = tokenizer.pretokenizer.decode(atomic_tokens)
        tokens = tokenizer.encode(text)
        lengths.append((count, len(text), len(atomic_tokens), len(tokens)))
    total_char_len = sum(char_len * count for count, char_len, _, _ in lengths)
    total_base_len = sum(count * atomic_tok_len for count, _, atomic_tok_len, _ in lengths)
    total_tokens_len = sum(count * encoded_len for count, _, _, encoded_len in lengths)
    result = {
        "total_char_len": total_char_len,
        "total_base_len": total_base_len,
        "total_tokens_len": total_tokens_len,
    }
    return file, ptok, result


all_results = {}

if __name__ == "__main__":
    for save_tag, train_corpus_name in TRAIN_CORPORA.items():
        RESULS_FILE = f"../results/{save_tag}_stats.json"
        tokenizers = {
            ptok: BPETokenizer.load(tokenizer_save_path(train_corpus_name, TRAIN_N, ptok)) for ptok in PRETOKENIZERS
        }

        # Load existing stats
        try:
            with open(RESULS_FILE, "r") as f:
                length_stats = json.load(f)
        except FileNotFoundError:
            length_stats = {}

        # Create processing tasks (filtering out already computed)
        tasks = [
            (file, ptok, tokenizers[ptok])
            for file in TEST_FILES
            for ptok in PRETOKENIZERS
            if ptok in tokenizers and ptok not in length_stats.get(file, {})
        ]

        print(
            f"Processing {len(tasks)} remaining tasks of out {len(TEST_FILES) * len(PRETOKENIZERS)} for {train_corpus_name}..."
        )

        with multiprocessing.Pool(NUM_PROCESSES) as pool:
            for file, ptok, stats in pool.imap_unordered(process_file_ptok, tasks):
                print(f"Completed {save_tag}  {ptok:<25} on {file:<15}: {stats}")
                length_stats.setdefault(file, {})
                length_stats[file][ptok] = stats
                with open(RESULS_FILE, "w") as f:  # Save after each result
                    json.dump(length_stats, f, indent=2)
        all_results[save_tag] = length_stats

# %%
merged_val_tag = "CulturaX (validation merged)"

for save_tag, length_stats in all_results.items():
    combined_results = defaultdict(Counter)
    for train_set, stats_by_pretok in length_stats.items():
        if train_set not in SPLIT_VAL_CORPORA:
            continue
        for pretok, stats in stats_by_pretok.items():
            for k, v in stats.items():
                combined_results[pretok][k] += v
    length_stats[merged_val_tag] = combined_results

if merged_val_tag not in TEST_FILES:
    TEST_FILES.append(merged_val_tag)

# %%
pretokenizers_to_vis = PRETOKENIZERS  # also ensures order

for save_tag, length_stats in all_results.items():
    df = pd.DataFrame.from_dict(length_stats, orient="index").map(
        lambda d: (d["total_tokens_len"] / d["total_char_len"])
    )
    df = df.loc[TEST_FILES, [c for c in pretokenizers_to_vis if c in df.columns]]  # order consistenly
    df.loc["mean monolingual"] = df.iloc[:12].mean()
    rel_val = df.div(df.median(axis=1), axis=0)
    (
        df.style.highlight_min(axis=1, props="font-weight:bold; text-decoration:underline;")
        .background_gradient(cmap="RdYlGn_r", gmap=rel_val, axis=None, vmin=0.9, vmax=1.1)
        .set_caption(f"Table 4: Compression ratios (#tokens/char) for {save_tag}")
        .format("{:.4f}")
    )
