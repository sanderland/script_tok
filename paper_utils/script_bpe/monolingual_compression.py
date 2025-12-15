# %%
# Monolingual tokenizer analysis

# %%
from script_bpe.corpus.registry import MONOLINGUAL_DATASETS, load_corpus_by_name
from script_bpe.train import load_tokenizers_for_dataset
from script_bpe.tokenizers.bpe.stats import compression_curve
import pandas as pd

# %%
N = 64000
num_undecodable = {}
compression_stats = []

for file in MONOLINGUAL_DATASETS:
    # BPE models
    bpe_tokenizers = load_tokenizers_for_dataset(file, n=N, model_name="bpe")
    bpe_curves = {ptok: compression_curve(tok) for ptok, tok in bpe_tokenizers.items() if tok}
    print(f"BPE: Loaded {len(bpe_curves)} tokenizers for {file}: {sorted(bpe_curves.keys())}")
    num_undecodable[file] = {}

    # Determine character count using previous heuristic: initial curve value / 2 from a script pretokenizer
    ref_name = "scriptenc" if "scriptenc" in bpe_curves else ("scriptenc_cb" if "scriptenc_cb" in bpe_curves else None)
    total_char_len = None
    if ref_name is not None:
        total_char_len = bpe_curves[ref_name][0] / 2

    # Record BPE start/end where possible
    for ptok, curve in bpe_curves.items():
        if total_char_len and total_char_len > 0:
            compression_stats.append(
                dict(
                    file=file,
                    model="bpe",
                    pretokenizer=ptok,
                    start=curve[0] / total_char_len,
                    end=curve[-1] / total_char_len,
                )
            )
        # undecodable count
        num_undecodable[file][f"bpe:{ptok}"] = bpe_tokenizers[ptok].stats()["num_undecodable"]

    # Unigram models
    uni_tokenizers = load_tokenizers_for_dataset(file, n=N, model_name="unigram")
    uni_tokenizers = {ptok: tok for ptok, tok in uni_tokenizers.items() if tok}
    if uni_tokenizers:
        print(f"Unigram: Loaded {len(uni_tokenizers)} tokenizers for {file}")
    for ptok, tok in uni_tokenizers.items():
        corpus = load_corpus_by_name(file, tok.pretokenizer)
        perf = tok.corpus_performance(corpus)
        compression_stats.append(
            dict(file=file, model="unigram", pretokenizer=ptok, start=None, end=perf["tokens_per_char"])
        )
        num_undecodable[file][f"unigram:{ptok}"] = tok.stats()["num_undecodable"]


# %%
def format_results(df, columns=None, floatfmt="{:.4f}", gradient="RdYlGn_r", relative=None):
    df = df[columns or df.columns]
    df = pd.concat([df, df.mean().rename("mean").to_frame().T])
    df_style = df.style
    if relative is not None:
        rel_val = df.div(df.median(axis=1), axis=0)
        df_style = df_style.background_gradient(
            cmap=gradient, gmap=rel_val, axis=None, vmin=1 - relative, vmax=1 + relative
        )
    elif gradient:
        df_style = df_style.background_gradient(cmap=gradient, axis=None)
    return df_style.format(floatfmt)


cdf = pd.DataFrame(compression_stats)

# BPE-only start table
bpe_start_df = cdf.query("model == 'bpe'").pivot(index="file", columns="pretokenizer", values="start")
if "bytes_gpt4" in bpe_start_df.columns:
    bpe_start_df = bpe_start_df.sort_values("bytes_gpt4", ascending=False)

format_results(
    bpe_start_df, columns=[c for c in ["bytes_gpt4", "scriptenc", "scriptenc_cb"] if c in bpe_start_df.columns]
).set_caption("Table 1 (Left two columns): Tokens/Char at start (BPE only)")

# %% Combined end table, model-prefixed columns
tmp = cdf.copy()
tmp["col"] = tmp["model"].fillna("bpe") + ":" + tmp["pretokenizer"]
end_df = tmp.pivot(index="file", columns="col", values="end")

nonosplit = [c for c in end_df.columns if "nosplit" not in c]
sort_col = "bpe:bytes_gpt4" if "bpe:bytes_gpt4" in end_df.columns else None
if sort_col:
    end_df = end_df.sort_values(sort_col, ascending=False)

format_results(end_df[nonosplit], relative=0.5).set_caption(
    "Table 1 (Compression) / Table 5 (Expanded): Tokens/Char after training (BPE and Unigram)"
)

# %%
nonosplit = [c for c in end_df.columns if "nosplit" not in c]
num_undecodable_df = pd.DataFrame.from_dict(num_undecodable, orient="index")
if "bytes_gpt4" in num_undecodable_df.columns:
    num_undecodable_df = num_undecodable_df.sort_values(by="bytes_gpt4", ascending=False)
format_results(num_undecodable_df[nonosplit], floatfmt="{:.1f}").set_caption(
    "Table 2 (Partial char): Number tokens including partial characters"
)
