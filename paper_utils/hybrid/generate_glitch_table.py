#!/usr/bin/env python3
"""Generate the under-trained ("glitch") token table from the trained d24 base models.

Under-trained tokens (Land & Bartolo, "Fishing for Magikarp") show up as anomalously low
INPUT-embedding norm: vocab the LM barely saw stays near init. We read `wte` from each
d24 checkpoint, flag the low-norm cluster, and categorise by the DECODED string:
  - atomic   : decodes to U+FFFD (sub-character SCRIPT unit) -- ~shared across methods,
               since all methods share the SCRIPT pretokenizer (not a tokenizer signal).
  - single   : one real character.
  - multi    : a real multi-character token -- the DIFFERENTIATING signal: vocabulary a
               method minted that the LM never learned ("wasted" vocab).

Run across all configured downstream seeds; the multi-char count is near-deterministic across seeds
(it tracks corpus rarity, not init), so we report the mean. Two-stage so it renders ANYWHERE
from cache (the 4.3GB checkpoints are gitignored and optional):
  - If `results/downstream/cache_glitch.json` exists, render from it (no torch / checkpoints).
  - Otherwise compute from checkpoints (needs torch + pynanochat), write cache, render.
    Pass --recompute to force.

By default recompute looks under `results/downstream/checkpoints/`, with the
same `nc_runs/...` and `nanochat_d24_bpe32k/...` layout used by the training box.
Set GLITCH_CHECKPOINT_ROOT to point at a different local checkout.

Writes `results/mingram_paper/tables/table_glitch.tex` (booktabs, wrap in `table`).

Usage:
  uv run python -m paper_utils.hybrid.generate_glitch_table [--recompute]
"""

import argparse
import json
import os
import statistics as st
from pathlib import Path

from paper_utils.hybrid.utils import paper_table_path

REPO_ROOT = Path(__file__).parents[2]
CACHE = REPO_ROOT / "results" / "downstream" / "cache_glitch.json"
F115_CACHE = REPO_ROOT / "results" / "downstream" / "cache_glitch_f115.json"
OUT_TEX = paper_table_path("table_glitch.tex", extra=True)
CHECKPOINT_ROOT = Path(os.environ.get("GLITCH_CHECKPOINT_ROOT", REPO_ROOT / "results" / "downstream" / "checkpoints"))
NC_RUNS_ROOT = CHECKPOINT_ROOT / "nc_runs"
CANON_ROOT = CHECKPOINT_ROOT / "nanochat_d24_bpe32k"

SEEDS, DEPTH = list(range(42, 62)), 24  # n=20: 42-45 original + 46-61 scale-up (matches downstream table)


def _ncr(tag: str) -> dict[int, str]:
    return {
        s: str(NC_RUNS_ROOT / f"{tag}_d24_s{s}" / "base_checkpoints" / f"{tag}_d24_s{s}" / "model_005590.pt")
        for s in SEEDS
    }


def _canon(sub: str) -> str:
    return str(CANON_ROOT / "base_checkpoints" / sub / "model_005590.pt")


# bpe/mingram seed-42 checkpoints predate the seed-tagged naming (lived in the shared
# nanochat_d24_bpe32k base_dir under their original tags); 43-45 use bpe_d24_s{S} there too,
# and the 46-61 scale-up seeds live in the standard nc_runs/{tag}_d24_s{S} layout.
_BPE_CK = {
    42: _canon("bpe_n32768_d24"),
    **{s: _canon(f"bpe_d24_s{s}") for s in (43, 44, 45)},
    **{s: str(NC_RUNS_ROOT / f"bpe_d24_s{s}" / "base_checkpoints" / f"bpe_d24_s{s}" / "model_005590.pt")
       for s in range(46, 62)},
}
# MinGram-PP (careful MinGram-PP prune) was eval'd downstream at f=8 (its compression
# optimum) on seeds 45-64; checkpoints live in slurm or _dbg (debug-node) base dirs.
def _mingram_pp_f8_ck() -> dict[int, str]:
    import glob
    out: dict[int, str] = {}
    for s in SEEDS + list(range(62, 65)):
        for sub in (f"ds_mingram_pp_f8_d24_s{s}", f"ds_mingram_pp_f8_d24_s{s}_dbg"):
            g = glob.glob(str(NC_RUNS_ROOT / sub / "base_checkpoints" / "*" / "model_005590.pt"))
            if g:
                out[s] = g[0]
                break
    return out


_MINGRAM_PP_TOK = "results/mingram/fineweb_en_5gb/mingram_f8.0_em2_p0.9_pcmi_n32768_1f852aea12ed62ea.model.json.gz"


# canonical PathPiece is now pb=0.1; its downstream checkpoints live under ds_pathpiece_pb01_d24_s*
# (slurm) or *_dbg (debug node), seeds 42-61.
def _pp_pb01_ck() -> dict[int, str]:
    import glob
    out: dict[int, str] = {}
    for s in SEEDS:
        for sub in (f"ds_pathpiece_pb01_d24_s{s}", f"ds_pathpiece_pb01_d24_s{s}_dbg"):
            g = glob.glob(str(NC_RUNS_ROOT / sub / "base_checkpoints" / "*" / "model_005590.pt"))
            if g:
                out[s] = g[0]
                break
    return out


_CACHE_ONLY_CK: dict[int, str] = {}

# (label, key, tokenizer dotted class, tokenizer path, {seed: checkpoint})
RUNS = [
    ("BPE", "bpe", "script_bpe.tokenizers.bpe.BPETokenizer",
     "results/hybrid/fineweb_en_5gb/bpe_n32768.model.json.gz", _BPE_CK),
    ("Unigram", "unigram_default", "script_bpe.tokenizers.unigram.model.UnigramModel",
     "results/unigram_sweeps/fineweb_en_5gb/corpus_long_n32768_6746a31be63768c7.model.json.gz",
     _ncr("unigram_default")),
    ("FSP", "unigram_fsp", "script_bpe.tokenizers.unigram.model.UnigramModel",
     "results/unigram_sweeps/fineweb_en_5gb/corpus_long_n32768_5ea754f096ae45e5.model.json.gz",
     _ncr("unigram_fsp")),
    ("Unigram\\hspace{0pt}-BPE\\hspace{0pt}-Init", "bpe_init_f1.15",
     "script_bpe.tokenizers.unigram.model.UnigramModel",
     "results/hybrid/fineweb_en_5gb/bpe_init_1.15_n32768_ddecc876b8230ccc.model.json.gz",
     _CACHE_ONLY_CK),
    ("FSP\\hspace{0pt}-BPE\\hspace{0pt}-Init", "fsp_bpe_init_f1.15",
     "script_bpe.tokenizers.unigram.model.UnigramModel",
     "results/hybrid/fineweb_en_5gb/bpe_init_1.15_n32768_f6a13682effae861.model.json.gz",
     _CACHE_ONLY_CK),
    ("MinGram", "mingram_f1.15", "script_bpe.tokenizers.mingram.model.MinGramModel",
     "results/mingram/fineweb_en_5gb/mingram_f1.15_em2_p0.0_n32768_6a46f861edc5191c.model.json.gz",
     _CACHE_ONLY_CK),
    ("\\mingrampp{}", "mingram_pp", "script_bpe.tokenizers.mingram.model.MinGramModel",
     _MINGRAM_PP_TOK, _mingram_pp_f8_ck()),
    ("PathPiece\\hspace{0pt}-BPE", "pathpiece", "script_bpe.tokenizers.pathpiece.model.PathPieceModel",
     "results/pathpiece/fineweb_en_5gb/pathpiece_bpe_iv262144_L1024_pb0.1_n32768_d1e30bdf3b87710c.model.json.gz",
     _pp_pb01_ck()),
    ("ConvexTok", "convextok", "script_bpe.tokenizers.convextok.model.ConvexTokModel",
     "results/convextok_tokenizers/fineweb_en_5gb/n32768_cmin50_mp200000_L32_det.json.gz",
     _ncr("convextok")),
]


def _flag_one(adapter, ckpt, np, torch):
    """Return (flagged, frag, single, multi, single_examples, multi_examples) for one checkpoint."""
    V = adapter.get_vocab_size()
    inn = torch.load(ckpt, map_location="cpu", weights_only=False)["transformer.wte.weight"].float().norm(dim=1).numpy()
    n_pad = len(inn) - V
    pad_level = float(np.median(inn[V:])) if n_pad > 0 else float(inn.min())
    thr = (pad_level + float(np.median(inn[:V]))) / 2.0
    bos = adapter.get_bos_token_id()
    frag, single, multi = [], [], []
    for i in range(V):
        if inn[i] >= thr or i == bos:
            continue
        s = adapter.decode([i])
        (frag if "�" in s or not s else single if len(s) == 1 else multi).append((float(inn[i]), s))
    single.sort()
    multi.sort()
    return frag, single, multi


def _compute() -> dict:
    """Per-method, per-seed flag counts. Heavy imports local so cache-render needs none."""
    import importlib

    import numpy as np
    import torch

    from pynanochat import ScriptBPETokenizerAdapter

    methods = []
    for label, key, dotted, tok_path, ck_by_seed in RUNS:
        mod, _, cls = dotted.rpartition(".")
        adapter = ScriptBPETokenizerAdapter(getattr(importlib.import_module(mod), cls).load(tok_path))
        per_seed, ex_single, ex_multi = {}, None, None
        for s in SEEDS:
            ckpt = ck_by_seed.get(s)
            if not ckpt or not Path(ckpt).exists():
                print(f"[skip] {label} s{s}: missing {ckpt}")
                continue
            frag, single, multi = _flag_one(adapter, ckpt, np, torch)
            per_seed[str(s)] = {"flagged": len(frag) + len(single) + len(multi),
                                "frag": len(frag), "single": len(single), "multi": len(multi)}
            if s == 42 or ex_multi is None:
                ex_single, ex_multi = single, multi
            print(f"[ok] {label} s{s}: {len(frag)} atomic + {len(single)} single + {len(multi)} multi")
        if not per_seed:
            continue
        multis = [d["multi"] for d in per_seed.values()]
        methods.append({
            "label": label, "key": key, "vocab": adapter.get_vocab_size(),
            "per_seed": per_seed,
            "atomic_mean": st.mean(d["frag"] for d in per_seed.values()),
            "single_mean": st.mean(d["single"] for d in per_seed.values()),
            "multi_mean": st.mean(multis), "multi_min": min(multis), "multi_max": max(multis),
            "n_seeds": len(per_seed),
            "single_examples_s42": ex_single, "multi_examples_s42": ex_multi,
        })
    return {"seeds": SEEDS, "depth": DEPTH, "methods": methods}


def _fmt(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def _fmt_mean(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _fmt_range(low: float, high: float) -> str:
    if low == high:
        return str(int(low))
    return f"{int(low)}--{int(high)}"


def _is_multi_seed(data: dict) -> bool:
    return "seeds" in data and data["methods"] and "per_seed" in data["methods"][0]


def _load_cached() -> dict:
    data = json.loads(CACHE.read_text())
    if F115_CACHE.exists():
        f115 = json.loads(F115_CACHE.read_text())
        by_key = {method["key"]: method for method in data["methods"]}
        by_key.update({method["key"]: method for method in f115["methods"]})
        order = [key for _, key, *_ in RUNS if key in by_key and key != "mingram"]
        data["methods"] = [by_key[key] for key in order]
        data["seeds"] = sorted(set(data.get("seeds", [])) | set(f115.get("seeds", [])))
    return data


def _render(data: dict) -> str:
    methods = data["methods"]
    by_key = {m["key"]: m for m in methods}
    order = [r[1] for r in RUNS if r[1] in by_key]
    multi_seed = _is_multi_seed(data)
    if multi_seed:
        min_multi = min(m["multi_mean"] for m in methods) if methods else 0
        seed_text = ", ".join(str(seed) for seed in data["seeds"])
        header_comment = f"% Under-trained (low input-embedding-norm) tokens; depth-{data['depth']} checkpoints, seeds {seed_text}."
        column_header = "Method & $n$ & atomic mean & single mean & multi mean & multi range \\\\"
    else:
        min_multi = min(m["multi"] for m in methods) if methods else 0
        header_comment = f"% Under-trained (low input-embedding-norm) tokens; seed-{data['seed']} depth-{data['depth']} checkpoints."
        column_header = "Method & Under-tr. & atomic & single-char & multi-char \\\\"

    lines = [
        "% Intended to be wrapped in \\begin{table}...\\end{table}. Requires \\usepackage{booktabs}.",
        header_comment,
        "% 'atomic' = shared SCRIPT sub-char units (not a tokenizer signal); 'multi-char' = wasted vocab.",
        "% multi-char is near-deterministic across seeds (tracks corpus rarity, not init).",
        "\\begin{tabular}{@{}lrrrrr@{}}" if multi_seed else "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        column_header,
        "\\midrule",
    ]
    for key in order:
        m = by_key[key]
        if multi_seed:
            multi_cell = _fmt_mean(m["multi_mean"])
            if m["multi_mean"] == min_multi:
                multi_cell = f"\\textbf{{{multi_cell}}}"
            lines.append(
                f"{m['label']} & {m['n_seeds']} & {_fmt_mean(m['atomic_mean'])} & {_fmt_mean(m['single_mean'])} "
                f"& {multi_cell} & {_fmt_range(m['multi_min'], m['multi_max'])} \\\\"
            )
        else:
            multi_cell = f"\\textbf{{{m['multi']}}}" if m["multi"] == min_multi else str(m["multi"])
            lines.append(f"{m['label']} & {m['flagged']} & {m['frag']} & {m['single']} & {multi_cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="recompute from checkpoints (ignore cache)")
    args = ap.parse_args()

    if CACHE.exists() and not args.recompute:
        print(f"[cache] loading {CACHE}")
        data = _load_cached()
    else:
        data = _compute()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"[cache] wrote {CACHE}")

    tex = _render(data)
    OUT_TEX.write_text(tex)
    print(f"wrote {OUT_TEX}\n")
    print(tex)
    # echo the differentiating multi-char tokens for the caption / write-up
    suffix = "s42" if _is_multi_seed(data) else None
    print("multi-char under-trained tokens (the 'wasted vocab' signal; seed 42 examples):")
    for m in data["methods"]:
        examples = m[f"multi_examples_{suffix}"] if suffix else m["multi_examples"]
        toks = ", ".join(repr(s) for _, s in examples) or "(none)"
        print(f"  {m['label']}: {toks}")


if __name__ == "__main__":
    main()
