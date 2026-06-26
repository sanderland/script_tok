#!/usr/bin/env python3
"""BPE-Init, FSP-BPE-Init, and MinGram compression sweep across overshoot factor f.

For each f in OVERSHOOT_FACTORS and each of the 6 compression-eval languages,
tokenize the Goldfish validation corpus with `bpe_init`, `bpe_init_fsp`, and
`mingram` (em=2, p=0.0) and report delta vs Default Unigram. Reuses the grid
cache where available and appends new entries for missing cells.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from paper_utils.hybrid.train_hybrid import OVERSHOOT_FACTORS, get_model_path as get_hybrid_model_path
from paper_utils.hybrid.train_mingram import get_model_path as get_mingram_model_path
from paper_utils.unigram.train_hyperparameters import DEFAULTS, get_model_path as get_unigram_model_path
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.tokenizers.unigram import UnigramModel

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
CACHE_FILE = RESULTS_DIR / "cache_train_eval_compression_grid.json"
OUT_FILE = RESULTS_DIR / "bpe_init_fsp_fsweep.json"

LANGS = [
    ("eng", "English",   "fineweb_en_5gb", "eng_latn_fishfood"),
    ("deu", "German",    "fineweb_de_5gb", "deu_latn_fishfood"),
    ("fin", "Finnish",   "fineweb_fi_5gb", "fin_latn_fishfood"),
    ("rus", "Russian",   "fineweb_ru_5gb", "rus_cyrl_fishfood"),
    ("arb", "Arabic",    "fineweb_ar_5gb", "arb_arab_fishfood"),
    ("kor", "Korean",    "fineweb_ko_5gb", "kor_hang_fishfood"),
]

FSP_OVERRIDES = dict(flat_score_prune=True, pre_final_vocab_factor=1.0)
MINGRAM_EM = 2
MINGRAM_P = 0.0


def _model_path_for(method: str, train: str, f: float) -> Path:
    if method == "bpe_init":
        return get_hybrid_model_path(train, {**DEFAULTS, "overshoot_factor": f})
    if method == "bpe_init_fsp":
        return get_hybrid_model_path(train, {**DEFAULTS, **FSP_OVERRIDES, "overshoot_factor": f})
    if method == "mingram":
        return get_mingram_model_path(train, f, MINGRAM_EM, MINGRAM_P)
    raise ValueError(f"unknown method {method}")


METHODS = ("bpe_init", "bpe_init_fsp", "mingram")


def _cache_key(train: str, eval_c: str, method: str, model_name: str) -> str:
    return f"tokens/{train}/{eval_c}/{method}/{model_name}"


def _evaluate(model_path: str, eval_corpus_name: str) -> int:
    model = UnigramModel.load(model_path)
    eval_corpus = load_corpus_by_name(eval_corpus_name, model.pretokenizer)
    perf = model.corpus_performance(eval_corpus)
    return int(perf["total_tokens_len"])


def main() -> None:
    cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}

    tasks: list[dict] = []

    baselines: dict[str, tuple[int, str]] = {}
    for lang, _label, train, eval_c in LANGS:
        default_path = get_unigram_model_path(train, DEFAULTS)
        default_key = _cache_key(train, eval_c, "default", default_path.name)
        assert default_key in cache, f"Missing default baseline for {lang}"
        baselines[lang] = (cache[default_key], default_key)

    for lang, _label, train, eval_c in LANGS:
        for f in OVERSHOOT_FACTORS:
            for method in METHODS:
                mp = _model_path_for(method, train, f)
                if not mp.exists():
                    print(f"SKIP: missing model {mp}")
                    continue
                key = _cache_key(train, eval_c, method, mp.name)
                if key in cache:
                    continue
                tasks.append({
                    "lang": lang, "train": train, "eval_c": eval_c,
                    "f": f, "method": method,
                    "model_path": str(mp), "cache_key": key,
                })

    print(f"Tasks to run: {len(tasks)}")
    if tasks:
        with ProcessPoolExecutor(max_workers=24) as ex:
            futures = {ex.submit(_evaluate, t["model_path"], t["eval_c"]): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                tokens = fut.result()
                cache[t["cache_key"]] = tokens
                print(f"  done: {t['lang']} f={t['f']} {t['method']:>12s} tokens={tokens}")
        CACHE_FILE.write_text(json.dumps(cache, indent=2))

    existing = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {"languages": {}}
    results: dict = {"factors": OVERSHOOT_FACTORS, "languages": {}}
    for lang, label, train, eval_c in LANGS:
        baseline, _ = baselines[lang]
        prior_by_f = existing["languages"].get(lang, {}).get("by_f", {})
        per_f: dict = {}
        for f in OVERSHOOT_FACTORS:
            prior_row = prior_by_f.get(str(f), {})
            row: dict = {}
            for method in METHODS:
                mp = _model_path_for(method, train, f)
                key = _cache_key(train, eval_c, method, mp.name)
                if key in cache:
                    tokens = cache[key]
                    row[method] = (tokens - baseline) / baseline * 100
                elif prior_row.get(method) is not None:
                    row[method] = prior_row[method]
                else:
                    row[method] = None
            per_f[str(f)] = row
        results["languages"][lang] = {
            "label": label, "train": train, "eval": eval_c,
            "baseline_tokens": baseline, "by_f": per_f,
        }

    OUT_FILE.write_text(json.dumps(results, indent=2))

    print()
    print("=" * 80)
    print("Compression delta vs Default (%, Goldfish eval)")
    print("=" * 80)
    for method in METHODS:
        print(f"\n[{method}]")
        print(f"{'f':>6}  " + "  ".join(f"{lang:>8s}" for lang, *_ in LANGS))
        for f in OVERSHOOT_FACTORS:
            cells = []
            for lang, *_ in LANGS:
                v = results["languages"][lang]["by_f"][str(f)][method]
                cells.append(f"{v:+.2f}%" if v is not None else "   -   ")
            print(f"{f:>6}  " + "  ".join(f"{c:>8s}" for c in cells))
    print()
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
