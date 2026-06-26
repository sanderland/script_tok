#!/usr/bin/env python3
"""Tiebreak policy ablation for MinGram decoding.

All policies share the same minimum-token-count objective (PENALTY dominates).
They differ only in the secondary score used to rank equal-token-count paths:

  logprob : summed log-probability
  random  : fixed random per-token secondary score
  reverse : negated summed log-probability

All three runs keep the same deterministic exact-tie rule, so any differences
reflect the secondary score rather than a separate longest/shortest preference.
This script reports MorphAlign for each (language, policy) pair on the paper's
default f=1.15 MinGram models trained on FineWeb 5GB.
"""

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

from paper_utils.hybrid.utils import morphalign_paper_score
from paper_utils.hybrid.train_mingram import ADDITIONAL_VOCAB_SIZE
from script_bpe.tokenizers.mingram.model import MinGramModel
from script_bpe.tokenizers.unigram.model import UnigramToken

PAPER_UTILS_DIR = Path(__file__).parent
REPO_ROOT = PAPER_UTILS_DIR.parents[1]

MINGRAM_DIR = REPO_ROOT / "results/mingram"
RESULTS_DIR = REPO_ROOT / "results/hybrid"
OUT_JSON = RESULTS_DIR / "tiebreak_ablation.json"
SEG_DIR = RESULTS_DIR / "tiebreak_segmented"

MORPH_TOK_EVAL_DIR = REPO_ROOT / "data/morph-tok-eval"
MORPHALIGN_THRESHOLDS = [0.01]
MORPHALIGN_ITERATIONS = 10
MORPHALIGN_MODEL = "IBM1"
MORPHALIGN_METRIC_NAME = "test-morpho-score-mean-split-0.01-IBM1"

LANGUAGES = [
    {"lang": "eng", "label": "English",   "train_corpus": "fineweb_en_5gb",
     "gold_file": MORPH_TOK_EVAL_DIR / "data/morpho/eng-unimorph2uniseg_CELEX.tsv"},
    {"lang": "deu", "label": "German",    "train_corpus": "fineweb_de_5gb",
     "gold_file": MORPH_TOK_EVAL_DIR / "data/morpho/deu-unimorph2uniseg_CELEX.tsv"},
    # MorphyNet→UniSeg gold (parity with eng/deu CELEX; raw fin-unimorph.tsv
    # does not discriminate between tokenizers). Matches generate_morphalign_scatter.py.
    {"lang": "fin", "label": "Finnish",   "train_corpus": "fineweb_fi_5gb",
     "gold_file": MORPH_TOK_EVAL_DIR / "data/morpho/fin-unimorph2uniseg_morphynet.tsv"},
]

POLICIES = ["logprob", "random", "reverse"]
POLICY_LABELS = {
    "logprob": "log-prob",
    "random": "random",
    "reverse": "reverse",
}

DEFAULT_F = 1.15
DEFAULT_EM = 2
DEFAULT_P = 0.0
RANDOM_SCORE_SEED = 0
PENALTY = 100_000.0
EPS = 1e-10


def _load_align_module():
    align_path = MORPH_TOK_EVAL_DIR / "align.py"
    spec = importlib.util.spec_from_file_location("morph_tok_eval_align", align_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALIGN_MODULE = _load_align_module()


def _find_model(train_corpus: str, f: float, em: int, p: float) -> Path:
    model_dir = MINGRAM_DIR / train_corpus
    pat = re.compile(
        rf"mingram_f{re.escape(str(f))}_em{em}_p{re.escape(str(p))}_n{ADDITIONAL_VOCAB_SIZE}_[0-9a-f]+\.model\.json\.gz$"
    )
    for path in sorted(model_dir.glob("mingram_*.model.json.gz")):
        if pat.match(path.name):
            return path
    raise FileNotFoundError(f"No MinGram model found for {train_corpus} f={f} em={em} p={p}")


def _make_policy_scores(model: MinGramModel, policy: str) -> dict[int, float]:
    tokens = [model.tokens[token_id] for token_id in sorted(model.tokens)]
    if policy == "logprob":
        return {token.id: token.log_prob for token in tokens}
    if policy == "random":
        rng = random.Random(RANDOM_SCORE_SEED)
        return {token.id: rng.random() for token in tokens}
    if policy == "reverse":
        return {token.id: -token.log_prob for token in tokens}
    raise ValueError(f"Unknown policy: {policy}")


def encode_chunk_policy(
    model: MinGramModel,
    chunk,
    secondary_scores: dict[int, float],
) -> list[UnigramToken]:
    """Re-implementation of encode_chunk with selectable tiebreak policy.

    Score decomposes as (-PENALTY * num_tokens) + secondary(token) summed along path.
    Within a minimum-token-count path, the secondary objective decides. Exact
    score ties keep the last writer for all runs so the secondary score is the
    only experimental difference.
    """
    chunk_len = len(chunk)

    best_score = [float("-inf")] * (chunk_len + 1)
    best_prev: list[UnigramToken | None] = [None] * (chunk_len + 1)
    best_score[0] = 0.0

    trie_root = model.trie.root
    for pos in range(chunk_len):
        score = best_score[pos]
        if score == float("-inf"):
            continue
        node = trie_root
        for i in range(pos, chunk_len):
            node = node.get(chunk[i])
            if node is None:
                break
            token = node.get(None)
            if token is not None:
                nxt = i + 1
                new_score = score + secondary_scores[token.id] - PENALTY
                if new_score > best_score[nxt] + EPS or abs(new_score - best_score[nxt]) <= EPS:
                    best_score[nxt] = new_score
                    best_prev[nxt] = token

    tokens: list[UnigramToken] = []
    pos = chunk_len
    while pos > 0:
        token = best_prev[pos]
        assert token is not None, f"No segmentation for chunk: {chunk!r}"
        tokens.append(token)
        pos -= len(token.atomic_tokens)
    tokens.reverse()
    return tokens


def tokenize_word(
    model: MinGramModel,
    word: str,
    secondary_scores: dict[int, float],
) -> list[str]:
    out: list[str] = []
    pending_atomic_tokens: list[int] = []
    for chunk in model.pretokenizer.pretokenize(word):
        path = encode_chunk_policy(model, chunk, secondary_scores)
        for tok in path:
            pending_atomic_tokens.extend(int(tid) for tid in tok.atomic_tokens)
            decoded = model.pretokenizer.try_decode_strict(pending_atomic_tokens)
            if decoded is not None:
                out.append(decoded)
                pending_atomic_tokens = []
    assert not pending_atomic_tokens, f"Tokenization ended inside a partial character: {word!r}"
    assert "".join(out) == word, f"Tokenization mismatch: {word!r} != {'|'.join(out)!r}"
    return out


def segment_gold_file(
    model: MinGramModel,
    gold_file: Path,
    out_tsv: Path,
    secondary_scores: dict[int, float],
) -> int:
    total_tokens = 0
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_file, encoding="utf-8") as f_in, open(out_tsv, "w", encoding="utf-8") as f_out:
        for line in f_in:
            word, tag, _segments = line.rstrip("\n").split("\t")
            toks = tokenize_word(model, word, secondary_scores)
            total_tokens += len(toks)
            print(word, tag, "|".join(toks), sep="\t", file=f_out)
    return total_tokens


def _fmt_morphalign(value: float) -> str:
    return f"{morphalign_paper_score(value):.2f}"


def _segmentation_overlap(reference_path: Path, candidate_path: Path) -> float:
    matches = 0
    total = 0
    with open(reference_path, encoding="utf-8") as ref_file, open(candidate_path, encoding="utf-8") as cand_file:
        for ref_line, cand_line in zip(ref_file, cand_file, strict=True):
            ref_segmentation = ref_line.rstrip("\n").split("\t", 2)[2]
            cand_segmentation = cand_line.rstrip("\n").split("\t", 2)[2]
            total += 1
            if ref_segmentation == cand_segmentation:
                matches += 1
    return matches / total


def run_ablation(f: float, em: int, p: float, policies: list[str]) -> dict:
    reference_policy = "logprob"
    assert reference_policy in policies
    results: dict = {
        "f": f,
        "em": em,
        "p": p,
        "random_score_seed": RANDOM_SCORE_SEED,
        "reference_policy": reference_policy,
        "policy_order": policies,
        "policy_labels": {policy: POLICY_LABELS[policy] for policy in policies},
        "languages": {},
    }
    for cfg in LANGUAGES:
        lang = cfg["lang"]
        model_path = _find_model(cfg["train_corpus"], f, em, p)
        print(f"[{lang}] loading {model_path.name}")
        model = MinGramModel.load(str(model_path))
        policy_scores = {policy: _make_policy_scores(model, policy) for policy in policies}
        per_policy: dict = {}
        seg_paths: dict[str, Path] = {}
        for policy in policies:
            seg_name = f"{lang}_{model_path.stem}_{policy}.tsv"
            seg_path = SEG_DIR / seg_name
            seg_paths[policy] = seg_path
            secondary_scores = policy_scores[policy]
            total_tokens = segment_gold_file(model, cfg["gold_file"], seg_path, secondary_scores)
            morph, _ = ALIGN_MODULE.evaluate_segmentations(
                str(cfg["gold_file"]),
                str(seg_path),
                MORPHALIGN_THRESHOLDS,
                MORPHALIGN_ITERATIONS,
                MORPHALIGN_MODEL,
                skip_gold_train=True,
            )
            score = float(morph[MORPHALIGN_METRIC_NAME])
            per_policy[policy] = {"morphalign": score, "gold_tokens": total_tokens}
            print(f"[{lang}] {POLICY_LABELS[policy]:8s} MorphAlign={_fmt_morphalign(score)} tokens={total_tokens}")

        reference_path = seg_paths[reference_policy]
        for policy in policies:
            overlap = 1.0 if policy == reference_policy else _segmentation_overlap(reference_path, seg_paths[policy])
            per_policy[policy]["overlap_with_reference"] = overlap
            if policy != reference_policy:
                print(
                    f"[{lang}] overlap({POLICY_LABELS[policy]} vs {POLICY_LABELS[reference_policy]})="
                    f"{overlap:.6f}"
                )
        results["languages"][lang] = {"label": cfg["label"], "model": model_path.name, "policies": per_policy}
    return results


def print_report(results: dict) -> None:
    policy_order = results["policy_order"]
    col_width = max(12, max(len(POLICY_LABELS[policy]) for policy in policy_order) + 2)
    print()
    print("=" * 80)
    print(
        "Tiebreak ablation  |  "
        f"f={results['f']}  em={results['em']}  p={results['p']}  "
        "MorphAlign=Score x100"
    )
    print("=" * 80)
    header = f"{'Lang':12s}" + "".join(f"{POLICY_LABELS[p]:>{col_width}s}" for p in policy_order) + f"{'max-min':>{col_width}s}"
    print(header)
    for lang, entry in results["languages"].items():
        row = f"{entry['label']:12s}"
        vals = []
        for p in policy_order:
            v = entry["policies"][p]["morphalign"]
            vals.append(v)
            row += f"{_fmt_morphalign(v):>{col_width}s}"
        row += f"{_fmt_morphalign(max(vals) - min(vals)):>{col_width}s}"
        print(row)
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=float, default=DEFAULT_F)
    parser.add_argument("--em", type=int, default=DEFAULT_EM)
    parser.add_argument("--p", type=float, default=DEFAULT_P)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    args = parser.parse_args()

    results = run_ablation(args.f, args.em, args.p, args.policies)
    print_report(results)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
