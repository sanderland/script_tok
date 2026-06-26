#!/usr/bin/env python3
"""Generate the appendix tiebreak ablation table.

Reads `results/hybrid/tiebreak_ablation.json` produced by
`run_tiebreak_ablation.py` and writes a LaTeX tabular body to
`results/mingram_paper/tables/app_table_tiebreak.tex`.

Compression is identical across all policies by construction, so the
table reports MorphAlign and overlap vs the log-prob reference. PathPiece-BPE
is included as an external MorphAlign reference point.
"""

import json
from pathlib import Path

from paper_utils.hybrid.train_pathpiece import get_model_path as get_pathpiece_model_path
from paper_utils.hybrid.utils import morphalign_paper_score, paper_table_path

REPO_ROOT = Path(__file__).parents[2]
RESULTS_DIR = REPO_ROOT / "results/hybrid"
SRC_JSON = RESULTS_DIR / "tiebreak_ablation.json"
MORPHALIGN_JSON = RESULTS_DIR / "cache_morphalign_scatter.json"
OUT_TEX = paper_table_path("table_tiebreak.tex", appendix=True)

LANG_ORDER = ["eng", "deu", "fin"]
PATHPIECE_LABEL = "PathPiece"
TRAIN_CORPUS_BY_LANG = {
    "eng": "fineweb_en_5gb",
    "deu": "fineweb_de_5gb",
    "fin": "fineweb_fi_5gb",
}


def _fmt(value: float) -> str:
    return f"{morphalign_paper_score(value):.2f}"


def _bold_best_per_row(values: list[float]) -> list[str]:
    """Return formatted strings with the row maximum in \\textbf{}."""
    best = max(values)
    out = []
    for value in values:
        formatted = _fmt(value)
        if abs(value - best) < 1e-9:
            formatted = rf"\textbf{{{morphalign_paper_score(value):.2f}}}"
        out.append(formatted)
    return out


def _fmt_overlap(value: float) -> str:
    return rf"{value * 100:.0f}\%"


def _pathpiece_morphalign(morphalign_cache: dict, lang: str) -> float:
    model_path = get_pathpiece_model_path(TRAIN_CORPUS_BY_LANG[lang], init="bpe")
    cache_key = f"{lang}/pathpiece_bpe/{model_path.name}"
    return float(morphalign_cache[cache_key])


def build_table(data: dict, morphalign_cache: dict) -> str:
    langs = data["languages"]
    policy_order = data["policy_order"]
    policy_labels = data["policy_labels"]
    reference_policy = data["reference_policy"]
    overlap_policies = [policy for policy in policy_order if policy != reference_policy]
    morphalign_columns = len(policy_order) + 1

    col_spec = "l" + "r" * morphalign_columns + "|" + "r" * len(overlap_policies)
    header_top = [
        "",
        f"\\multicolumn{{{morphalign_columns}}}{{c}}{{MorphAlign Score}}",
        f"\\multicolumn{{{len(overlap_policies)}}}{{c}}{{Overlap vs {policy_labels[reference_policy]}}}",
    ]
    header_bottom = (
        ["Language"]
        + [policy_labels[policy] for policy in policy_order]
        + [PATHPIECE_LABEL]
        + [policy_labels[policy] for policy in overlap_policies]
    )

    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_top) + " \\\\",
        f"\\cmidrule(lr){{2-{1 + morphalign_columns}}}\\cmidrule(lr){{{2 + morphalign_columns}-{1 + morphalign_columns + len(overlap_policies)}}}",
        " & ".join(header_bottom) + " \\\\",
        "\\midrule",
    ]

    for lang in LANG_ORDER:
        if lang not in langs:
            continue
        entry = langs[lang]
        values = [float(entry["policies"][policy]["morphalign"]) for policy in policy_order]
        values.append(_pathpiece_morphalign(morphalign_cache, lang))
        formatted = _bold_best_per_row(values)
        overlap_cells = [
            _fmt_overlap(float(entry["policies"][policy]["overlap_with_reference"]))
            for policy in overlap_policies
        ]
        lines.append(" & ".join([entry["label"], *formatted, *overlap_cells]) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def main() -> None:
    data = json.loads(SRC_JSON.read_text())
    morphalign_cache = json.loads(MORPHALIGN_JSON.read_text())
    tex = build_table(data, morphalign_cache)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(tex)
    print(f"Wrote {OUT_TEX}")
    print()
    print(tex)
    print()
    print(f"Config: f={data['f']} em={data['em']} p={data['p']}")


if __name__ == "__main__":
    main()
