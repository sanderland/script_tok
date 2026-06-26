#!/usr/bin/env python3
"""Generate the main-text downstream LM results table.

Reads the canonical per-seed result files and writes
a booktabs tabular body to `results/mingram_paper/tables/table_downstream.tex`
(intended to be wrapped in `table*`).

Each tokenizer trains a depth-24 nanochat base model (matched vocab 34,685, FineWeb-en,
with matched seeds). We report:
  - val bits-per-byte (lower is better).
  - DCLM CORE (centered accuracy, higher is better).
  - rare multi-unit tokens from the shared token-usage count artifact.

Conventions:
  - Deltas are % relative to the best method for each metric, oriented so
    negative values are worse than best on both axes.
  - bpb-first column order, with rows sorted by mean bpb. Best-in-column
    bolded, second-best underlined.
"""

import statistics as st

from scipy import stats

from paper_utils.hybrid.downstream_results import (
    METHOD_LABEL,
    METHOD_ORDER,
    PAPER_SEEDS_PER_METHOD,
    load_metric_data,
    paper_methods,
)
from paper_utils.hybrid.token_usage_counts import rare_multi_unit_counts_by_method
from paper_utils.hybrid.utils import paper_table_path

OUT_TEX = paper_table_path("table_downstream.tex")


def _fmt_count(value: int) -> str:
    return str(value)


def _stars(p: float) -> str:
    if p < 0.001:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def _top2(values: dict[str, float], *, higher_better: bool) -> tuple[str | None, str | None]:
    pairs = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_better)
    best = pairs[0][0] if pairs else None
    second = pairs[1][0] if len(pairs) > 1 else None
    return best, second


def _style(method: str, best: str | None, second: str | None) -> str | None:
    return "best" if method == best else "second" if method == second else None


def _wrap(text: str, style: str | None) -> str:
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\underline{{{text}}}"
    return text


def _ordered_methods(data: dict[str, dict[str, dict[int, float]]], rare_counts: dict[str, int]) -> list[str]:
    methods = paper_methods(data)
    missing_counts = set(methods) - set(rare_counts)
    if missing_counts:
        raise ValueError(f"downstream methods missing rare-token counts: {sorted(missing_counts)}")
    return methods


def _seeds_by_method(data: dict[str, dict[str, dict[int, float]]], methods: list[str]) -> dict[str, list[int]]:
    seeds_by_method = {}
    for method in methods:
        seeds = sorted(data[method]["val_bpb"])
        if sorted(data[method]["core"]) != seeds:
            raise ValueError(f"seed mismatch for {method}")
        seeds_by_method[method] = seeds
    return seeds_by_method


def _values(data: dict[str, dict[str, dict[int, float]]], method: str, metric: str, seeds: list[int]) -> list[float]:
    return [data[method][metric][seed] for seed in seeds]


def _welch_p(
    data: dict[str, dict[str, dict[int, float]]],
    method: str,
    reference: str,
    metric: str,
    seeds_by_method: dict[str, list[int]],
) -> float:
    values = _values(data, method, metric, seeds_by_method[method])
    reference_values = _values(data, reference, metric, seeds_by_method[reference])
    return stats.ttest_ind(values, reference_values, equal_var=False).pvalue


def _delta_vs_best(value: float, best: float, *, higher_better: bool) -> float:
    if higher_better:
        return (value - best) / best * 100.0
    return (best - value) / best * 100.0


def _delta_cell(
    data: dict[str, dict[str, dict[int, float]]],
    method: str,
    metric: str,
    value: float,
    best_method: str,
    best_value: float,
    seeds_by_method: dict[str, list[int]],
    *,
    higher_better: bool,
) -> str:
    delta = _delta_vs_best(value, best_value, higher_better=higher_better)
    if method == best_method:
        return "--"
    p = _welch_p(data, method, best_method, metric, seeds_by_method)
    return f"{delta:+.2f}\\%{_stars(p)}"


def main() -> None:
    data = load_metric_data()
    rare_counts = rare_multi_unit_counts_by_method()
    methods = _ordered_methods(data, rare_counts)
    seeds_by_method = _seeds_by_method(data, methods)
    bpb_mean = {m: st.mean(_values(data, m, "val_bpb", seeds_by_method[m])) for m in methods}
    core_mean = {m: st.mean(_values(data, m, "core", seeds_by_method[m])) for m in methods}
    bpb_best, bpb_second = _top2(bpb_mean, higher_better=False)
    core_best, core_second = _top2(core_mean, higher_better=True)
    assert bpb_best is not None
    assert core_best is not None
    method_rank = {method: i for i, method in enumerate(METHOD_ORDER)}
    methods = sorted(methods, key=lambda m: (bpb_mean[m], method_rank[m]))

    lines = [
        "% Intended to be wrapped in a table* environment. Requires \\usepackage{booktabs}.",
        f"% Downstream LM eval: depth-24 nanochat, matched vocab 34,685, FineWeb-en, n={PAPER_SEEDS_PER_METHOD} lowest seeds per method.",
        "% Rows are sorted by mean validation bits/byte (lower is better).",
        "% Delta columns are signed vs the metric's best method, so negative means worse than best.",
        "% Delta stars = Welch's t-test vs the metric's best method: ** p<.001, * p<.05.",
        "% Rare tokens = multi-unit tokenizer entries with corpus count / mean total token count < 1e-7 (lower is better).",
        "\\setlength{\\tabcolsep}{6pt}",
        "\\begin{tabular}{lrlrlr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{bits/byte $\\downarrow$} & \\multicolumn{2}{c}{CORE $\\uparrow$} & rare tokens $\\downarrow$ \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-6}",
        "Method & bpb & $\\Delta$ vs Best & CORE & $\\Delta$ vs Best & count \\\\",
        "\\midrule",
    ]
    min_rare_count = min(rare_counts[m] for m in methods)
    second_rare_count = sorted(set(rare_counts[m] for m in methods))[1]
    for m in methods:
        bpb_m = bpb_mean[m]
        core_m = core_mean[m]
        bpb_cell = _wrap(f"{bpb_m:.4f}", _style(m, bpb_best, bpb_second))
        core_cell = _wrap(f"{core_m:.4f}", _style(m, core_best, core_second))
        rare_count = rare_counts[m]
        rare_count_style = "best" if rare_count == min_rare_count else "second" if rare_count == second_rare_count else None
        rare_count_cell = _wrap(_fmt_count(rare_count), rare_count_style)
        row = [
            METHOD_LABEL[m],
            bpb_cell,
            _delta_cell(
                data,
                m,
                "val_bpb",
                bpb_m,
                bpb_best,
                bpb_mean[bpb_best],
                seeds_by_method,
                higher_better=False,
            ),
            core_cell,
            _delta_cell(
                data,
                m,
                "core",
                core_m,
                core_best,
                core_mean[core_best],
                seeds_by_method,
                higher_better=True,
            ),
            rare_count_cell,
        ]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]

    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_TEX}")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
