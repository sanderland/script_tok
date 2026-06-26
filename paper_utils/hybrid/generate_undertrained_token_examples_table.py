#!/usr/bin/env python3
"""Generate an appendix table of least-common multi-unit token examples."""

from paper_utils.hybrid.downstream_results import METHOD_LABEL, METHOD_ORDER
from paper_utils.hybrid.token_usage_counts import least_common_tokens, rare_multi_unit_counts_by_method, similar_neighbour_order
from paper_utils.hybrid.utils import paper_table_path

N_EXAMPLES = 10
SECOND_PANEL_METHODS = ["mingram_f1.15", "mingram_pp_f8", "pathpiece", "convextok"]
OUT_TEX = paper_table_path("table_undertrained_token_examples.tex", appendix=True)


def _escape_latex(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == " ":
            out.append(r"\textvisiblespace{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch in {"_", "%", "&", "#", "$", "{", "}"}:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u052f" for ch in text)


def _format_token(token: str, count: int) -> str:
    escaped = _escape_latex(token)
    if _has_cyrillic(token):
        return rf"{{\fontencoding{{T2A}}\ttfamily\selectfont {escaped}}}~({count})"
    return rf"\texttt{{{escaped}}}~({count})"


def _cell_by_rank(rows) -> dict[tuple[int, str], str]:
    return {
        (row["rank"], row["method"]): _format_token(row["token_str"], row["count"])
        for row in rows.rows(named=True)
    }


def _tabular(methods: list[str], cells: dict[tuple[int, str], str], rare_counts: dict[str, int]) -> list[str]:
    headers = ["rank"] + [METHOD_LABEL[method] for method in methods]
    lines = [
        "\\begin{tabular}{r" + "l" * len(methods) + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for rank in range(1, N_EXAMPLES + 1):
        line = [str(rank)] + [cells[(rank, method)] for method in methods]
        lines.append(" & ".join(line) + " \\\\")
    lines += [
        "\\midrule",
        "total rare & " + " & ".join(str(rare_counts[method]) for method in methods) + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return lines


def build_table() -> str:
    rows = least_common_tokens(n=N_EXAMPLES)
    methods = [method for method in METHOD_ORDER if method in rows["method"].unique().to_list()]
    methods = similar_neighbour_order(rows, methods)
    cells = _cell_by_rank(rows)
    rare_counts = rare_multi_unit_counts_by_method()
    second_panel = [method for method in SECOND_PANEL_METHODS if method in methods]
    first_panel = [method for method in methods if method not in second_panel]
    panels = [first_panel, second_panel]
    lines = [
        "% Intended to be wrapped in \\begin{table*}...\\end{table*}. Requires \\usepackage{booktabs}.",
        "% Cells show token string and corpus count among multi-unit tokens.",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    for i, panel_methods in enumerate(panels):
        if i > 0:
            lines += ["", "\\medskip", ""]
        lines.extend(_tabular(panel_methods, cells, rare_counts))
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_TEX.write_text(build_table())
    print(f"wrote {OUT_TEX}")
    print(OUT_TEX.read_text())


if __name__ == "__main__":
    main()
