"""Shared downstream result loading for paper table generators."""

import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
DOWNSTREAM_RESULTS_TSV = REPO_ROOT / "results" / "downstream" / "d24_master_results.tsv"
PAPER_SEEDS_PER_METHOD = 20

SUMMARY_COLUMNS = {"method", "seed", "val_bpb", "train_bpb", "core"}
METHOD_ORDER = [
    "bpe",
    "unigram_default",
    "unigram_fsp",
    "bpe_init_f1.15",
    "fsp_bpe_init_f1.15",
    "mingram_f1.15",
    "mingram_pp_f8",
    "pathpiece",
    "convextok",
]
METHOD_LABEL = {
    "bpe": "BPE",
    "unigram_default": "Unigram",
    "unigram_fsp": "FSP",
    "bpe_init_f1.15": "Unigram\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "fsp_bpe_init_f1.15": "FSP\\hspace{0pt}-BPE\\hspace{0pt}-Init",
    "mingram_f1.15": "MinGram",
    "mingram_pp_f8": "\\mingrampp{}",
    "pathpiece": "PathPiece\\hspace{0pt}-BPE",
    "convextok": "ConvexTok",
}
METHOD_SHORT_LABEL = {
    "bpe": "BPE",
    "unigram_default": "Uni.",
    "unigram_fsp": "FSP",
    "bpe_init_f1.15": "U-BPE",
    "fsp_bpe_init_f1.15": "F-BPE",
    "mingram_f1.15": "Min",
    "mingram_pp_f8": "Min-MI",
    "pathpiece": "Path",
    "convextok": "Conv",
}
GLITCH_KEY = {
    "mingram_pp_f8": "mingram_pp",
}
TASK_LABEL = {
    "hellaswag_zeroshot": "HellaSwag-0",
    "jeopardy": "Jeopardy",
    "bigbench_qa_wikidata": "Wikidata QA",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Chal.",
    "copa": "COPA",
    "commonsense_qa": "CSQA",
    "piqa": "PIQA",
    "openbook_qa": "OpenBookQA",
    "lambada_openai": "LAMBADA",
    "hellaswag": "HellaSwag",
    "winograd": "Winograd",
    "winogrande": "WinoGrande",
    "bigbench_dyck_languages": "Dyck Lang.",
    "agi_eval_lsat_ar": "LSAT-AR",
    "bigbench_cs_algorithms": "CS Alg.",
    "bigbench_operators": "Operators",
    "bigbench_repeat_copy_logic": "Repeat Copy",
    "squad": "SQuAD",
    "coqa": "CoQA",
    "boolq": "BoolQ",
    "bigbench_language_identification": "Lang. ID",
}


def load_rows(
    path: Path = DOWNSTREAM_RESULTS_TSV,
    *,
    methods: list[str] | None = METHOD_ORDER,
    seeds_per_method: int | None = PAPER_SEEDS_PER_METHOD,
) -> list[dict[str, str]]:
    with open(path) as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))

    method_filter = set(methods) if methods is not None else None
    if method_filter is not None:
        rows = [row for row in rows if row["method"] in method_filter]

    if seeds_per_method is None:
        return rows

    rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_method[row["method"]].append(row)

    trimmed = []
    method_order = methods if methods is not None else sorted(rows_by_method)
    for method in method_order:
        if method not in rows_by_method:
            continue
        method_rows = sorted(rows_by_method[method], key=lambda row: int(row["seed"]))
        if len(method_rows) < seeds_per_method:
            raise ValueError(f"{method} has {len(method_rows)} downstream seeds, need {seeds_per_method}")
        trimmed.extend(method_rows[:seeds_per_method])
    return trimmed


def task_columns(rows: list[dict[str, str]]) -> list[str]:
    return [column for column in rows[0] if column not in SUMMARY_COLUMNS]


def load_metric_data(path: Path = DOWNSTREAM_RESULTS_TSV) -> dict[str, dict[str, dict[int, float]]]:
    data: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in load_rows(path):
        method = row["method"]
        seed = int(row["seed"])
        for column, value in row.items():
            if column in {"method", "seed"}:
                continue
            data[method][column][seed] = float(value)
    return data


def paper_methods(data: dict[str, dict[str, dict[int, float]]]) -> list[str]:
    return [method for method in METHOD_ORDER if method in data]
