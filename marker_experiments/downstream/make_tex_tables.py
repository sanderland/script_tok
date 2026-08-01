#!/usr/bin/env python3
"""Generate the downstream LaTeX tables from the run artifacts.

Numbers in the paper are not transcribed by hand. This reads the two artifacts the
pipeline writes and emits the table bodies that acl_latex.tex \\input{}s:

    manifest.json   one entry per trained tokenizer (train_matched.py / merge_manifests.py)
    results.tsv     one row per downstream run    (collect_results.py)

Re-run it whenever another arm or seed lands; it only emits rows for artifacts that
exist, so a partial sweep produces a partial table rather than a fabricated one, and
records in the caption how many runs each cell averages.

    uv run python marker_experiments/downstream/make_tex_tables.py
"""

import csv
import json
import os
import statistics

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(REPO, "marker_experiments", "paper", "generated", "downstream_tables.tex")

# Presentation order, and the label each arm carries in the paper.
ARM_ORDER = ["plain", "bnd_w", "bnd_wp", "bnd_wpd", "bnd_wpd_caps"]
ARM_LABEL = {
    "plain": "plain",
    "bnd_w": r"\bnd{w}",
    "bnd_wp": r"\bnd{wp}",
    "bnd_wpd": r"\bnd{wpd}",
    "bnd_wpd_caps": r"\bnd{wpd\_caps}",
}
# Arms whose tokenization keeps CORE's prefix property; see core_prefix_check.py.
CORE_SAFE = {"plain", "bnd_w"}

app = cyclopts.App()


def _load_manifest(path, trainer="bpe", corpus="fineweb_en_5gb"):
    """Arm -> manifest entry, restricted to one trainer and corpus.

    Manifest keys are {corpus}_{arm}_{trainer}_v{vocab}, so keying on `arm` alone lets a
    throwaway `tiny_*` entry, which train_matched.py writes into the same file when run
    with --text-file, overwrite the real one and be reported under the real caption.
    """
    if not os.path.exists(path):
        return {}
    return {
        v["arm"]: v
        for v in json.load(open(path)).values()
        if v.get("trainer") == trainer and v.get("corpus") == corpus
    }


def _load_results(path):
    """results.tsv rows grouped by arm."""
    if not os.path.exists(path):
        return {}
    by_arm: dict[str, list[dict]] = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            arm = row.get("arm") or row.get("method", "")
            by_arm.setdefault(arm, []).append(row)
    return by_arm


def _num(value):
    """Thousands separator LaTeX keeps at the right width inside math-free text."""
    return f"{value:,}".replace(",", "{,}")


def _tex_escape(text):
    return text.replace("_", r"\_")


def _mean_sd(rows, column):
    vals = [float(r[column]) for r in rows if r.get(column) not in (None, "", "None")]
    if not vals:
        return None, None, 0
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.fmean(vals), sd, len(vals)


def compression_table(manifest):
    """Chars/token at the matched vocabulary, as a relative change against plain."""
    base = manifest.get("plain", {}).get("eval_chars_per_token")
    lines = []
    for arm in ARM_ORDER:
        info = manifest.get(arm)
        if not info:
            continue
        cpt = info["eval_chars_per_token"]
        if arm == "plain" or base is None:
            rel = "---"
        else:
            pct = 100.0 * (cpt - base) / base
            rel = f"${pct:+.2f}$"
        lines.append(
            f"{ARM_LABEL[arm]} & {_num(info['total_vocab'])} & {cpt:.4f} & {rel} & "
            f"{info['roundtrip_failures']} \\\\"
        )
    return lines


def downstream_table(by_arm):
    """Validation bits-per-byte, and CORE where the arm admits it."""
    lines = []
    for arm in ARM_ORDER:
        rows = by_arm.get(arm)
        if not rows:
            continue
        # val_bpb_true, not val_bpb: the raw column divides by summed token byte length,
        # which undercounts every arm that elides a character between two tokens, so it is
        # not comparable across these tokenizers. See runner.measure_byte_factor.
        val, val_sd, n = _mean_sd(rows, "val_bpb_true")
        raw, _, _ = _mean_sd(rows, "val_bpb")
        core, core_sd, n_core = _mean_sd(rows, "core")
        val_cell = "---" if val is None else f"{val:.4f}"
        if val is not None and n > 1:
            val_cell += f" {{\\footnotesize $\\pm$ {val_sd:.4f}}}"
        # Driven by the data, not by a hardcoded arm list: CORE_SAFE_ARMS is settable at
        # submit time, and a hardcoded list would print n/a over a number we measured.
        if core is None:
            core_cell = r"\textit{n/a}" if arm not in CORE_SAFE else "---"
        else:
            core_cell = f"{core:.4f}"
            if n_core > 1:
                core_cell += f" {{\\footnotesize $\\pm$ {core_sd:.4f}}}"
        raw_cell = "---" if raw is None else f"{raw:.4f}"
        lines.append(f"{ARM_LABEL[arm]} & {n} & {val_cell} & {raw_cell} & {core_cell} \\\\")
    return lines


@app.default
def main(
    manifest: str = os.path.join(HERE, "manifest.json"),
    results: str = os.path.join(REPO, "results", "marker_downstream", "results.tsv"),
    out: str = DEFAULT_OUT,
) -> None:
    """Write the generated tables.

    Args:
        manifest: Merged tokenizer manifest.
        results: TSV from collect_results.py.
        out: LaTeX file to write, \\input{} by the paper.
    """
    man = _load_manifest(manifest)
    res = _load_results(results)

    comp = compression_table(man)
    down = downstream_table(res)

    corpus = next(iter(man.values()), {}).get("corpus", "fineweb\\_en\\_5gb")
    vocab = next(iter(man.values()), {}).get("total_vocab")
    n_seeds = sorted({r.get("seed") for rows in res.values() for r in rows} - {None, ""})

    body = [
        "% Generated by marker_experiments/downstream/make_tex_tables.py. Do not edit.",
        f"% sources: {os.path.relpath(manifest, REPO)}, {os.path.relpath(results, REPO)}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Arm & Vocab & chars/token & $\Delta$ (\%) & RT fail \\",
        r"\midrule",
    ]
    body += comp or [r"\multicolumn{5}{c}{\textit{no tokenizers trained yet}} \\"]
    body += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Vocabulary-matched tokenizers on {_tex_escape(corpus)}"
        + (f", all at {_num(vocab)} total vocabulary" if vocab else "")
        + r". Chars/token on the held-out FineWiki English slice (500 documents, "
        r"3{,}602{,}925 characters); $\Delta$ relative to \texttt{plain}. RT fail is "
        r"roundtrip failures.}",
        r"\label{tab:downstream-compression}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Arm & $n$ & bpb/byte & raw bpb & CORE \\",
        r"\midrule",
    ]
    body += down or [r"\multicolumn{5}{c}{\textit{no downstream runs finished yet}} \\"]
    body += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{nanochat depth-12 pretraining on ClimbMix"
        + (f", seeds {', '.join(n_seeds)}" if n_seeds else "")
        + r". bpb/byte is summed loss over the true UTF-8 length of the evaluation text; "
        r"raw bpb is nanochat's figure, which divides instead by the summed byte length of "
        r"the emitted tokens and so is not comparable across these tokenizers "
        r"(\S\ref{sec:downstream}). $n$ is runs averaged. $\pm$ spans the seeds, which vary "
        r"weight initialization only: the data order is identical across seeds, so this "
        r"understates run-to-run variance. All arms train on the same number of tokens, so "
        r"a worse-compressing arm sees less text; \bnd{w} reads 95.1\% of what \texttt{plain} "
        r"does. CORE is \textit{n/a} for the arms whose tokenization does not satisfy the "
        r"prefix property its language-modelling tasks assume.}",
        r"\label{tab:downstream-lm}",
        r"\end{table}",
        "",
    ]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(body))

    print(f"[tex] {out}")
    print(f"[tex] {len(comp)} tokenizer row(s), {len(down)} downstream row(s)")
    missing = [a for a in ARM_ORDER if a in ARM_LABEL and a not in man and a != "bnd_wp"]
    if missing:
        print(f"[tex] not yet trained: {', '.join(missing)}")


if __name__ == "__main__":
    app()
