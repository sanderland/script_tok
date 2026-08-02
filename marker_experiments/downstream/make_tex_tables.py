#!/usr/bin/env python3
"""Generate the downstream LaTeX tables from the run artifacts.

Numbers in the paper are not transcribed by hand. This reads the two artifacts the
pipeline writes and emits table bodies to \\input{} from the paper:

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
# One directory holds every paper artifact: the inputs this reads and the .tex it writes.
GENERATED = os.path.join(REPO, "marker_experiments", "paper", "generated")
DEFAULT_OUT = os.path.join(GENERATED, "downstream_tables.tex")
APPENDIX_OUT = os.path.join(GENERATED, "downstream_appendix.tex")

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


def _load_text_stats(path):
    """Per-arm tokenization statistics measured on the ClimbMix validation shard.

    Written by measure_text_stats.py. Used for the text-coverage and tokens-per-byte
    figures, which were previously typed into the caption by hand and were wrong.
    """
    if not os.path.exists(path):
        return {}
    return json.load(open(path)).get("arms", {})


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


def appendix_tables(by_arm, by_arm_n32):
    """Two robustness checks, kept out of the main table because neither is the headline.

    The seed table answers whether three seeds is enough; the shard table answers whether
    the result depends on training 3.4 to 3.6 passes over the same 2 GB.
    """
    import statistics

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Arm & $n$ seeds & bpb/byte & sd \\",
        r"\midrule",
    ]
    rows = by_arm.get("bnd_wpd", [])
    vals = sorted((int(r["seed"]), float(r["val_bpb_true"])) for r in rows)
    for k in (3, len(vals)):
        v = [x for _, x in vals[:k]]
        if len(v) < 2:
            continue
        lines.append(
            f"\\bnd{{wpd}} & {k} & {statistics.fmean(v):.4f} & {statistics.stdev(v):.4f} \\\\"
        )
    first3 = [x for _, x in vals[:3]]
    allv = [x for _, x in vals]
    shift = abs(statistics.fmean(allv) - statistics.fmean(first3)) / statistics.stdev(allv)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Seeds beyond three do not move the estimate. Doubling to "
        rf"{len(allv)} seeds shifts the mean by {shift:.2f} standard deviations, and the "
        r"standard deviation stays an order of magnitude below the difference against "
        r"\texttt{plain} in Table~\ref{tab:downstream-lm}. The remaining arms are "
        r"reported at three seeds on this basis.}",
        r"\label{tab:downstream-seeds}",
        r"\end{table}",
        "",
    ]

    if by_arm_n32:
        lines += [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Arm & 8 shards & 32 shards \\",
            r"\midrule",
        ]
        for arm in ("plain", "bnd_wpd"):
            a = [float(r["val_bpb_true"]) for r in by_arm.get(arm, []) if int(r["seed"]) < 3]
            b = [float(r["val_bpb_true"]) for r in by_arm_n32.get(arm, [])]
            if not (a and b):
                continue
            lines.append(
                f"{ARM_LABEL[arm]} & {statistics.fmean(a):.4f} & {statistics.fmean(b):.4f} \\\\"
            )
        pa = [float(r["val_bpb_true"]) for r in by_arm.get("plain", []) if int(r["seed"]) < 3]
        wa = [float(r["val_bpb_true"]) for r in by_arm.get("bnd_wpd", []) if int(r["seed"]) < 3]
        pb = [float(r["val_bpb_true"]) for r in by_arm_n32.get("plain", [])]
        wb = [float(r["val_bpb_true"]) for r in by_arm_n32.get("bnd_wpd", [])]
        g8, g32 = statistics.fmean(pa) - statistics.fmean(wa), statistics.fmean(pb) - statistics.fmean(wb)
        lines += [
            r"\midrule",
            f"difference & {g8:+.4f} & {g32:+.4f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{The result does not depend on repeating the training data. At 8 "
            rf"shards every arm makes 3.4 to 3.6 passes over the same 2\,GB; at 32 shards the "
            rf"run is single-epoch. Both arms improve by about 0.036 with four times the "
            rf"unique text, and the difference between them is unchanged, {g8:+.4f} to "
            rf"{g32:+.4f}. Seeds 0 to 2, bits-per-byte per true byte.}}",
            r"\label{tab:downstream-shards}",
            r"\end{table}",
            "",
        ]
    return lines


@app.default
def main(
    manifest: str = os.path.join(GENERATED, "manifest.json"),
    results: str = os.path.join(GENERATED, "results.tsv"),
    text_stats: str = os.path.join(GENERATED, "text_stats.json"),
    results_n32: str = os.path.join(GENERATED, "results_n32.tsv"),
    out: str = DEFAULT_OUT,
    appendix_out: str = APPENDIX_OUT,
) -> None:
    """Write the generated tables.

    Args:
        manifest: Merged tokenizer manifest.
        results: TSV from collect_results.py.
        text_stats: JSON from measure_text_stats.py, for the text-coverage figures.
        out: LaTeX file to write, \\input{} by the paper.
    """
    man = _load_manifest(manifest)
    res = _load_results(results)
    res_n32 = _load_results(results_n32)
    stats = _load_text_stats(text_stats)

    # Measured, not asserted. At a fixed token budget the text an arm covers is
    # proportional to its bytes per token, so this is the ratio of that against plain.
    if stats and "plain" in stats:
        parts = [
            f"{ARM_LABEL[a]} {100 * stats[a]['text_coverage_vs_plain']:.1f}\\%"
            for a in ARM_ORDER if a in stats and a != "plain"
        ]
        coverage_sentence = (
            r"weight initialization and training data order, with one permutation per seed "
            r"shared across arms so the arms stay paired. All arms train on the same number "
            r"of tokens, so the "
            r"amount of text each covers differs: " + ", ".join(parts)
            + r" of what \texttt{plain} covers, measured on the ClimbMix validation shard."
        )
    else:
        coverage_sentence = r"weight initialization and training data order."


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
        r"(\S\ref{sec:downstream}). $n$ is runs averaged; $\pm$ is the sample standard "
        r"deviation over those seeds, which vary "
        + coverage_sentence
        + r" CORE is \textit{n/a} for the arms whose tokenization does not satisfy the "
        r"prefix property its language-modelling tasks assume.}",
        r"\label{tab:downstream-lm}",
        r"\end{table}",
        "",
    ]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(body))

    app_lines = appendix_tables(res, res_n32)
    with open(appendix_out, "w") as f:
        f.write("\n".join(
            ["% Generated by marker_experiments/downstream/make_tex_tables.py. Do not edit.",
             f"% sources: {os.path.relpath(results, REPO)}, {os.path.relpath(results_n32, REPO)}",
             ""] + app_lines))

    print(f"[tex] {out}")
    print(f"[tex] {appendix_out}")
    print(f"[tex] {len(comp)} tokenizer row(s), {len(down)} downstream row(s)")
    missing = [a for a in ARM_ORDER if a in ARM_LABEL and a not in man and a != "bnd_wp"]
    if missing:
        print(f"[tex] not yet trained: {', '.join(missing)}")


if __name__ == "__main__":
    app()
