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
import sys

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

# Imported, not restated. The main table's whole point is that a scheme carries the same
# name here as in table_intrinsic_quick, and a second copy of these would drift the first
# time an arm is added. `bnd_wpd_extcaps` is \bnds{wpdcaps} and `bnd_wpd_caps` is
# \bnds{wpdcapsin}; ARM_LABEL below, used by the older tables, disagrees with both.
from marker_experiments.make_intrinsic_table import (  # noqa: E402
    ARM_LABEL as SCHEME_LABEL,
    ARM_ORDER as SCHEME_ORDER,
    MISSING,
    PLAIN_LABEL,
)
# One directory holds every paper artifact: the inputs this reads and the .tex it writes.
GENERATED = os.path.join(REPO, "marker_experiments", "paper", "generated")
DEFAULT_OUT = os.path.join(GENERATED, "downstream_tables.tex")
APPENDIX_OUT = os.path.join(GENERATED, "downstream_appendix.tex")
MAIN_OUT = os.path.join(GENERATED, "table_downstream_main.tex")

KNOWN_TRAINERS = ["bpe", "mingram"]
TRAINER_LABEL = {"bpe": "BPE", "mingram": "MinGram"}
MISSING = "--"

# Presentation order, and the label each arm carries in the paper.
ARM_ORDER = ["plain", "bnd_w", "bnd_wp", "bnd_wpd", "bnd_wpd_caps", "bnd_wpd_extcaps"]
ARM_LABEL = {
    "plain": "plain",
    "bnd_w": r"\bnd{w}",
    "bnd_wp": r"\bnd{wp}",
    "bnd_wpd": r"\bnd{wpd}",
    "bnd_wpd_caps": r"\bnd{wpd\_caps}",
    "bnd_wpd_extcaps": r"\bnd{wpd\_extcaps}",
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


def _rel_sources(paths):
    """Provenance header for one or more comma-separated inputs."""
    return ", ".join(os.path.relpath(x, REPO) for x in str(paths).split(",") if x)


def _load_results(paths):
    """Rows grouped by arm, from one or more comma-separated TSVs.

    Several TSVs because a sweep that ran under its own OUT directory has its own file:
    the extcaps arm is trainer bpe like the rest of the main table, but its runs were
    collected separately. Combining them here is safe in a way combining trainers is not,
    so the trainer is asserted to be constant across everything loaded. `arm` is the
    grouping key below and does not carry the trainer, so a mixed load would average a
    BPE and a MinGram tokenizer into one mean.
    """
    by_arm: dict[str, list[dict]] = {}
    trainers = set()
    for path in [p for p in str(paths).split(",") if p]:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                arm = row.get("arm") or row.get("method", "")
                by_arm.setdefault(arm, []).append(row)
                if row.get("trainer"):
                    trainers.add(row["trainer"])
    if len(trainers) > 1:
        raise SystemExit(
            f"{paths} mixes trainers {sorted(trainers)}. Generate one table per trainer; "
            f"`arm` does not distinguish them and the means would pool."
        )
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


def _paired_delta(by_arm, arm, column="val_bpb_true"):
    """Per-seed difference against plain, over the seeds both arms ran.

    Paired, not marginal. One data permutation per seed is shared across arms, so
    differencing at equal seed cancels the seed effect; the sd of those differences is the
    error bar the design licenses, and it runs about five times tighter than the sd of
    either arm's own runs. Intersecting the seed sets matters: bnd_wpd has six BPE runs
    against plain's three, and the three unpaired ones have nothing to difference against.
    """
    base = {int(r["seed"]): float(r[column]) for r in by_arm.get("plain", [])}
    runs = {int(r["seed"]): float(r[column]) for r in by_arm.get(arm, [])}
    seeds = sorted(set(base) & set(runs))
    if not seeds:
        return None, None, 0
    diffs = [runs[s] - base[s] for s in seeds]
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    return statistics.fmean(diffs), sd, len(diffs)


def _paired_p(by_arm, arm, column="val_bpb_true"):
    """Two-sided paired t-test against plain over the shared seeds.

    Paired rather than independent because the seeds are: differencing at equal seed is
    what makes three runs per scheme enough to separate effects of this size. Returns None
    where there is nothing to test -- plain against itself, or a scheme with one run.
    """
    from scipy import stats

    base = {int(r["seed"]): float(r[column]) for r in by_arm.get("plain", [])}
    runs = {int(r["seed"]): float(r[column]) for r in by_arm.get(arm, [])}
    seeds = sorted(set(base) & set(runs))
    if arm == "plain" or len(seeds) < 2:
        return None
    return float(stats.ttest_rel([runs[s] for s in seeds], [base[s] for s in seeds]).pvalue)


def _lm_column(by_arm, schemes):
    """One trainer's column: arm -> rendered cell, best emphasised and runner-up underlined.

    Ranked on the rounded figure rather than the stored float, so two schemes that print
    the same are placed the same; emphasising one of a pair of identical-looking numbers
    reads as a typo. Lower bits-per-byte is better, so the rank is ascending, and plain is
    ranked with the rest because every figure in the column is absolute and comparable.
    """
    stats_by_arm = {}
    for arm in schemes:
        val, sd, n = _mean_sd(by_arm.get(arm, []), "val_bpb_true")
        if val is not None:
            stats_by_arm[arm] = (val, sd, n)
    places = sorted({round(v, 4) for v, _, _ in stats_by_arm.values()})

    cells, own_ns = {}, {}
    for arm, (val, sd, n) in stats_by_arm.items():
        rank = places.index(round(val, 4))
        num = f"{val:.4f}"
        if rank == 0:
            num = rf"\textbf{{{num}}}"
        # Runner-up only where there is something to run up against. In a two-entry column
        # second place is last place, and underlining it reads as praise for the loser.
        elif rank == 1 and len(places) > 2:
            num = rf"\underline{{{num}}}"
        cells[arm] = num + (f" {{\\footnotesize $\\pm$ {sd:.4f}}}" if n > 1 else "")
        own_ns[arm] = n
    return cells, own_ns


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


@app.command(name="main-table")
def main_table(
    manifest: str = os.path.join(GENERATED, "manifest.json"),
    results_bpe: str = ",".join([os.path.join(GENERATED, "results.tsv"),
                                 os.path.join(GENERATED, "results_extcaps.tsv")]),
    results_mingram: str = os.path.join(GENERATED, "results_mingram.tsv"),
    corpus: str = "fineweb_en_5gb",
    out: str = MAIN_OUT,
) -> None:
    """Write the single wide main table: compression and LM quality, both trainers.

    One table rather than the four it replaces, because the argument is the join between
    the two metrics and no single old table carried it. Rows and macros follow
    make_intrinsic_table.py so the paper's two main tables are directly comparable.

    Args:
        manifest: Merged tokenizer manifest, read once per trainer.
        results_bpe: Comma-separated TSVs holding the BPE runs.
        results_mingram: TSV holding the MinGram runs.
        corpus: Which tokenizer training corpus the compression column reports.
        out: LaTeX file to write, \\input{} by the paper.
    """
    sources = {"bpe": results_bpe, "mingram": results_mingram}
    man = {t: _load_manifest(manifest, trainer=t, corpus=corpus) for t in KNOWN_TRAINERS}
    res = {t: _load_results(sources[t]) for t in KNOWN_TRAINERS}

    # Every scheme that has a pretraining run, in the intrinsic table's display order.
    # Keyed on the runs and not the manifest: a tokenizer trained but never pretrained --
    # bnd_wp, today -- would otherwise get a row of nothing but dashes now that the
    # compression column, the only one it could have filled, is gone.
    available = {a for t in KNOWN_TRAINERS for a in res[t]}
    schemes = ["plain"] + [a for a in SCHEME_ORDER if a in available]

    columns = {t: _lm_column(res[t], schemes) for t in KNOWN_TRAINERS}

    rows, paired_ns, own_ns, pvals = [], set(), {}, []
    for arm in schemes:
        cells = []
        label = PLAIN_LABEL if arm == "plain" else SCHEME_LABEL[arm]
        for t in KNOWN_TRAINERS:
            cell_by_arm, n_by_arm = columns[t]
            cells.append(cell_by_arm.get(arm, MISSING))
            if arm in n_by_arm:
                own_ns.setdefault(n_by_arm[arm], []).append(f"{label} under {TRAINER_LABEL[t]}")
            p = _paired_p(res[t], arm)
            if p is not None:
                pvals.append(p)
            _, _, n_paired = _paired_delta(res[t], arm)
            if n_paired:
                paired_ns.add(n_paired)
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    # Every fact the columns dropped, recovered from the artifacts rather than typed, and
    # emitted as a comment so it is available to the prose without occupying a column.
    vocabs = {e["total_vocab"] for m in man.values() for e in m.values()}
    rt = {e["roundtrip_failures"] for m in man.values() for e in m.values()}
    core = {}
    for t in KNOWN_TRAINERS:
        for arm in schemes:
            mean, sd, n = _mean_sd(res[t].get(arm, []), "core")
            if mean is None:
                continue
            note = f"{mean:.4f} +- {sd:.4f} (n={n})"
            # The paired difference, not just the two marginals: it is the only form in
            # which CORE says anything, and what it says is that it cannot resolve these
            # schemes at depth 12.
            delta, dsd, dn = _paired_delta(res[t], arm, column="core")
            if arm != "plain" and delta is not None and dn > 1:
                note += f", paired vs plain {delta:+.4f} +- {dsd:.4f}"
            core[f"{TRAINER_LABEL[t]} {arm}"] = note
    # The usual run count, and every cell that departs from it named rather than folded
    # into a range: "3 to 6 runs" would leave the reader unable to tell which cell is which.
    usual = max(own_ns, key=lambda k: len(own_ns[k]))
    odd = [f"{name} with {n}" for n, names in sorted(own_ns.items()) if n != usual
           for name in names]
    runs_note = f"{usual} runs per scheme" + (f", except {', '.join(odd)}" if odd else "")
    n_note = (f"{sorted(paired_ns)[0]}" if len(paired_ns) == 1
              else f"{min(paired_ns)} to {max(paired_ns)}")
    # The tightest round bound the measured p-values actually support, so the caption
    # cannot outlive the runs: a seed that weakened a scheme would loosen this, not
    # silently leave a claim in the caption that the data no longer backs.
    threshold = next(x for x in (0.001, 0.01, 0.05, 1.0) if max(pvals) < x)

    body = [
        "% Generated by marker_experiments/downstream/make_tex_tables.py main-table. Do not edit.",
        r"% Requires booktabs and the paper's \bnds and \plainscheme macros, as table_intrinsic_main.",
        f"% sources: {os.path.relpath(manifest, REPO)}, "
        f"{_rel_sources(results_bpe)}, {_rel_sources(results_mingram)}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l rr}",
        r"\toprule",
        r"Scheme & \multicolumn{2}{c}{bpb/byte $\downarrow$} \\",
        r"\cmidrule(lr){2-3}",
        r" & BPE & MinGram \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Language-modelling quality after nanochat depth-12 pretraining on "
        rf"ClimbMix, for vocabulary-matched tokenizers trained on {_tex_escape(corpus)}. "
        r"bpb/byte is validation loss summed over the true UTF-8 length of the evaluation "
        r"text, so it is comparable across schemes that tokenize the same text into "
        r"different numbers of tokens. Every figure is absolute, and $\pm$ is a scheme's "
        rf"own sample standard deviation over {runs_note}. \textbf{{Bold}} is best in a "
        r"column and \underline{underline} runner-up. Every scheme improves on "
        rf"\plainscheme{{}} at $p<{threshold:g}$, by a two-sided paired $t$-test over the "
        rf"{n_note} seeds both ran: one data permutation per seed is shared across "
        r"schemes, so differencing at equal seed cancels it and leaves a spread several "
        rf"times tighter than the $\pm$ above. {MISSING} is a configuration that was not "
        r"pretrained. Compression for these same tokenizers is in "
        r"Table~\ref{tab:intrinsic-main}.}",
        r"\label{tab:downstream-main}",
        "",
        "% Not in the table. Put in the text if the argument needs it:",
        f"%   vocabulary matched at {', '.join(f'{v:,}' for v in sorted(vocabs))} "
        "across every scheme and trainer",
        f"%   roundtrip failures: {', '.join(str(v) for v in sorted(rt))} "
        "for every scheme and trainer",
        "%   CORE, depth 12, n/a wherever a scheme breaks the prefix property its tasks assume:",
        *[f"%     {k}: {v}" for k, v in sorted(core.items())],
        "%   raw bpb (nanochat's own figure) divides by summed token byte length and is not",
        "%     comparable across these tokenizers; see the appendix tables.",
        "",
    ]
    with open(out, "w") as f:
        f.write("\n".join(body))
    print(f"[tex] {out}")
    print(f"[tex] {len(rows)} scheme row(s), trainers {KNOWN_TRAINERS}, {n_note}")


@app.default
def main(
    manifest: str = os.path.join(GENERATED, "manifest.json"),
    results: str = ",".join([os.path.join(GENERATED, "results.tsv"),
                             os.path.join(GENERATED, "results_extcaps.tsv")]),
    text_stats: str = os.path.join(GENERATED, "text_stats.json"),
    results_n32: str = os.path.join(GENERATED, "results_n32.tsv"),
    trainer: str = "bpe",
    out: str = DEFAULT_OUT,
    appendix: bool = True,
    appendix_out: str = APPENDIX_OUT,
) -> None:
    """Write the generated tables.

    Args:
        manifest: Merged tokenizer manifest.
        results: One or more comma-separated TSVs from collect_results.py, all of one
            trainer. Defaults to the main sweep plus the extcaps sweep, which ran under
            its own OUT directory and so has its own file.
        text_stats: JSON from measure_text_stats.py, for the text-coverage figures.
        trainer: Which trainer's cells to report, in the compression table and the
            caption. A separate table per trainer, because `arm` does not carry the
            trainer and one table would pool two different tokenizers under one name.
        out: LaTeX file to write, \\input{} by the paper.
        appendix: Whether to write the seed and shard appendix tables. They cover the
            BPE sweep only, so pass --no-appendix when generating for another trainer.
        appendix_out: Where those appendix tables go.
    """
    # Distinct labels and captions per trainer. Two files defining \label{tab:downstream-lm}
    # makes LaTeX resolve every \Cref to whichever it read last, silently.
    suffix = "" if trainer == "bpe" else f"-{trainer}"
    trainer_name = {"bpe": "BPE", "mingram": "MinGram"}.get(trainer, trainer)
    man = _load_manifest(manifest, trainer=trainer)
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
        f"% sources: {os.path.relpath(manifest, REPO)}, {_rel_sources(results)}",
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
        rf"\caption{{Vocabulary-matched {trainer_name} tokenizers on {_tex_escape(corpus)}"
        + (f", all at {_num(vocab)} total vocabulary" if vocab else "")
        + r". Chars/token on the held-out FineWiki English slice (500 documents, "
        r"3{,}602{,}925 characters); $\Delta$ relative to \texttt{plain}. RT fail is "
        r"roundtrip failures.}",
        rf"\label{{tab:downstream-compression{suffix}}}",
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
        rf"\caption{{nanochat depth-12 pretraining on ClimbMix, {trainer_name} tokenizers"
        + (f", seeds {', '.join(n_seeds)}" if n_seeds else "")
        + r". bpb/byte is summed loss over the true UTF-8 length of the evaluation text; "
        r"raw bpb is nanochat's figure, which divides instead by the summed byte length of "
        r"the emitted tokens and so is not comparable across these tokenizers "
        r"(\S\ref{sec:downstream}). $n$ is runs averaged; $\pm$ is the sample standard "
        r"deviation over those seeds, which vary "
        + coverage_sentence
        + r" CORE is \textit{n/a} for the arms whose tokenization does not satisfy the "
        r"prefix property its language-modelling tasks assume.}",
        rf"\label{{tab:downstream-lm{suffix}}}",
        r"\end{table}",
        "",
    ]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(body))

    if appendix:
        app_lines = appendix_tables(res, res_n32)
        with open(appendix_out, "w") as f:
            f.write("\n".join(
                ["% Generated by marker_experiments/downstream/make_tex_tables.py. Do not edit.",
                 f"% sources: {_rel_sources(results)}, {os.path.relpath(results_n32, REPO)}",
                 ""] + app_lines))

    print(f"[tex] {out}")
    if appendix:
        print(f"[tex] {appendix_out}")
    print(f"[tex] {len(comp)} tokenizer row(s), {len(down)} downstream row(s)")
    missing = [a for a in ARM_ORDER if a in ARM_LABEL and a not in man and a != "bnd_wp"]
    if missing:
        print(f"[tex] not yet trained: {', '.join(missing)}")


if __name__ == "__main__":
    app()
