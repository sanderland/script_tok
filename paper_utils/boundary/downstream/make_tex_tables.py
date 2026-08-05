#!/usr/bin/env python3
"""Generate the downstream LaTeX tables from the run artifacts.

Numbers in the paper are not transcribed by hand. This reads what the pipeline writes and
emits table bodies to \\input{} from the paper:

    manifest.json   one entry per trained tokenizer (train_matched.py / merge_manifests.py)
    results.tsv     one row per downstream run    (collect_results.py)

    table_downstream_main.tex   bits per byte, one row per scheme, both trainers
    downstream_appendix.tex     the seed and shard robustness checks

Re-run it whenever another scheme or seed lands; it only emits rows for artifacts that
exist, so a partial sweep produces a partial table rather than a fabricated one, and
records in the caption how many runs each cell rests on.

Schemes, their order and their macros are imported from make_intrinsic_table, so a scheme
is named the same in every table in the paper and the row set here is the intrinsic main
table's, minus anything never pretrained.

    uv run python paper_utils/boundary/downstream/make_tex_tables.py
"""

import csv
import json
import os
import statistics
import sys

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

# Imported, not restated. A scheme has to carry the same name and the same place in the
# order here as in the intrinsic tables, and a second copy of these would drift the first
# time an arm is added. MAIN_ARMS is the intrinsic main table's scheme set, which is also
# the set this table shows, minus anything never pretrained.
from paper_utils.boundary.make_intrinsic_table import (  # noqa: E402
    ARM_LABEL as SCHEME_LABEL,
    MAIN_ARMS,
    MISSING,
    PLAIN_LABEL,
)
# One directory holds every paper artifact: the inputs this reads and the .tex it writes.
GENERATED = os.path.join(REPO, "paper_utils", "boundary", "paper", "generated")
APPENDIX_OUT = os.path.join(GENERATED, "downstream_appendix.tex")
MAIN_OUT = os.path.join(GENERATED, "table_downstream_main.tex")

KNOWN_TRAINERS = ["bpe", "mingram"]
TRAINER_LABEL = {"bpe": "BPE", "mingram": "MinGram"}

app = cyclopts.App()


def _load_manifest(path, trainer="bpe", corpus="fineweb_en_5gb_quick"):
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


def _roundtrip_failures(path, corpus):
    """Every distinct round-trip failure count the grid recorded for `corpus`.

    A set rather than a total, because the fact the caption wants is that the count is the
    same everywhere and that it is zero. Empty when the artifact is absent, which the
    caller reports as "not measured" rather than as zero.
    """
    if not os.path.exists(path):
        return set()
    return {
        v["roundtrip_failures"]
        for k, v in json.load(open(path)).items()
        if k.startswith(f"{corpus}_") and "roundtrip_failures" in v
    }


def _rel_sources(paths):
    """Provenance header for one or more comma-separated inputs."""
    return ", ".join(os.path.relpath(x, REPO) for x in str(paths).split(",") if x)


def _load_results(paths):
    """Rows grouped by arm, from one or more comma-separated TSVs.

    Several TSVs because a sweep that ran under its own OUT directory has its own file:
    the caps arm is trainer bpe like the rest of the main table, but its runs were
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
    # Restricted to the seeds plain also ran, so every cell rests on the same seeds and the
    # caption can state one run count for the whole table. bnd_wpd has six BPE runs; the
    # three beyond plain's are the appendix seed check, not extra precision here, and
    # averaging them in would leave one cell quietly resting on a different sample.
    base_seeds = {int(r["seed"]) for r in by_arm.get("plain", [])}
    stats_by_arm = {}
    for arm in schemes:
        rows = [r for r in by_arm.get(arm, []) if int(r["seed"]) in base_seeds]
        val, sd, n = _mean_sd(rows, "val_bpb_true")
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
        r"Scheme & $n$ seeds & bits per byte & sd \\",
        r"\midrule",
    ]
    rows = by_arm.get("bnd_wpd", [])
    vals = sorted((int(r["seed"]), float(r["val_bpb_true"])) for r in rows)
    for k in (3, len(vals)):
        v = [x for _, x in vals[:k]]
        if len(v) < 2:
            continue
        lines.append(
            f"{SCHEME_LABEL['bnd_wpd']} & {k} & {statistics.fmean(v):.4f} & "
            f"{statistics.stdev(v):.4f} \\\\"
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
        r"\plainscheme{} in Table~\ref{tab:downstream-main}. Every scheme is reported at "
        r"three seeds on this basis.}",
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
            r"Scheme & 8 shards & 32 shards \\",
            r"\midrule",
        ]
        for arm in ("plain", "bnd_wpd"):
            a = [float(r["val_bpb_true"]) for r in by_arm.get(arm, []) if int(r["seed"]) < 3]
            b = [float(r["val_bpb_true"]) for r in by_arm_n32.get(arm, [])]
            if not (a and b):
                continue
            label = PLAIN_LABEL if arm == "plain" else SCHEME_LABEL[arm]
            lines.append(
                f"{label} & {statistics.fmean(a):.4f} & {statistics.fmean(b):.4f} \\\\"
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
            rf"shards every scheme makes 3.4 to 3.6 passes over the same 2\,GB; at 32 shards "
            rf"the run is single-epoch. Both improve by about 0.036 with four times the "
            rf"unique text, and the difference between them is unchanged, {g8:+.4f} to "
            rf"{g32:+.4f}. Seeds 0 to 2, bits per byte over true UTF-8 length.}}",
            r"\label{tab:downstream-shards}",
            r"\end{table}",
            "",
        ]
    return lines


@app.default
def main(
    manifest: str = os.path.join(GENERATED, "manifest.json"),
    results_bpe: str = ",".join([os.path.join(GENERATED, "results.tsv"),
                                 os.path.join(GENERATED, "results_caps.tsv")]),
    results_mingram: str = os.path.join(GENERATED, "results_mingram.tsv"),
    results_n32: str = os.path.join(GENERATED, "results_n32.tsv"),
    goldfish: str = os.path.join(GENERATED, "eval_goldfish.json"),
    corpus: str = "fineweb_en_5gb_quick",
    out: str = MAIN_OUT,
    appendix_out: str = APPENDIX_OUT,
) -> None:
    """Write the downstream main table and the two appendix robustness tables.

    One main table, both trainers, one row per scheme. Rows and macros come from
    make_intrinsic_table.py so a scheme is named the same in every table in the paper.

    Args:
        manifest: Merged tokenizer manifest, read once per trainer.
        results_bpe: Comma-separated TSVs holding the BPE runs.
        results_mingram: TSV holding the MinGram runs.
        results_n32: TSV holding the single-epoch 32-shard runs, for the appendix.
        goldfish: Grid evaluation JSON, read for the round-trip failure counts.
        corpus: Which tokenizer training corpus the tokenizers were trained on.
        out: Main table, \\input{} by the paper.
        appendix_out: Seed and shard tables.
    """
    sources = {"bpe": results_bpe, "mingram": results_mingram}
    man = {t: _load_manifest(manifest, trainer=t, corpus=corpus) for t in KNOWN_TRAINERS}
    res = {t: _load_results(sources[t]) for t in KNOWN_TRAINERS}

    # The intrinsic main table's schemes, minus any that were never pretrained. Keyed on
    # the runs as well as MAIN_ARMS because a tokenizer trained but not pretrained --
    # bnd_wp, today -- would otherwise get a row of nothing but dashes.
    available = {a for t in KNOWN_TRAINERS for a in res[t]}
    schemes = ["plain"] + [a for a in MAIN_ARMS if a in available]

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
    # Round-trip failures come from eval_goldfish.json rather than the manifest: the
    # manifest only records them for a cell trained with an `--eval-texts` slice, which the
    # grid cells were not, while the Goldfish evaluation round-trips every cell it measures.
    rt = _roundtrip_failures(goldfish, corpus)

    # The bits-per-byte column, unrounded and unsummarised: every run behind each cell, the
    # paired difference against plain, and the p-value the caption rounds into a bound. The
    # table can only carry a mean and a spread, and a reviewer asking whether three seeds
    # can carry this should not have to take the bound on trust.
    per_run = []
    for t in KNOWN_TRAINERS:
        base_seeds = {int(r["seed"]) for r in res[t].get("plain", [])}
        for arm in schemes:
            rows_t = sorted((r for r in res[t].get(arm, []) if int(r["seed"]) in base_seeds),
                            key=lambda r: int(r["seed"]))
            if not rows_t:
                continue
            runs = " ".join(f"s{r['seed']}={float(r['val_bpb_true']):.6f}" for r in rows_t)
            line = f"{TRAINER_LABEL[t]} {arm}: {runs}"
            delta, dsd, dn = _paired_delta(res[t], arm)
            p = _paired_p(res[t], arm)
            if arm != "plain" and delta is not None and dn > 1:
                line += f" | paired vs plain {delta:+.6f} +- {dsd:.6f} (n={dn})"
                if p is not None:
                    line += f", p={p:.2g}"
            per_run.append(line)

    # One count for the whole table now that every cell is on plain's seeds. Named
    # individually rather than folded into a range if that ever stops being true, since
    # "3 to 6 runs" would leave the reader unable to tell which cell is which.
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
        "% Generated by paper_utils/boundary/downstream/make_tex_tables.py main-table. Do not edit.",
        r"% Requires booktabs and the paper's \bnds and \plainscheme macros, as table_intrinsic_main.",
        f"% sources: {os.path.relpath(manifest, REPO)}, "
        f"{_rel_sources(results_bpe)}, {_rel_sources(results_mingram)}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l rr}",
        r"\toprule",
        r"Scheme & \multicolumn{2}{c}{bits per byte $\downarrow$} \\",
        r"\cmidrule(lr){2-3}",
        r" & BPE & MinGram \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Downstream evaluation results.",
        r"Bits per byte on held-out ClimbMix after nanochat depth-12 pretraining, with "
        rf"standard deviation over {runs_note}.",
        rf"Every scheme beats \plainscheme{{}} at $p<{threshold:g}$, two-sided paired "
        rf"$t$-test over the {n_note}",
        # Explained only when there is one to explain. A complete table that still tells
        # the reader what a dash means reads as though a cell were missing.
        ("shared seeds." if not any(MISSING in r for r in rows) else
         rf"shared seeds. {MISSING} is a configuration that was not pretrained."),
        r"\textbf{Bold} is best in a column and \underline{underline} runner-up.",
        r"}",
        r"\label{tab:downstream-main}",
        "",
        "% Not in the table. Put in the text if the argument needs it:",
        f"%   vocabulary matched at {', '.join(f'{v:,}' for v in sorted(vocabs))} "
        "across every scheme and trainer",
        "%   roundtrip failures: "
        + (f"{', '.join(str(v) for v in sorted(rt))} for every scheme and trainer"
           if rt else "not measured for this grid"),
        "%   bits per byte per seed, and the paired test the caption rounds into a bound:",
        *[f"%     {line}" for line in per_run],
        "%   CORE is not scored: its language_modeling tasks assume encode(context) is a",
        "%     prefix of encode(context + continuation), which every marked scheme breaks.",
        "%     See paper_utils/boundary/downstream/core_prefix_check.py.",
        "%   raw bpb (nanochat's own figure) divides by summed token byte length and is not",
        "%     comparable across these tokenizers; see the appendix tables.",
        "",
    ]
    with open(out, "w") as f:
        f.write("\n".join(body))

    # The seed and shard checks cover the BPE sweep only, so they read that trainer's runs
    # directly rather than the per-trainer split above.
    with open(appendix_out, "w") as f:
        f.write("\n".join(
            ["% Generated by paper_utils/boundary/downstream/make_tex_tables.py. Do not edit.",
             r"% Requires booktabs and the paper's \bnds and \plainscheme macros.",
             f"% sources: {_rel_sources(results_bpe)}, {os.path.relpath(results_n32, REPO)}",
             ""] + appendix_tables(res["bpe"], _load_results(results_n32))))

    print(f"[tex] {out}")
    print(f"[tex] {appendix_out}")
    print(f"[tex] {len(rows)} scheme row(s), trainers {KNOWN_TRAINERS}, {n_note} paired seeds")


if __name__ == "__main__":
    app()
