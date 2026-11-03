#!/usr/bin/env python3
"""Generate the multilingual downstream table from the run artifacts.

English, Korean and Russian side by side, one row per scheme, both trainers:

    results.tsv, results_caps.tsv, results_mingram.tsv   English, the runs the paper reports
    results_ko_bpe.tsv, results_ko_mingram.tsv           Korean
    results_ru_bpe.tsv, results_ru_mingram.tsv           Russian

    table_downstream_multilang.tex   paired difference against plain, per language and trainer

Cells are the difference against `plain` at equal seed, so a negative cell is a scheme that
scored below `plain`, which is better. Pairing is what makes three seeds enough: one data
order per seed is shared across the schemes of a language, so differencing at equal seed
cancels it. Absolute values sit in the `plain` row, and only there, because bits per byte is
not comparable across languages: a Cyrillic byte carries less text than a Latin one.

Korean and Russian were trained in September 2026 with the same pipeline as the English runs,
on FineWeb-2 rather than ClimbMix; `DESIGN_CHOICES_MULTILANG.md` records what that changed.
The English column is the published run, not the September retrain of it, whose files sit
beside these as `results_en_retrain_*.tsv`.

    uv run python paper_utils/boundary/downstream/make_multilang_table.py
"""
import os
import sys

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

from paper_utils.boundary.downstream.make_tex_tables import (  # noqa: E402
    _load_results,
    _mean_sd,
    _paired_delta,
    _paired_p,
)
from paper_utils.boundary.make_intrinsic_table import (  # noqa: E402
    ARM_LABEL as SCHEME_LABEL,
    MAIN_ARMS,
    MISSING,
    PLAIN_LABEL,
)

GENERATED = os.path.join(REPO, "paper_utils", "boundary", "paper", "generated")
OUT = os.path.join(GENERATED, "table_downstream_multilang.tex")
TRAINER_LABEL = {"bpe": "BPE", "mingram": "MinGram"}
# Language, then the TSVs holding each trainer's runs. English keeps its split files: the caps
# arm ran under its own output directory, and `_load_results` refuses to merge trainers but
# not files.
LANGUAGES = [
    ("English", {"bpe": "results.tsv,results_caps.tsv", "mingram": "results_mingram.tsv"}),
    ("Korean", {"bpe": "results_ko_bpe.tsv", "mingram": "results_ko_mingram.tsv"}),
    ("Russian", {"bpe": "results_ru_bpe.tsv", "mingram": "results_ru_mingram.tsv"}),
]
# The baseline, then the schemes the intrinsic main table shows, minus bnd_wp, which was never
# pretrained. MAIN_ARMS holds the marker schemes only, so plain is prepended here.
SCHEMES = ["plain"] + [a for a in MAIN_ARMS if a != "bnd_wp"]


def _bound(p_values):
    """The tightest round bound the measured p-values support, so the caption cannot overstate."""
    worst = max(p_values)
    for bound in (0.001, 0.005, 0.01, 0.05):
        if worst < bound:
            return bound
    return None


@cyclopts.App().default
def main(generated: str = GENERATED, out: str = OUT) -> None:
    """Write the multilingual table.

    Args:
        generated: directory holding the result TSVs.
        out: .tex file to write.
    """
    loaded = {}
    for language, per_trainer in LANGUAGES:
        for trainer, files in per_trainer.items():
            paths = ",".join(os.path.join(generated, f) for f in files.split(","))
            by_arm = _load_results(paths)
            if by_arm:
                loaded[(language, trainer)] = by_arm

    columns = [(lang, tr) for lang, _ in LANGUAGES for tr in ("bpe", "mingram") if (lang, tr) in loaded]
    if not columns:
        raise SystemExit(f"no result TSVs under {generated}")

    rows, seed_counts, gain_p, loss_p = [], set(), [], []
    for scheme in SCHEMES:
        cells = []
        for key in columns:
            by_arm = loaded[key]
            if scheme == "plain":
                mean, _, n = _mean_sd(by_arm.get("plain", []), "val_bpb_true")
                cells.append(f"{mean:.4f}" if n else MISSING)
                seed_counts.add(n)
                continue
            delta, _, n = _paired_delta(by_arm, scheme)
            if not n:
                cells.append(MISSING)
                continue
            seed_counts.add(n)
            cells.append(f"{delta:+.4f}")
            p = _paired_p(by_arm, scheme)
            if p is not None:
                (loss_p if delta > 0 else gain_p).append(p)
        label = PLAIN_LABEL if scheme == "plain" else SCHEME_LABEL[scheme]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")

    header = " & ".join(rf"\textbf{{{lang}}}, {TRAINER_LABEL[tr]}" for lang, tr in columns)
    n_seeds = sorted(seed_counts)
    seeds_text = f"{n_seeds[0]} seeds" if len(n_seeds) == 1 else f"{min(n_seeds)} to {max(n_seeds)} seeds"
    gain_bound = _bound(gain_p) if gain_p else None
    loss_bound = _bound(loss_p) if loss_p else None

    caption = [
        r"\caption{Boundary markers across three languages.",
        r"The \textsc{plain} row is validation bits per byte; every other row is that scheme",
        r"minus \textsc{plain} at equal seed, so a negative entry is a scheme that scored below",
        rf"the baseline. Each cell rests on {seeds_text}, paired by seed.",
    ]
    if gain_bound is not None:
        caption.append(
            rf"Every entry below the baseline is significant at $p < {gain_bound}$ (paired $t$-test)."
        )
    if loss_bound is None and loss_p:
        caption.append(
            rf"No entry above the baseline reaches $p < 0.05$; the smallest is $p = {min(loss_p):.2f}$."
        )
    caption.append(
        r"Bits per byte is not comparable across languages, so only the sign and size within a"
    )
    caption.append(r"column carry meaning.}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(columns) + "}",
        r"\toprule",
        f" & {header} \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        *caption,
        r"\label{tab:downstream-multilang}",
        r"\end{table}",
    ]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(rows)} scheme(s) x {len(columns)} column(s), {seeds_text}")
    for (lang, tr), by_arm in loaded.items():
        for scheme in SCHEMES:
            if scheme == "plain":
                continue
            delta, sd, n = _paired_delta(by_arm, scheme)
            if n:
                p = _paired_p(by_arm, scheme)
                print(f"  {lang:8s} {TRAINER_LABEL[tr]:7s} {scheme:13s} {delta:+.5f} "
                      f"sd {sd:.5f} n {n} p {p:.3f}")


if __name__ == "__main__":
    main()
