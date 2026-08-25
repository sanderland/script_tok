#!/usr/bin/env python3
"""N-gram bits-per-byte for the MinGram paper's nine tokenizer arms. CPU only.

The point of the exercise: `run_downstream_eval.py` needs an H100 and hours per arm, so
the tokenizer ranking it produces is expensive enough that nobody iterates against it.
This scores the same nine arms with a Kneser-Ney n-gram model over each tokenizer's own
token stream, in minutes on a laptop. Whether that is worth anything is an empirical
question, and `correlate_ngram_downstream.py` is what answers it: it ranks these numbers
against the CORE and val-bpb columns the GPU runs already produced.

    # the nine arms, orders 1-5, on the text the tokenizers were trained from
    uv run python paper_utils/hybrid/ngram/run_ngram_eval.py --corpus fineweb_en_5gb

    # match nanochat's pretraining distribution instead of the tokenizer training corpus
    uv run python paper_utils/hybrid/ngram/run_ngram_eval.py --corpus fineweb_en_5gb \
        --text-source 'hf:OptimalScale/ClimbMix#train' --tag climbmix

Both the LM training text and the held-out text come from `--text-source`, in disjoint
contiguous slices (see `script_bpe.ngram.text.take_split`). The default reads the
registry's `_sampled_text` cache, which already holds the exact documents the corpus was
built from -- no re-download, and no pretokenized corpus involved, since that stores a
bag of chunks with sequence order discarded.
"""

import csv
import sys

import numpy as np
from pathlib import Path

from cyclopts import App

from paper_utils.hybrid.build_token_usage_counts import TokenizerSpec, tokenizer_specs
from paper_utils.hybrid.utils import REPO_ROOT
from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.ngram import evaluate_ngram_bpb
from script_bpe.ngram.text import take_split
from script_bpe.tokenizers import detect_tokenizer_model_class
from script_bpe.utils import create_logger

RESULTS_DIR = REPO_ROOT / "results" / "ngram"
# Enough held-out text that the bpb estimate is tight, and enough training text that a
# 5-gram over a 32k vocabulary is not pure backoff, while keeping the counting inside a
# few GB. Counting is the memory bound and it scales with *token* count: measured on
# adversarially unrepeatable text, 20M tokens costs 2.8 GB at order 3 and 4.3 GB at order
# 5, and real text is well under that because it repeats. 100M chars is roughly 25M
# tokens at these compression ratios. Raise both freely if the machine has the RAM --
# encoding is the only slow step and it is cached per (tokenizer, source, budget).
DEFAULT_EVAL_CHARS = 20_000_000
DEFAULT_TRAIN_CHARS = 100_000_000

app = App()


@app.default
def cli(
    corpus: str = "fineweb_en_5gb",
    text_source: str | None = None,
    orders: str = "1,2,3,4,5",
    methods: str | None = None,
    extra: list[str] | None = None,
    eval_chars: int = DEFAULT_EVAL_CHARS,
    train_chars: int = DEFAULT_TRAIN_CHARS,
    skip_chars: int = 0,
    num_workers: int = 8,
    tag: str | None = None,
    corpora_dir: str = PretokenizedCorpus.DEFAULT_BASE_PATH,
    out: str | None = None,
) -> None:
    """Score every trained arm of `corpus` at each n-gram order and write a TSV.

    Args:
        corpus: Tokenizer training corpus, e.g. fineweb_en_5gb. Selects which tokenizers to score.
        text_source: Text for the n-gram LM (sampled:/file:/hf:). Defaults to sampled:<corpus>.
        orders: Comma-separated n-gram orders.
        methods: Comma-separated subset of arm names; default is all nine.
        extra: Extra tokenizers outside the standard arms, as name=path. Repeatable.
            For variants the paper's spec list doesn't name (a second PathPiece
            setting, a retrained arm) so they land in the same TSV as the rest.
        eval_chars: Characters of held-out text. Taken from the front of the source.
        train_chars: Characters of LM training text, taken after the held-out slice.
        skip_chars: Discard this much text off the front first. Use it for a replicate on
            disjoint text, to check whether a ranking survives a different sample.
        num_workers: Processes for tokenizer encoding, which dominates wall-clock.
        tag: Suffix for the output filename, to keep runs on different text sources apart.
        corpora_dir: Where the registry keeps corpora and the sampled-text cache.
        out: Explicit output TSV path.
    """
    logger = create_logger("ngram-eval")
    source = text_source or f"sampled:{corpus}"
    order_list = [int(o) for o in orders.split(",")]
    wanted = set(methods.split(",")) if methods else None

    specs = [s for s in tokenizer_specs(corpus) if wanted is None or s.name in wanted]
    if wanted and len(specs) != len(wanted):
        raise SystemExit(f"unknown arm(s): {wanted - {s.name for s in specs}}")
    for item in extra or []:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--extra expects name=path, got {item!r}")
        specs.append(TokenizerSpec(name=name, path=Path(path)))
    missing = [s for s in specs if not Path(s.path).exists()]
    if missing:
        raise SystemExit(
            "these tokenizers are not on disk:\n"
            + "\n".join(f"  {s.name:24s} {s.path}" for s in missing)
            + "\n\nresults/ is gitignored, so a fresh clone has none of them. Either copy the\n"
            "trained models across, or retrain with:\n"
            "  uv run bash paper_utils/hybrid/run_all_experiments.sh"
        )

    skipping = f", after skipping {skip_chars:,}" if skip_chars else ""
    logger.info(f"reading text from {source}: {eval_chars:,} eval + {train_chars:,} train chars{skipping}")
    eval_docs, train_docs = take_split(source, eval_chars=eval_chars, train_chars=train_chars,
                                       skip_chars=skip_chars, base_dir=corpora_dir)
    logger.info(f"{len(eval_docs):,} eval docs / {len(train_docs):,} train docs")

    rows = []
    doc_bits: dict[str, "np.ndarray"] = {}
    doc_bytes = None
    for spec in specs:
        cls = detect_tokenizer_model_class(str(spec.path))
        model = cls.load(str(spec.path))
        results = evaluate_ngram_bpb(
            model,
            eval_docs=eval_docs,
            train_docs=train_docs,
            orders=order_list,
            tokenizer_id=spec.name,
            tokenizer_path=str(spec.path),
            tokenizer_class=f"{cls.__module__}.{cls.__name__}",
            num_workers=num_workers,
            cache_dir=str(RESULTS_DIR / "encoded"),
            spec=f"{source}:{eval_chars}:{train_chars}:{skip_chars}",
            logger=logger,
        )
        rows.extend({"corpus": corpus, "text_source": source, **r.as_row()} for r in results)
        for r in results:
            doc_bits[f"{spec.name}|{r.order}"] = np.asarray(r.doc_bits)
            doc_bytes = np.asarray(r.doc_bytes)

    out_path = Path(out) if out else RESULTS_DIR / f"ngram_bpb_{corpus}{'_' + tag if tag else ''}.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"wrote {len(rows)} rows to {out_path}")

    # Per-document bits alongside the TSV. Every arm scores the same held-out documents, so
    # storing them per document is what lets arm_significance.py run a *paired* bootstrap --
    # the only way to ask whether two arms really differ, given the metric is deterministic
    # and so has no seed-to-seed spread to quote.
    if doc_bits and doc_bytes is not None:
        bits_path = out_path.with_name(out_path.stem + "_docbits.npz")
        np.savez_compressed(bits_path, doc_bytes=doc_bytes, **doc_bits)
        logger.info(f"wrote per-document bits for {len(doc_bits)} (arm, order) pairs to {bits_path}")


if __name__ == "__main__":
    sys.exit(app())
