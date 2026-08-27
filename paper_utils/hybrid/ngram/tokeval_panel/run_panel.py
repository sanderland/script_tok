"""Score the TokEval std-1B panel with n-gram bpb on the mixture corpus.

One row per (run, order). HF tokenizers go through HFTokenizerAdapter (Rust encoding);
the three SCRIPT-encoding runs are native script_bpe models. Two runs share the apertus
tokenizer (a seed replicate); both are scored -- the encoding cache collapses the cost --
and deduplication happens at analysis time. Eval and train are the whole respective
mixture files: they were built as disjoint slices of each source, so the split is
already train/eval-clean.
"""
import csv
import json
import os
import sys

from script_bpe.ngram import evaluate_ngram_bpb
from script_bpe.ngram.hf_adapter import HFTokenizerAdapter
from script_bpe.ngram.text import iter_documents
from script_bpe.tokenizers import load_tokenizer
from script_bpe.utils import create_logger

SP = os.environ.get("TOKEVAL_WORKDIR", ".")
ORDERS = [1, 2, 3]


def main(out_tsv):
    logger = create_logger("panel")
    # Docs capped at 100K chars: the tail (a handful of huge code/math files, max 853K)
    # dominates nothing statistically but is where pathological tokenizer behaviour lives.
    CAP = 100_000
    eval_docs = [d[:CAP] for d in iter_documents(f"file:{SP}/mix_eval.jsonl") if d]
    train_docs = [d[:CAP] for d in iter_documents(f"file:{SP}/mix_train.jsonl") if d]
    logger.info(f"{len(eval_docs):,} eval docs, {len(train_docs):,} train docs")

    panel = json.load(open(f"{SP}/panel_tokenizers.json"))
    for run, path in json.load(open(f"{SP}/scripttok_paths.json")).items():
        panel[run] = {"path": path, "slug": "scripttok", "native": True}

    rows = []
    fieldnames = None
    if os.path.exists(out_tsv):  # resume: keep completed runs, redo the partial one
        rows = list(csv.DictReader(open(out_tsv), delimiter="	"))
        fieldnames = list(rows[0]) if rows else None
        done = {r["run"] for r in rows}
        logger.info(f"resuming past {len(done)} completed runs")
    else:
        done = set()
    skipped = []
    for i, (run, info) in enumerate(sorted(panel.items())):
        if run in done:
            continue
        try:
            native = info.get("native", False)
            model = load_tokenizer(info["path"]) if native else HFTokenizerAdapter(info["path"])
            cls = (f"{type(model).__module__}.{type(model).__name__}" if native
                   else "script_bpe.ngram.hf_adapter.HFTokenizerAdapter")
            results = evaluate_ngram_bpb(
                model, eval_docs=eval_docs, train_docs=train_docs, orders=ORDERS,
                tokenizer_id=run, tokenizer_path=info["path"], tokenizer_class=cls,
                num_workers=4 if native else 1,
                cache_dir=f"{SP}/panel_encoded", spec="tokeval-mix-v2-cap100k", logger=logger,
            )
            for r in results:
                row = {"run": run, **r.as_row()}
                rows.append(row)
                fieldnames = fieldnames or list(row)
            with open(out_tsv, "w", newline="") as f:  # rewrite each round: crash-safe progress
                w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                w.writeheader()
                w.writerows(rows)
                logger.info(f"[{i+1}/{len(panel)}] {run} done")
        except Exception as e:
            # e.g. the nl-split scripttok variant: its config carries split_line_breaks,
            # which this script_bpe version does not implement. Stripping it would score a
            # DIFFERENT pretokenizer than the one that trained the model, so skip loudly.
            skipped.append(run)
            logger.info(f"[{i+1}/{len(panel)}] {run} SKIPPED: {type(e).__name__}: {str(e)[:120]}")
    logger.info(f"wrote {len(rows)} rows to {out_tsv}; skipped: {skipped or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1])
