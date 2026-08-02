#!/usr/bin/env python3
"""Per-arm tokenization statistics on the evaluation data, as a saved artifact.

Two figures the paper needs and that were previously typed by hand:

  tokens_per_true_byte  All arms train on the same number of tokens, so the amount of
                        text each covers is inversely proportional to this. The caption
                        quotes coverage relative to plain.
  zero_byte_share       Fraction of emitted tokens that decode to the empty string. These
                        are the marker and caps codes whose loss nanochat's bpb would drop
                        without the one-byte floor, so this number is the size of the
                        effect that motivated the floor.

    uv run python marker_experiments/downstream/measure_text_stats.py

Writes marker_experiments/paper/generated/text_stats.json, which make_tex_tables.py reads.
"""

import json
import os
import sys

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval", "py-nanochat"))

app = cyclopts.App()
DOTTED = "marker_experiments.downstream.boundary_tokenizer.BoundaryBPETokenizer"


@app.default
def main(
    arms: str = "plain,bnd_w,bnd_wpd,bnd_wpd_caps",
    corpus: str = "fineweb_en_5gb",
    trainer: str = "bpe",
    vocab: int = 34685,
    base_dir: str | None = None,
    max_chars: int = 20_000_000,
    out: str = os.path.join(REPO, "marker_experiments", "paper", "generated", "text_stats.json"),
) -> None:
    """Measure and record the statistics.

    Args:
        arms: Comma-separated arms.
        corpus: Corpus the tokenizers were trained on (part of their filename).
        trainer: Trainer used (part of their filename).
        vocab: Matched total vocabulary (part of their filename).
        base_dir: Shared NANOCHAT_BASE holding base_data_climbmix. Defaults to $NANOCHAT_BASE.
        max_chars: Characters of validation text to measure over.
        out: JSON artifact to write.
    """
    import importlib

    import pyarrow.parquet as pq

    from pynanochat.tokenizer import special_aware_token_bytes
    from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter

    base = base_dir or os.environ.get("NANOCHAT_BASE")
    if not base:
        raise SystemExit("set NANOCHAT_BASE or pass --base-dir")

    data_dir = os.path.join(base, "base_data_climbmix")
    shards = sorted(f for f in os.listdir(data_dir) if f.startswith("shard_"))
    if not shards:
        raise SystemExit(f"no shards under {data_dir}")
    shard = shards[-1]  # the validation shard, the one base_eval scores

    docs, nchars = [], 0
    for doc in pq.read_table(os.path.join(data_dir, shard)).column("text").to_pylist():
        docs.append(doc)
        nchars += len(doc)
        if nchars >= max_chars:
            break
    text = "\n".join(docs)
    true_bytes = len(text.encode("utf-8"))

    module_name, _, cls_name = DOTTED.rpartition(".")
    cls = getattr(importlib.import_module(module_name), cls_name)

    stats = {}
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        path = os.path.join(HERE, "tokenizers", f"{corpus}_{arm}_{trainer}_v{vocab}.json.gz")
        if not os.path.exists(path):
            raise SystemExit(f"missing tokenizer for arm {arm}: {path}")
        adapter = ScriptBPETokenizerAdapter(cls.load(path))
        table = special_aware_token_bytes(adapter)
        ids = adapter.encode(text)
        zero = sum(1 for t in ids if table[t] == 0)
        # Zero-byte tokens under the ORIGINAL rule, which is the quantity that motivated
        # the one-byte floor: how much loss the metric would drop without it.
        specials = set(adapter.get_special_tokens())
        unfloored_zero_ids = {
            t for t in set(ids)
            if (d := adapter.decode([t])) not in specials and len(d.encode("utf-8")) == 0
        }
        unfloored_zero = sum(1 for t in ids if t in unfloored_zero_ids)
        stats[arm] = {
            "tokens": len(ids),
            "true_bytes": true_bytes,
            "token_bytes_sum": sum(table[t] for t in ids),
            "tokens_per_true_byte": len(ids) / true_bytes,
            "true_bytes_per_token": true_bytes / len(ids),
            "zero_byte_tokens_after_floor": zero,
            "zero_byte_share_before_floor": unfloored_zero / len(ids),
        }

    base_arm = stats.get("plain")
    if base_arm:
        for arm, v in stats.items():
            # At a fixed token budget, text covered is proportional to bytes per token.
            v["text_coverage_vs_plain"] = (
                v["true_bytes_per_token"] / base_arm["true_bytes_per_token"]
            )

    payload = {"shard": shard, "true_bytes": true_bytes, "documents": len(docs), "arms": stats}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"[stats] {shard}, {len(docs):,} docs, {true_bytes:,} true bytes -> {out}\n")
    print(f"  {'arm':<14} {'bytes/token':>12} {'coverage vs plain':>18} {'zero-byte share':>16}")
    for arm, v in stats.items():
        print(f"  {arm:<14} {v['true_bytes_per_token']:>12.4f} "
              f"{100 * v.get('text_coverage_vs_plain', float('nan')):>17.2f}% "
              f"{100 * v['zero_byte_share_before_floor']:>15.3f}%")


if __name__ == "__main__":
    app()
