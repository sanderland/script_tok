#!/usr/bin/env python3
"""Tokenizer-side checks for the bnd_wpd_extcaps downstream sweep, before any GPU time.

Three sweeps have been discarded for measurement bugs, so every property the sweep relies
on is measured here rather than assumed:

  * vocabulary is exactly the matched size (the adapter reports vocab+1, for the synthetic
    BOS it adds on top of the trained vocabulary);
  * the held-out slice round-trips with zero failures;
  * chars/token beside the arms this one is meant to be compared against;
  * only BOS carries zero bytes in the table evaluate_bpb masks on, and every token that
    decodes to the empty string carries the floored count of 1. A boundary marker decodes
    to nothing, and without the floor the cost of predicting it would be dropped from the
    bpb numerator for exactly the arms under test.

The CORE prefix property is measured separately by core_prefix_check.py, which needs the
CORE bundle.

    uv run python marker_experiments/downstream/extcaps_preflight.py
"""

import json
import os
import sys

import cyclopts

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from marker_experiments.downstream.train_matched import eval_compression  # noqa: E402

app = cyclopts.App()


@app.default
def main(
    arm: str = "bnd_wpd_extcaps",
    against: str = "bnd_wpd_caps,plain",
    corpus: str = "fineweb_en_5gb",
    trainer: str = "bpe",
    vocab: int = 34_685,
    eval_texts: str = os.path.join(REPO, "marker_experiments", "eval_texts", "en.json"),
) -> None:
    """Check the trained tokenizer and print the numbers the sweep is gated on.

    Args:
        arm: The arm under test.
        against: Comma-separated arms to report chars/token beside it.
        corpus: Corpus name in the tokenizer filename.
        trainer: Trainer name in the tokenizer filename.
        vocab: Matched total vocabulary the arm must land on exactly.
        eval_texts: Held-out slice for chars/token and round-trip.
    """
    from marker_experiments.downstream.boundary_tokenizer import BoundaryBPETokenizer
    from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter
    from pynanochat.tokenizer import special_aware_token_bytes, special_token_ids

    tok_dir = os.path.join(HERE, "tokenizers")
    path = os.path.join(tok_dir, f"{corpus}_{arm}_{trainer}_v{vocab}.json.gz")
    if not os.path.exists(path):
        raise SystemExit(f"missing tokenizer: {path}")

    failures = []
    print(f"tokenizer: {path}")
    tokenizer = BoundaryBPETokenizer.load(path)
    adapter = ScriptBPETokenizerAdapter(tokenizer)

    # 1. vocabulary
    n_adapter = adapter.get_vocab_size()
    n_trained = len(tokenizer.tokens)
    print(f"\n[1] vocabulary  trained={n_trained:,}  adapter={n_adapter:,} (+1 synthetic BOS)")
    if n_trained != vocab:
        failures.append(f"trained vocabulary {n_trained} != {vocab}")
    if n_adapter != vocab + 1:
        failures.append(f"adapter vocabulary {n_adapter} != {vocab + 1}")

    # 2 + 3. round-trip and chars/token, beside the comparison arms
    print(f"\n[2/3] held-out slice {os.path.relpath(eval_texts, REPO)}")
    stats = eval_compression(tokenizer, eval_texts)
    if not stats:
        raise SystemExit(f"no eval slice at {eval_texts}")
    print(f"    {arm:<20} chars/token {stats['eval_chars_per_token']:.4f}  "
          f"roundtrip_failures {stats['roundtrip_failures']}  "
          f"({stats['eval_chars']:,} chars, {stats['eval_tokens']:,} tokens)")
    if stats["roundtrip_failures"] != 0:
        failures.append(f"{stats['roundtrip_failures']} round-trip failures on the held-out slice")

    base = stats["eval_chars_per_token"]
    for other in [a.strip() for a in against.split(",") if a.strip()]:
        other_path = os.path.join(tok_dir, f"{corpus}_{other}_{trainer}_v{vocab}.json.gz")
        if not os.path.exists(other_path):
            print(f"    {other:<20} MISSING at {other_path}")
            failures.append(f"comparison tokenizer missing: {other_path}")
            continue
        other_tok = BoundaryBPETokenizer.load(other_path)
        s = eval_compression(other_tok, eval_texts)
        delta = 100 * (base - s["eval_chars_per_token"]) / s["eval_chars_per_token"]
        print(f"    {other:<20} chars/token {s['eval_chars_per_token']:.4f}  "
              f"roundtrip_failures {s['roundtrip_failures']}  "
              f"({arm} is {delta:+.2f}% against it)")

    # 4. the byte table evaluate_bpb masks on
    print("\n[4] byte table (0 bytes -> loss masked by evaluate_bpb)")
    counts = special_aware_token_bytes(adapter)
    specials = special_token_ids(adapter)
    bos = adapter.get_bos_token_id()
    zeros = [i for i, c in enumerate(counts) if c == 0]
    empty = [i for i in range(n_adapter) if i not in specials and adapter.decode([i]) == ""]
    print(f"    special ids {sorted(specials)}  bos id {bos}")
    print(f"    ids with 0 bytes: {zeros}")
    print(f"    non-special ids decoding to the empty string: {len(empty)}")
    if empty:
        bad = [i for i in empty if counts[i] != 1]
        print(f"    of those, byte count 1: {len(empty) - len(bad)}; not 1: {len(bad)}")
        if bad:
            failures.append(f"{len(bad)} empty-decoding tokens are not floored to 1 byte")
    if zeros != [bos]:
        failures.append(f"ids with 0 bytes are {zeros}, expected exactly [{bos}] (BOS)")

    print("\n" + "=" * 60)
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("all tokenizer-side checks passed")


if __name__ == "__main__":
    app()
