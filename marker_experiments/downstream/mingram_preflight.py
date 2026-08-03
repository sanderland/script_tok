#!/usr/bin/env python3
"""Checks the MinGram tokenizers must pass before any GPU time is spent on them.

Run after training and before submitting the downstream sweep:

    uv run python marker_experiments/downstream/mingram_preflight.py

Five checks, each of which has a specific failure in mind:

1. Vocabulary is exactly the matched size. MinGram reaches its target by pruning down from
   a BPE initialisation, so unlike BPE it can stop early or overshoot. An arm that misses
   34,685 changes the embedding shape, hence the parameter count, hence the token horizon,
   and the comparison against the BPE runs stops being matched.
2. Round-trip on the held-out slice. The boundary scheme rebuilds elided spaces at decode
   time; a segmentation change could break that.
3. chars/token in the same range as the BPE tokenizer for the same arm. A wildly different
   number means the trainer did not converge to a comparable vocabulary.
4. The CORE prefix property. Both encoders are chunk-local: BPE and MinGram each encode
   one pretokenizer chunk at a time, so whether encode(context) is a prefix of
   encode(context + continuation) is a property of the pretokenizer, not of the trainer,
   and the answer must match the BPE measurement. This check exists to confirm that rather
   than to discover something new, and it would catch a change in the pretokenizer or in
   the chunk-locality assumption. Getting the answer wrong costs a whole run: base_eval
   raises on the first violation and then nothing is reported, bits-per-byte included.
5. A marker token decodes to the empty string and still receives byte count 1. That floor
   is what stops the bpb metric masking marker loss, and it is what makes the corrected
   bits-per-byte exact.

Prints the CORE_SAFE_ARMS value the sweep should use, and exits non-zero if any check fails.
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


@app.default
def main(
    arms: str = "plain,bnd_wpd",
    trainer: str = "mingram",
    corpus: str = "fineweb_en_5gb",
    vocab: int = 34685,
    eval_texts: str = os.path.join(REPO, "marker_experiments", "eval_texts", "en.json"),
    max_per_task: int = 60,
) -> None:
    """Run the checks and report the CORE_SAFE_ARMS the sweep should use.

    Args:
        arms: Comma-separated arms to check.
        trainer: Which tokenizers to check. `bpe` reruns these checks against the BPE
            tokenizers, whose answers are already known, which is how the checks
            themselves get validated.
        corpus: Corpus the tokenizers were trained on.
        vocab: Matched total vocabulary every arm must hit exactly.
        eval_texts: Held-out slice for round-trip and chars/token.
        max_per_task: CORE examples per task for the prefix check.
    """
    from marker_experiments.downstream.boundary_tokenizer import (
        BoundaryBPETokenizer,
        BoundaryMinGramModel,
    )
    from pynanochat.tokenizer import special_aware_token_bytes
    from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter

    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    texts = json.load(open(eval_texts))
    failures = []
    core_safe = []

    def check(name, ok, detail=""):
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{('  | ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    for arm in arm_list:
        print(f"\n{arm} ({trainer})")
        path = os.path.join(HERE, "tokenizers", f"{corpus}_{arm}_{trainer}_v{vocab}.json.gz")
        if not os.path.exists(path):
            check(f"{arm}: tokenizer exists", False, path)
            continue
        cls = BoundaryBPETokenizer if trainer == "bpe" else BoundaryMinGramModel
        model = cls.load(path)
        adapter = ScriptBPETokenizerAdapter(model)

        n = adapter.get_vocab_size()
        # +1 for the synthetic BOS the adapter appends.
        check("vocabulary is exactly the matched size", n == vocab + 1, f"{n:,} vs {vocab + 1:,}")

        chars = toks = fails = 0
        for t in texts:
            ids = model.encode(t)
            chars += len(t)
            toks += len(ids)
            if model.decode(ids) != t:
                fails += 1
        check("round-trip on the held-out slice", fails == 0, f"{fails} failure(s)")
        cpt = chars / toks

        bpe_path = os.path.join(HERE, "tokenizers", f"{corpus}_{arm}_bpe_v{vocab}.json.gz")
        if trainer != "bpe" and os.path.exists(bpe_path):
            bpe = BoundaryBPETokenizer.load(bpe_path)
            bc = bt = 0
            for t in texts:
                bc += len(t)
                bt += len(bpe.encode(t))
            bcpt = bc / bt
            rel = 100.0 * (cpt - bcpt) / bcpt
            # MinGram beat BPE by 1.2 to 1.4% on the baseline in the published grid, so a
            # few percent either way is expected and a large gap is not.
            check("chars/token close to the BPE tokenizer for this arm", abs(rel) < 8.0,
                  f"mingram {cpt:.4f} vs bpe {bcpt:.4f} ({rel:+.2f}%)")
        elif trainer == "bpe":
            print("  [skip] chars/token comparison: this is the BPE tokenizer itself")
        else:
            print(f"  [skip] no BPE tokenizer at {bpe_path} to compare against")

        table = special_aware_token_bytes(adapter)
        zeros = [i for i, v in enumerate(table) if v == 0]
        # `zeros == [bos]` alone holds by construction for any tokenizer, so it protects
        # nothing. What matters is that the tokens which decode to nothing still carry a
        # byte, since that is what keeps their loss in the numerator.
        empties = [i for i in range(adapter.get_vocab_size())
                   if i != adapter.get_bos_token_id() and adapter.decode([i]) == ""]
        check("only BOS carries zero bytes", zeros == [adapter.get_bos_token_id()],
              f"{len(zeros)} zero-byte id(s)")
        # plain has no marker or caps codes, so having none to floor is the correct state
        # for the baseline, not a failure. What must never happen is one existing at 0.
        check("empty-decoding tokens are floored to 1 byte",
              all(table[i] == 1 for i in empties),
              f"{len(empties)} such token(s)"
              + (f", e.g. id {empties[0]}" if empties else ", none expected for this arm"))

        aborts, total = _core_prefix_aborts(adapter, max_per_task)
        safe = aborts == 0
        if safe:
            core_safe.append(arm)
        print(f"  [{'ok  ' if safe else 'note'}] CORE prefix property: "
              f"{aborts}/{total} examples would abort"
              f"  -> {'core,bpb' if safe else 'bpb only'}")

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
    else:
        print("all checks passed")
    print(f"CORE_SAFE_ARMS=\"{' '.join(core_safe)}\"")
    print("=" * 60)
    raise SystemExit(1 if failures else 0)


def _core_prefix_aborts(adapter, max_per_task):
    """How many CORE language_modeling examples would trip base_eval's prefix assertion.

    Replays nanochat's own render_prompts_lm against the real core.yaml metadata, so the
    prompts are the ones base_eval would tokenize.
    """
    import random

    import yaml

    sys.path.insert(0, os.path.join(REPO, "eval", "py-nanochat", "vendor", "nanochat"))
    from nanochat.core_eval import render_prompts_lm

    bundle = os.path.join(os.environ["NANOCHAT_BASE"], "eval_bundle")
    with open(os.path.join(bundle, "core.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bos = adapter.get_bos_token_id()
    aborts = total = 0
    for task in config["icl_tasks"]:
        if task["icl_task_type"] != "language_modeling":
            continue
        with open(os.path.join(bundle, "eval_data", task["dataset_uri"]), encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]
        random.Random(1337).shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        nfew = task["num_fewshot"][0]
        delim = task.get("continuation_delimiter", " ")
        for idx, item in enumerate(data):
            fewshot = []
            if nfew > 0:
                rng = random.Random(1234 + idx)
                fewshot = [data[i] for i in rng.sample([i for i in range(len(data)) if i != idx], nfew)]
            without, with_ = render_prompts_lm(item, delim, fewshot)
            a = adapter.encode(without, prepend=bos)
            b = adapter.encode(with_, prepend=bos)
            total += 1
            if not (len(a) < len(b) and a == b[: len(a)]):
                aborts += 1
    return aborts, total


if __name__ == "__main__":
    app()
