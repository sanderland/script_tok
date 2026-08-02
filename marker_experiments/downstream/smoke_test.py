#!/usr/bin/env python3
"""Everything about the downstream pipeline that can be checked without a GPU.

The GPU leg (nanochat pretrain + CORE/bpb eval) is NOT exercised here -- it needs
torch and an H100, neither of which the development container has. What *is*
exercised is the whole tokenizer-side path, which is where the boundary tokenizers
differ from the tokenizers this harness has run before:

  1. fresh-process load by dotted class path (the harness loads tokenizers in
     subprocesses; `BPETokenizer.load` on a boundary model raises
     KeyError: 'BoundaryScriptPretokenizer' unless the registering module is imported)
  2. the `pynanochat.Tokenizer` contract, including `write_token_bytes`, which
     produces the byte-length table the bits-per-byte metric is computed against
  3. dense id space: contiguous [0, n), synthetic BOS at n, vocab n+1
  4. the uint16 bound `pretokenize.py` asserts (vocab <= 65535)
  5. round-trip through the adapter on marker-, caps- and digit-heavy text
  6. matched vocabulary across arms -- an unmatched set silently changes parameter
     count and token horizon between arms, so this is a hard failure, not a warning

Run:  uv run python marker_experiments/downstream/smoke_test.py
      uv run python marker_experiments/downstream/smoke_test.py --tokenizer-dir <dir>
"""

import json
import os
import subprocess
import sys
import tempfile

import cyclopts

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(REPO, "marker_experiments", "tokenizers")
PYNANOCHAT_SRC = os.path.join(REPO, "eval", "py-nanochat")
DOTTED_BPE = "marker_experiments.downstream.boundary_tokenizer.BoundaryBPETokenizer"
DOTTED_MINGRAM = "marker_experiments.downstream.boundary_tokenizer.BoundaryMinGramModel"


def dotted_for(filename):
    """`--tokenizer-class` for a tokenizer file. Trainers serialise differently: loading
    a MinGram model with BPETokenizer.load raises KeyError: 'merge_rules'."""
    return DOTTED_MINGRAM if "mingram" in filename else DOTTED_BPE

# marker pairs, an elided space, title case, all caps, mixed case (left literal),
# a digit run, and a script change inside one word span.
SAMPLES = [
    "The NASA report, 2024, says WiFi use grew 41.5% (n=1,203).",
    "one two  three\n\nfour\tfive",
    "latin кириллица mixed, 123 456.",
    "a,b c , d -- e.f.g 'quoted' \"double\"",
    "「日本語」と한국어 and العربية 42",
]

app = cyclopts.App()
FAILS: list[str] = []


CURRENT = [""]


def check(name, ok, detail=""):
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(f"{CURRENT[0]}: {name}")


def _import_adapter():
    """pynanochat is an optional extra; fall back to its source tree so this runs
    in a container that has not done `uv sync --extra downstream`."""
    try:
        from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter
        from pynanochat.tokenizer import write_token_bytes
    except ModuleNotFoundError:
        sys.path.insert(0, PYNANOCHAT_SRC)
        from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter
        from pynanochat.tokenizer import write_token_bytes
    return ScriptBPETokenizerAdapter, write_token_bytes


def fresh_process_load(path, dotted):
    """Load in a subprocess with cwd outside the repo: exactly what the harness does."""
    code = (
        "import importlib,sys\n"
        f"m,_,c={dotted!r}.rpartition('.')\n"
        "t=getattr(importlib.import_module(m),c).load(sys.argv[1])\n"
        "print(len(t.tokens))\n"
    )
    r = subprocess.run(
        # cwd is outside the repo, so the tokenizer path must be absolute.
        [sys.executable, "-c", code, os.path.abspath(path)], cwd=tempfile.gettempdir(),
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1] if r.stderr else "no stderr"
    return int(r.stdout.strip()), ""


@app.default
def main(tokenizer_dir: str = DEFAULT_DIR, pattern: str = "en_",
         require_matched_vocab: bool = False) -> None:
    """Check the tokenizer-side downstream path for every matching tokenizer.

    Args:
        tokenizer_dir: Directory of .json.gz tokenizers to check.
        pattern: Only check files containing this substring.
    """
    ScriptBPETokenizerAdapter, write_token_bytes = _import_adapter()
    paths = sorted(
        os.path.join(tokenizer_dir, f)
        for f in os.listdir(tokenizer_dir)
        if f.endswith(".json.gz") and pattern in f
    )
    if not paths:
        raise SystemExit(f"no tokenizers matching {pattern!r} in {tokenizer_dir}")

    long_text = ""
    tsw = os.path.join(REPO, "tests", "data", "taylorswift.txt")
    if os.path.exists(tsw):
        long_text = open(tsw, encoding="utf-8").read()

    vocabs, token_counts = {}, {}
    for path in paths:
        name = os.path.basename(path).replace(".json.gz", "")
        CURRENT[0] = name
        print(f"\n{name}")
        dotted = dotted_for(name)

        n_tokens, err = fresh_process_load(path, dotted)
        check("fresh-process load by dotted path", n_tokens is not None, err)
        if n_tokens is None:
            continue

        import importlib

        mod, _, cls = dotted.rpartition(".")
        tokenizer = getattr(importlib.import_module(mod), cls).load(path)
        adapter = ScriptBPETokenizerAdapter(tokenizer)
        vocab = adapter.get_vocab_size()
        bos = adapter.get_bos_token_id()
        vocabs[name] = vocab

        check("vocab == n_tokens + 1", vocab == n_tokens + 1, f"{vocab:,} vs {n_tokens:,}+1")
        check("bos is the last dense id", bos == vocab - 1, f"bos={bos}")
        check("uint16 bound (pretokenize.py asserts this)", vocab <= 65535, f"vocab={vocab:,}")

        for attr in ("encode", "decode", "get_bos_token_id", "get_vocab_size",
                     "get_special_tokens", "id_to_token"):
            if not hasattr(adapter, attr):
                check(f"contract: {attr}", False)
        check("contract: all six methods present", True)

        bad = []
        for s in SAMPLES:
            ids = adapter.encode(s, prepend=bos)
            if ids[0] != bos or adapter.decode(ids) != s:
                bad.append(s)
        check("round-trip through adapter (5 samples)", not bad, f"failed: {bad[:1]}")

        if long_text:
            ids = adapter.encode(long_text)
            token_counts[name] = len(ids)
            check(
                "round-trip on taylorswift.txt",
                adapter.decode(ids) == long_text,
                f"{len(long_text):,} chars -> {len(ids):,} tokens "
                f"({len(long_text) / len(ids):.4f} ch/tok)",
            )
            in_range = all(0 <= i < vocab for i in ids)
            check("all ids inside dense [0, vocab)", in_range)

        # batch encode is what pretokenize.py calls; it must agree with single encode.
        batch = adapter.encode(SAMPLES[:3], prepend=bos)
        singles = [adapter.encode(s, prepend=bos) for s in SAMPLES[:3]]
        check("batch encode == single encode", [list(x) for x in batch] == [list(x) for x in singles])

        # write_token_bytes serialises with torch.save, so it only runs where the
        # downstream extra is installed. On a cluster this is the first real check.
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            print("  [skip] write_token_bytes (bpb byte table)  torch not installed")
        else:
            with tempfile.TemporaryDirectory() as d:
                write_token_bytes(adapter, d)
                check("write_token_bytes (bpb byte table)", bool(os.listdir(d)),
                      ", ".join(os.listdir(d)))

    print("\nvocabulary sizes")
    for name, v in sorted(vocabs.items()):
        print(f"  {name:<40} {v:,}")
    if len(set(vocabs.values())) > 1:
        print("  NOTE: these are the compression-grid tokenizers, which match "
              "additional_vocab_size, not total vocab.\n"
              "        Use train_matched.py before running the downstream comparison.")
        if require_matched_vocab:
            # train_matched.py's own cross-arm assertion can no longer fire: each job of
            # the sweep is handed a per-arm manifest, so the dict it checks holds one arm.
            # This is the only place in a job that sees every arm at once.
            raise SystemExit(
                "vocabulary sizes are NOT matched across arms: "
                + ", ".join(f"{k}={v}" for k, v in sorted(vocabs.items()))
                + "\nA downstream comparison across unmatched vocabularies is not valid."
            )

    if token_counts:
        print("\ntokens on taylorswift.txt (lower is better)")
        base = next((v for k, v in token_counts.items() if "plain" in k), None)
        for name, n in sorted(token_counts.items(), key=lambda kv: kv[1]):
            delta = f"  {100 * (base - n) / base:+.2f}%" if base else ""
            print(f"  {name:<40} {n:,}{delta}")

    print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    print("NOT covered here: nanochat pretrain + CORE/bpb eval (needs torch + a GPU).")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    app()
