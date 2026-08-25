# N-gram tokenizer evaluation

Held-out **bits per byte** for a tokenizer, under an interpolated modified Kneser-Ney
n-gram model fitted to that tokenizer's own token stream. CPU only, minutes per arm.

The gap this fills: intrinsic metrics (compression, Rényi efficiency, vocabulary
statistics) are cheap but known to correlate unreliably with downstream results, and the
downstream eval in `eval/py-nanochat` needs an H100 and hours per arm. An n-gram LM sits
between them, and the order `n` is the dial: at `n=1` the metric is close to the existing
intrinsic measures, and each increment lets the model use sequential structure that
otherwise only a trained LM would see.

```bash
# score the MinGram paper's nine arms at orders 1-5
uv run python paper_utils/hybrid/ngram/run_ngram_eval.py --corpus fineweb_en_5gb

# then ask whether any of it predicts the GPU runs
uv run python paper_utils/hybrid/ngram/correlate_ngram_downstream.py \
    --ngram-tsv results/ngram/ngram_bpb_fineweb_en_5gb.tsv --seed-spread
```

## What the number is

Total bits the model spends on the held-out token stream, divided by the **true UTF-8
byte length of the held-out text**, measured before tokenization. Two consequences worth
knowing:

- There is no `byte_factor` to correct for. nanochat divides by the summed byte length of
  each target token decoded *in isolation*, which under-counts for any scheme that
  rebuilds a character from two touching tokens, and `pynanochat.runner` has to measure a
  correction factor to make its numbers comparable across tokenizers. Scoring against the
  real text removes that step. A *lossy* tokenizer would be flattered instead, so
  `roundtrip_ok` is reported alongside.
- Vocabulary size is priced. The model interpolates down to a uniform over the
  tokenizer's whole emittable alphabet, so tokens a vocabulary never earns back cost it
  bits. That is a property the metric should have.

## Caveats

- **Context is measured in tokens.** A trigram over a high-compression vocabulary sees
  several times more bytes of history than one over a byte-level vocabulary. This mirrors
  how a fixed-context transformer rewards compression, but it does structurally favour
  long tokens.
- **An n-gram cannot see inside a token.** Whatever compositional structure a transformer
  recovers from subword pieces is invisible here.
- **Rankings can move with the training-corpus size**, because a larger vocabulary means
  sparser counts. Check stability across at least two budgets before trusting an order.
- **`PretokenizedCorpus` is unusable as a source.** It stores a `Counter` of chunks, so
  sequence order is gone; `text.py` streams raw documents instead.

## Layout

| file | what it does |
|---|---|
| `counts.py` | n-gram counting via dense prefix ids — one `np.unique` per order, no Python loop over positions |
| `kn.py` | interpolated modified Kneser-Ney: discounts, interpolation weights, scoring |
| `text.py` | order-preserving raw-text sources and the disjoint train/eval split |
| `evaluate.py` | encode → fit → score → `NgramResult`, with an encoding cache |

Correctness rests on the model being a genuine probability distribution — otherwise the
bits are not a code length and a tokenizer could win by leaking mass. `tests/ngram/`
checks that it normalizes to 1 over the alphabet, that counting matches brute force, and
that the interpolation weights equal the mass the discounts removed.
