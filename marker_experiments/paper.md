# Boundary markers: removing with/without-space token duplication from SCRIPT tokenizers

**Status:** working draft. All numbers are measured; the FineWiki 6-language table is
filled in from `multilang_result.json` once `multilang_grid.py` completes.

## Abstract

Byte-pair and unigram vocabularies built over space-separated writing systems spend a
large fraction of their capacity representing the same word twice — once with a leading
space and once without (`' the'` and `'the'`). We measure this at **20–43% of the
vocabulary**, and **64% of all emitted tokens**, depending on corpus and vocabulary size.
We propose replacing the leading-space convention with an explicit boundary marker token
`<|>`: word spans are wrapped unconditionally, so a word has one canonical form
everywhere, and the single space between two adjacent spans is *elided* at encode time
and reconstructed at decode time from the resulting pair of touching markers. Applied to
words alone the scheme costs 7.5% compression. Extended to punctuation it costs 1.0–1.6%.
Extended further to digits it **overtakes the baseline**, reaching +1.17% chars/token at
64k vocabulary on a mixed prose+code corpus while reducing duplicate pairs from 13,480 to
120 and training faster than the baseline. The result is a tokenizer with a canonical word
form, ~41% of vocabulary freed, and no compression penalty.

## 1. The duplication we are trying to remove

SCRIPT encoding (`ScriptEncodingV3`, registry name `scriptenc3_cb`) pretokenizes text into
script/category runs and lets a lone space attach to the following span, which is what
produces the familiar `' word'` / `'word'` pair. Measured on 80M characters of FineWeb
English at 16k additional vocabulary:

| | value |
|---|---|
| duplicate `' X'`/`'X'` pairs | 1,792 |
| vocabulary slots consumed | 3,584 (**22.4%**) |
| share of all emitted tokens | **64.23%** |
| largest pairs | `' .'`/`'.'` 700,931 · `' ,'`/`','` 680,504 · `' the'`/`'the'` 639,584 |

The tax grows with vocabulary, because every additional word costs two slots rather than
one: 20.2% → 24.9% → 30.0% at 16k/32k/64k on prose, and 38.0% → 40.9% → **41.0%** on a
mixed prose+code corpus.

An important caveat, which took us a while to accept: this duplication is not *waste* in
compression terms. Both forms are used, and they earn their slots. Section 4 shows the
baseline compresses *better* than an early marker scheme despite spending 30% of its
vocabulary on duplicates. The motivation for removing duplication is representational —
one canonical form per word, and capacity freed for genuinely distinct content — not a
free compression win. Getting to *no compression cost* took three design iterations.

## 2. Method

A single atomic token `<|>` is added to the pretokenizer's vocabulary. Text is grouped into
units (maximal script/category runs, absorbing inherited marks), each classified as

- **word** — a script in `DEFAULT_SCRIPTS_LM_WITH_SPACES` with category `LM`
  (the 20 space-using writing systems: Latin, Cyrillic, Arabic, Hangul, Devanagari, …),
- **punct** — `combines_with_spaces` but not a letter (V3's `(✭, PSF)` blocks),
- **digit** — category `N` (v5 only; see §3),
- **space** — exactly one space character,
- **other** — everything else (multi-space runs, newlines, Han, emoji).

Marker placement:

- a **word** span carries `<|>` on **both** sides, unconditionally;
- **punct** and **digit** units carry `<|>` only on a side whose adjacent single space was
  elided;
- **other** units and non-elided spaces are emitted exactly as the baseline emits them.

A single space is elided when **both** neighbours are markable. Its two neighbours then
carry facing markers, which land adjacent in the atomic stream. Decoding is therefore:

```
<|> immediately followed by <|>   ->  emit exactly one space
a lone <|>                        ->  emit nothing (structural boundary)
```

This is unambiguous precisely because only word spans are marked unconditionally. Had
punctuation also been wrapped unconditionally, `a,b` would encode as
`<|>a<|> <|>,<|> <|>b<|>` — touching markers at both junctions, indistinguishable from
`a , b`. Marking punctuation only on a space side gives:

```
a,b     ->  <|>a<|>  ,     <|>b<|>          (no touch, no space)
a, b    ->  <|>a<|>  ,<|>  <|>b<|>          (one touch, one space)
a = b   ->  <|>a<|>  <|>=<|>  <|>b<|>       (two touches, two spaces)
a ,b    ->  <|>a<|>  <|>,   <|>b<|>         (asymmetric, correct)
```

**Word-adjacency exception.** Two *different* word scripts directly adjacent with no space
would both emit unconditional markers, they would touch, and decode would insert a phantom
space. We initially dismissed this as vanishingly rare; it is not, once code is in scope —
Greek letters as mathematical identifiers (`sπ`, `upperΔ`, `Δx`, `łʒλπ` in Julia, Tcl and
Python) broke **5 of 500** held-out code documents. A word therefore drops its *opening*
marker when the preceding unit is also a word, which (since same-script runs are absorbed
into one unit) can only mean a script change with no space. The preceding word keeps its
closing marker, which becomes a lone marker and decodes to nothing.

**Merge constraint.** `bpe_merge_allowed` forbids any merge across two touching markers.
Without it, BPE learns tokens like `'<|>the<|><|>'` that swallow the dangling half of the
*next* word's opening marker, reintroducing per-word duplication keyed on what follows
instead of what precedes.

**Chunking.** Units are *not* fused across an elided space. Given the merge constraint,
fusing admits exactly the same set of legal merges, while inflating the corpus from 335k to
**1.76M** unique chunks and BPE training from 40s to **442s**. Decoding reads the flat
atomic stream, so chunk boundaries do not affect reconstruction — the markers still touch
across a chunk boundary. This one change accounts for a ~10× training speedup.

## 3. Design iterations

| variant | prose 16k chars/token | vs plain | note |
|---|---|---|---|
| plain `scriptenc3_cb` | 3.9909 | — | baseline |
| v1: rename the merged space token | 3.9909 | 0.00% | no-op: BPE's *first* learned merge already reattaches the space to the same token id, so pre-collapsing it changes nothing |
| v2: hard chunk boundary, space left in place | 2.4201 | −39.4% | not the proposal — leaving the literal space and merely forbidding merges across it destroys compression |
| v3: words only, space elided, merge-ban | 3.6920 | −7.49% | correct mechanism, words only |
| **v4: + punctuation** | **3.9266** | **−1.61%** | closes 79% of v3's gap |
| **v5: + digits** | see §4 | **positive** | overtakes the baseline at ≥32k |

The step from v3 to v4 is worth dwelling on. v3 left punctuation unmarked, so `', b'`
became an unmarked `,` **plus a standalone `' '` token** plus `<|>b<|>`. Marking
punctuation on the space side turns that into `,<|>` + `<|>b<|>`, eliminating the bare
space. This closed 79% of the gap *without* improving word coverage at all (7,717 vs
v3's 7,752 single-token words) — the entire gain came from removing bare spaces.

The step from v4 to v5 was found by measuring which spaces v4 *failed* to elide:

| domain | single spaces | elided (v4) | not elided | of those, digit-adjacent |
|---|---|---|---|---|
| code | 42,242 | 87.5% | 12.5% | **97.9%** |
| prose | 287,220 | 96.7% | 3.3% | **98.7%** |

Digits account for ~98% of every miss. The cause is structural: `script_category_v3` folds
`L/M → LM`, `Z/Cc → ZC`, `So → So` and `P/S/Cf → PSF`, but leaves category `N` untouched,
so digits land in `(✭, N)` blocks that are absent from `script_cat_with_spaces` and were
therefore non-markable. Making digits markable under the same asymmetric rule as
punctuation drops non-elided spaces to **0.3%** (code) and **0.0%** (prose).

Digits and punctuation are safe to mark because they are *closed* sets, so the variant cost
is bounded — at 64k, marker variants occupy 993 slots (~1.5% of vocabulary). We deliberately
did **not** extend marking to Han, emoji or other open-set scripts, which account for only
~2% of remaining misses.

## 4. Results

Setup: `ScriptEncodingV3` with `enforce_char_boundaries=True` throughout. BPE is the
repository's greedy trainer; MinGram uses `overshoot_factor=1.15`, the repository's own
working default. Metric is characters per token on held-out documents (higher is better).
Roundtrip is verified on every held-out document of every cell.

### 4.1 FineWeb English prose (80M chars train, 1000 held-out docs)

| vocab | plain BPE | v4 BPE | gap | plain MinGram | v4 MinGram | gap |
|---|---|---|---|---|---|---|
| 16k | 3.9909 | 3.9266 | −1.61% | 4.0474 | 3.9670 | −1.99% |
| 32k | 4.2611 | 4.2170 | −1.03% | 4.3012 | 4.2483 | −1.23% |
| 64k | 4.4412 | 4.4032 | −0.86% | 4.4678 | 4.4204 | −1.06% |

Duplicate pairs collapse from 1,792/4,201/9,871 to 1/2/2, i.e. 20–30% of vocabulary to
0.01%. MinGram is uniformly ~1.0–1.4% better than BPE for both pretokenizers, and does not
change the ordering. v5 was not run on this corpus; §4.2 and §4.3 cover it.

### 4.2 Mixed prose + code (40M chars FineWeb + 40M chars code, 500 held-out docs per domain)

Code half: `codeparrot/codeparrot-clean-valid` whole Python files plus
`christopher/rosetta-code` multi-language snippets. No whitespace normalization.
Gaps are relative to plain; positive means **better than the baseline**.

| trainer | vocab | mixed v4 → v5 | prose v4 → v5 | code v4 → v5 |
|---|---|---|---|---|
| BPE | 16k | −2.13 → **−0.10** | −1.60 → **+0.05** | −3.71 → **−0.56** |
| BPE | 32k | −1.64 → **+0.67** | −1.17 → **+0.79** | −2.99 → **+0.33** |
| BPE | 64k | −1.33 → **+1.17** | −1.00 → **+1.13** | −2.28 → **+1.29** |
| MinGram | 16k | −2.40 → **−0.36** | −2.00 → **−0.32** | −3.60 → **−0.48** |
| MinGram | 32k | −1.88 → **+0.42** | −1.40 → **+0.55** | −3.25 → **+0.05** |
| MinGram | 64k | −1.44 → **+1.07** | −1.16 → **+0.98** | −2.26 → **+1.33** |

v5 crosses over between 16k and 32k and the advantage is still growing at 64k. At 64k BPE:

| | chars/token | duplicate pairs | vocab on duplicates | train time |
|---|---|---|---|---|
| plain | 4.0087 | 13,480 | **41.0%** | 108s |
| **v5** | **4.0557** | **120** | **0.4%** | **92s** |

So v5 is simultaneously better compressing, ~100× lower in duplicate pairs, 41% of
vocabulary freed, and faster to train.

### 4.3 FineWiki, 6 languages

Languages are those of `FINEWIKI_HYBRID6_CORPORA`: **en, de, fi, ru, ar, ko** — three
Latin, plus Cyrillic, Arabic and Hangul; all six are space-using scripts and hence
word-wrapped by the marker schemes. Setup matches the registry's `finewiki_{lang}_1gb`
(same dataset, same `normalize_whitespace` transform, which collapses `[ \t]+` and so
removes multi-space runs by construction) except for a smaller per-language character
budget, for compute reasons; see `multilang_grid.py` for the exact deviation.

*(table inserted from `multilang_result.json` on completion)*

## 5. Why the early versions cost anything at all

The v3 penalty was not an artifact of a bad merge order, and larger candidate pools do not
fix it. Sweeping MinGram's BPE-init overshoot over 1.10 / 1.15 / 1.25:

| f | plain | v3 | gap | v3 single-token words |
|---|---|---|---|---|
| 1.10 | 4.0455 | 3.7286 | −7.83% | 8,674 |
| 1.15 | 4.0474 | 3.7282 | −7.89% | 8,856 |
| 1.25 | 4.0470 | 3.7252 | −7.95% | 8,964 |

A 2.5× increase in overshoot budget bought 290 words and made compression marginally
*worse*. Switching from BPE to MinGram does recover a one-time ~900 words (7,752 → 8,674)
by pruning dead intermediates that permanently occupy BPE vocabulary slots, but that is
~900 of a ~5,000-word deficit, and further overshoot adds almost nothing.

The real cost of v3 was **bare space tokens**, not vocabulary accounting — which is why
marking punctuation (v4) and then digits (v5) recovered it entirely, while overshoot could
not. Bucketing the token stream on held-out code makes this concrete; v4's +3,256-token
deficit versus plain decomposes as:

| bucket | delta | share |
|---|---|---|
| whitespace | +1,708 | 52% |
| alpha | +721 | 22% |
| punct/symbol | +434 | 13% |
| marker-only (empty) | +382 | 12% |
| digit | +11 | 0% |

The whitespace term is *not* indentation. Multi-space runs are ~1 token either way: pure
space tokens of length > 1 account for **1.24%** of code tokens, with identical counts
under plain and v4, and BPE folds them into `"\n    "`-style tokens that run at 4.59
chars/token — better than alpha's 3.91. The whitespace deficit is entirely the digit case:
in `1 item`, plain absorbs the space into `' item'` while v4 cannot elide it (a digit is
not markable) and must emit a standalone `' '`. Counting the cases where the following unit
is markable but the preceding is not gives 1,834, against the measured +1,708 — the
mechanism accounts for the gap, and v5 removes it.

## 6. Limitations

- **Below ~16k vocabulary v5 is at parity or slightly behind.** The crossover sits between
  16k and 32k on the mixed corpus. Small-vocabulary settings do not benefit.
- **Marker-only tokens are pure overhead**: 382 emissions (0.27% of code tokens) are lone
  `<|>` carrying no characters.
- **One roundtrip failure per code cell, for plain and markers alike**: `U+F8FF`
  (private use) is absent from the V3 `char_encoding` and is silently dropped. Pre-existing,
  unrelated to this work, but it means "0 failures" is not achievable on that corpus without
  a script-config fix.
- **Adjacent word scripts with no separator** are handled by dropping one marker, which
  means those specific words lose the canonical-form guarantee. Rare (~1% of code docs,
  ~0% of prose docs) but not zero.
- **No language-modelling evaluation.** Everything here is compression and vocabulary
  structure. Whether a canonical word form helps or hurts downstream LM quality is exactly
  the question these numbers cannot answer, and is the obvious next experiment.
- **Open-set scripts are unmarked.** Han, emoji and other non-space scripts keep baseline
  behaviour, so ~2% of single spaces remain non-elided. CJK-heavy corpora were not studied.

## 7. Reproduction

```
marker_experiments/
  scriptenc_marker_v4.py   # words (unconditional) + punctuation (space side)
  scriptenc_marker_v5.py   # v4 + digits (space side)
  multilang_grid.py        # FineWiki 6-language grid, resumable
  prior_results.json       # English-prose and mixed-code numbers
  multilang_result.json    # FineWiki 6-language numbers
```

`v5` subclasses `v4` and overrides a single hook (`_extra_markable_script_ids`), so the two
schemes differ by exactly the digit set. Both are ordinary `ScriptPretokenizer` subclasses
and need no changes to the trainers.

One practical note for anyone extending this: `Pretokenizer.hash()` is derived from the
config only, so two prototypes whose *behaviour* differs but whose config fields match will
collide in the pretokenized-corpus cache. Use distinct corpus names per prototype.
