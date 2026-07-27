# Boundary markers: one canonical word form, and better compression, for SCRIPT tokenizers

**Status:** draft. The FineWiki 1 GB BPE results (§4.1) are complete; the MinGram
half (§4.2) is running and its table is marked pending. Results at smaller scale
(§4.3) come from an earlier per-script variant of the scheme and are labelled so.

## Abstract

Subword vocabularies built over space-separated writing systems spend a large
fraction of their capacity representing the same word twice — once with a leading
space and once without (`' the'` and `'the'`). Across six languages at 1 GB each we
measure this at **18–30% of a 32,768-entry vocabulary**, and at 16k vocabulary on web
text it accounts for **64% of all emitted tokens**. We replace the leading-space
convention with an explicit boundary marker `<|>`: spans are delimited, and the single
space between two adjacent delimited spans is elided at encode time and reconstructed
at decode time from the resulting pair of touching markers. *Which* units get delimited
decides everything. Delimiting words alone costs **−13.75%** compression on average;
adding punctuation brings it to **−2.99%**; adding digits makes it **+2.14%**, beating
the baseline in **all six languages** (range +0.88% to +3.77%) while reducing duplicate
vocabulary pairs from thousands to fewer than ten, with zero roundtrip failures and no
increase in training cost. The result is a tokenizer with one canonical form per word,
~20% of vocabulary reclaimed, and *better* compression than the convention it replaces.

## 1. The duplication

SCRIPT encoding (`ScriptEncodingV3`, registry name `scriptenc3_cb`) pretokenizes into
script/category runs and lets a lone space attach to the following span, producing the
`' word'` / `'word'` pair. Measured on FineWiki, 1 GB per language, 32,768 additional
vocabulary, BPE:

| lang | duplicate pairs | share of vocabulary |
|---|---|---|
| en | 3,196 | 18.5% |
| de | 3,226 | 18.7% |
| fi | 3,835 | 22.2% |
| ru | 3,297 | 19.1% |
| ar | 3,159 | 18.3% |
| ko | 5,244 | **30.4%** |

At 16k vocabulary on 80M characters of FineWeb the same measurement gives 1,792 pairs
occupying 22.4% of the vocabulary and accounting for **64.23%** of all emitted tokens —
the largest being `' .'`/`'.'` (700,931), `' ,'`/`','` (680,504), `' the'`/`'the'`
(639,584).

A caveat we had to accept early: this duplication is not *waste* in compression terms.
Both forms are used and both earn their slots; §3 shows the baseline beating an early
marker scheme by 7.5% despite spending 30% of its vocabulary on duplicates. The
motivation is representational — one form per word, capacity freed for distinct content.
That the final scheme also compresses *better* is a separate result, and it took three
iterations to reach.

## 2. Method

One atomic token `<|>` is added. Text is grouped into maximal script/category runs, then
collected into units:

- **word** — a maximal run of characters from *any* space-using script
  (`DEFAULT_SCRIPTS_LM_WITH_SPACES`, category LM), **merged across script changes**;
- **punct** — `combines_with_spaces` but not letters (V3's `(✭, PSF)` blocks);
- **digit** — category `N`;
- **space** — exactly one space character;
- **other** — everything else (multi-space runs, newlines, Han, emoji).

Marker placement:

- **word** spans are delimited on **both** sides, unconditionally — the point of the
  scheme: a word looks the same regardless of what precedes it;
- **punct** and **digit** units are delimited only on a side whose adjacent single space
  was elided;
- **other** units and non-elided spaces are emitted exactly as the baseline emits them.

A single space is elided when **both** neighbours are delimited. Decoding is then:

```
<|> immediately followed by <|>  ->  emit exactly one space
a lone <|>                       ->  emit nothing (structural boundary)
```

### 2.1 Why spans merge across scripts

Merging word runs across script changes is what makes the invariant hold without
exceptions. If each script run were delimited separately, `latin` immediately followed by
Cyrillic `кириллица` would put two unconditional markers face to face — indistinguishable
from an elided space — and decode would fabricate one. We found this empirically: Greek
letters used as identifiers (`sπ`, `upperΔ`, `Δx`) broke 5 of 500 held-out code documents.
Merging first means two delimited word spans can never be adjacent, so a touching pair is
unambiguously an elided space:

```
latinкириллица          ->  <|>latin | кириллица<|>          (one span, no inner marker)
123 latinкириллица 123  ->  123<|> | <|>latin | кириллица<|> | <|>123
```

Inside a span the baseline's script split is preserved — the marker rides the first and
last chunk — so no BPE merge crosses a script change the baseline would forbid.

### 2.2 Why punctuation and digits are asymmetric

Delimiting punctuation unconditionally would make `a,b` encode as
`<|>a<|> <|>,<|> <|>b<|>`: touching markers at both junctions, indistinguishable from
`a , b`. Marking only on a side that actually had a space keeps the invariant:

```
a,b   ->  <|>a<|>  ,     <|>b<|>        a, b   ->  <|>a<|>  ,<|>     <|>b<|>
a ,b  ->  <|>a<|>  <|>,  <|>b<|>        a = b  ->  <|>a<|>  <|>=<|>  <|>b<|>
```

The cost is up to four variants per mark (`,` `,<|>` `<|>,` `<|>,<|>`). For punctuation
this is cheap because the set is genuinely closed: 146–225 slots at 32,768 vocabulary,
under 0.7%.

**For digits it is not, and this is a real cost we initially mis-stated.** A digit *unit*
is a whole run, so the marked set is one entry per distinct *number*, not per digit. At
32,768 vocabulary the `bnd_wpd` runs of §4.1 spend, on pure-digit entries:

| lang | plain entries | `bnd_wpd` entries | distinct numbers covered | slots lost to variants |
|---|---|---|---|---|
| en | 1,426 | 1,682 | 589 (vs 1,426) | **1,093 (3.17%)** |
| de | 1,128 | 1,416 | 522 (vs 1,128) | 894 (2.59%) |
| ru | 786 | 1,269 | 445 (vs 786) | 824 (2.39%) |
| ko | 927 | 1,082 | 441 (vs 927) | 641 (1.86%) |

That is the very duplication the scheme exists to remove, reintroduced for numbers: more
entries spent, less than half the coverage. The fix is to bound the markable set with
`digit_handling`, which splits digit runs so only the run's first and last *group* can
carry a marker — 10 markable strings under `SPLIT`, 1110 under `RTL3` (`pretokenizer.py`
registers exactly those). §4.1 used `digit_handling=None`, so its `bnd_wpd` numbers are
achieved *while paying* this 2–3% tax, and should improve once it is removed.

### 2.3 Merge constraint and chunking

`bpe_merge_allowed` forbids any merge across two touching markers. Without it BPE learns
tokens like `'<|>the<|><|>'` that swallow the dangling half of the next span's opening
marker, reintroducing per-word duplication keyed on what *follows*.

Units are not fused across an elided space. Given the merge constraint, fusing admits
exactly the same legal merges while inflating an early prototype's corpus from 335k to
1.76M unique chunks and BPE training from 40s to 442s — ~10× cost for no benefit, since
decoding reads the flat atomic stream and markers still touch across a chunk boundary.

## 3. Design iterations

| variant | prose 16k chars/token | vs plain | why |
|---|---|---|---|
| plain `scriptenc3_cb` | 3.9909 | — | baseline |
| rename the merged space token | 3.9909 | 0.00% | no-op: BPE's *first* learned merge already reattaches the space to the same token id |
| hard chunk boundary, space kept | 2.4201 | −39.4% | not the proposal: forbidding merges across a retained space destroys compression |
| words only | 3.6920 | −7.49% | right mechanism, wrong scope |
| + punctuation | 3.9266 | −1.61% | removes bare space tokens after punctuation |
| + digits | see §4 | positive | removes the last non-elided spaces |

The words-only → punctuation step closed 79% of the gap *without improving word coverage
at all* (7,717 vs 7,752 single-token words). The gain came entirely from eliminating bare
space tokens: with only words delimited, `', b'` emits an unmarked `,`, a standalone
`' '`, and `<|>b<|>`, where the baseline absorbs the space into `' b'`.

The punctuation → digits step was found by measuring which spaces the scheme *failed* to
elide:

| domain | single spaces | elided | not elided | of those, digit-adjacent |
|---|---|---|---|---|
| code | 42,242 | 87.5% | 12.5% | **97.9%** |
| prose | 287,220 | 96.7% | 3.3% | **98.7%** |

Digits are ~98% of every miss. The cause is structural: `script_category_v3` folds
`L/M → LM`, `Z/Cc → ZC`, `So → So` and `P/S/Cf → PSF` but leaves category `N` alone, so
digits are neither letters nor `combines_with_spaces` and were invisible to the scheme.
Delimiting them drops non-elided spaces to 0.3% (code) and 0.0% (prose).

## 4. Results

`ScriptEncodingV3` with `enforce_char_boundaries=True` throughout; the four pretokenizers
differ *only* in which units carry a boundary, so the comparison isolates the scheme.
Metric is characters per token on held-out documents (higher is better); roundtrip is
verified on every held-out document of every cell.

### 4.1 FineWiki, 1 GB per language, 32,768 vocabulary, BPE

Languages are those of `FINEWIKI_HYBRID6_CORPORA`. `normalize_whitespace` is applied as
the registry's finewiki loader does, so multi-space runs are absent by construction.
Held-out slice is the last 500 documents.

| lang | script | plain | `bnd_w` | `bnd_wp` | `bnd_wpd` |
|---|---|---|---|---|---|
| en | Latin | 3.8310 | −15.97% | −2.90% | **+3.77%** |
| de | Latin | 4.1110 | −12.56% | −3.59% | **+1.60%** |
| fi | Latin | 3.9836 | −12.64% | −2.94% | **+2.02%** |
| ru | Cyrillic | 3.7955 | −13.72% | −2.35% | **+2.67%** |
| ar | Arabic | 3.9838 | −12.41% | −2.67% | **+1.91%** |
| ko | Hangul | 2.2310 | −15.21% | −3.48% | **+0.88%** |
| **mean** | | | **−13.75%** | **−2.99%** | **+2.14%** |

**Zero roundtrip failures across all 24 cells.** The progression is tight across scripts:
`bnd_w` clusters in −12 to −16%, `bnd_wp` in −2.4 to −3.6%, `bnd_wpd` positive everywhere.

Vocabulary structure at the same setting:

| lang | plain dup pairs | plain vocab on dups | `bnd_wpd` pairs | `bnd_wpd` variant slots |
|---|---|---|---|---|
| en | 3,196 | 18.5% | 4 | 213 |
| de | 3,226 | 18.7% | 4 | 188 |
| fi | 3,835 | 22.2% | 4 | 146 |
| ru | 3,297 | 19.1% | 5 | 185 |
| ar | 3,159 | 18.3% | 6 | 198 |
| ko | 5,244 | 30.4% | 4 | 218 |

Training cost is not a penalty: `bnd_wpd` is at or below baseline time in every language
(en 160s vs 152s, de 286s vs 329s, fi 322s vs 353s, ru 231s vs 241s, ar 200s vs 206s, ko
388s vs 456s) and yields fewer unique chunks (ko 8.60M vs 9.47M).

One metric misleads if read directly: *distinct words with their own token* falls from
~27k (plain) to ~11–16k (`bnd_wpd`). That is not lost coverage — the baseline
double-counts, holding `' the'` and `'the'` as two entries for one word, while the marker
vocabulary holds one canonical form.

### 4.2 FineWiki, 1 GB per language, MinGram

*(pending — 24 cells running, `overshoot_factor=1.15`, the repository's working default)*

At 100M characters per language MinGram reproduced the BPE ordering for an earlier variant
(en +2.89%, ru +3.21%, ko +0.88%), so the effect is not a BPE-specific artifact.

### 4.3 Smaller scale, earlier per-script variant

These predate the span merging of §2.1 and delimited each script run separately. They are
included because they establish scale and domain behaviour, and because the mixed
prose+code corpus is the only place code was studied.

**FineWeb English prose**, 80M chars, 1000 held-out docs, `+punct` variant:

| vocab | plain BPE | +punct | plain MinGram | +punct |
|---|---|---|---|---|
| 16k | 3.9909 | −1.61% | 4.0474 | −1.99% |
| 32k | 4.2611 | −1.03% | 4.3012 | −1.23% |
| 64k | 4.4412 | −0.86% | 4.4678 | −1.06% |

**Mixed prose + code**, 40M FineWeb + 40M code (codeparrot Python files, rosetta-code
snippets), no whitespace normalization, 500 held-out docs per domain, `+digits` against
baseline:

| trainer | vocab | mixed | prose | code |
|---|---|---|---|---|
| BPE | 16k | −0.10% | +0.05% | −0.56% |
| BPE | 32k | +0.67% | +0.79% | +0.33% |
| BPE | 64k | **+1.17%** | +1.13% | +1.29% |
| MinGram | 64k | +1.07% | +0.98% | +1.33% |

The span-merged design at 1 GB (+3.77% for en at 32k) outperforms the per-script design at
100M (+2.63% for en at 32k), so removing the inner boundary cost nothing and gained.

## 5. Analysis

**Overshoot does not substitute for the right boundary set.** Sweeping MinGram's BPE-init
overshoot at 16k on the words-only variant:

| f | plain | words-only | gap | words-only single-token words |
|---|---|---|---|---|
| 1.10 | 4.0455 | 3.7286 | −7.83% | 8,674 |
| 1.15 | 4.0474 | 3.7282 | −7.89% | 8,856 |
| 1.25 | 4.0470 | 3.7252 | −7.95% | 8,964 |

A 2.5× larger candidate pool bought 290 words and made compression marginally *worse*.
MinGram does recover a one-time ~900 words by pruning dead BPE intermediates, but that is
~900 of a ~5,000-word deficit.

**The deficit was bare space tokens, not vocabulary accounting.** Bucketing the token
stream on held-out code, the words+punct variant's +3,256-token deficit versus baseline
decomposes as whitespace +1,708 (52%), alpha +721, punct +434, marker-only +382, digit
+11. The whitespace term is *not* indentation — pure space tokens of length > 1 are 1.24%
of code tokens with identical counts under both schemes, and BPE folds them into
`"\n    "`-style tokens running at 4.59 chars/token. It is the digit case: in `1 item` the
baseline absorbs the space into `' item'` while an undelimited digit blocks elision.
Counting cases where the following unit is delimited but the preceding is not gives 1,834,
against the measured +1,708.

**Korean is the informative weak case.** It has the largest duplicate tax to reclaim
(30.4%) and the smallest gain from reclaiming it (+0.88%). Hangul syllable blocks give
Korean by far the lowest absolute compression (2.23 vs 3.80–4.11 chars/token), so word
spans are short and the two marker tokens are proportionally heavier. The rule that falls
out: the scheme pays in proportion to span length relative to its two markers.

## 6. Limitations

- **No language-modelling evaluation.** Everything here is compression and vocabulary
  structure. Whether a canonical word form helps or hurts downstream quality is the
  obvious next experiment and is not addressed.
- **Space-using scripts only.** All six languages use spaces; the scheme does nothing for
  Han, Thai or other spaceless scripts, which keep baseline behaviour.
- **Open-set scripts are undelimited**, so ~2% of single spaces remain non-elided in mixed
  text. CJK-heavy corpora were not studied.
- **Gains shrink where spans are short** relative to the marker pair (Korean, §5).
- **Marker-only tokens are pure overhead**: 382 emissions, 0.27% of code tokens, carrying
  no characters.
- **§4.3 uses first-N sampling**, not the registry's seeded reservoir sample over the full
  source, and its corpora are 80M chars against `fineweb_en_5gb`'s 5×10⁹.
- **One roundtrip failure per code cell in §4.3, baseline included**: `U+F8FF` is absent
  from the V3 `char_encoding` and is dropped. Pre-existing and unrelated, but zero failures
  is unreachable on that corpus without a script-config fix.
- **Single vocabulary size at 1 GB.** §4.1 is 32,768 only; the smaller-scale runs show the
  advantage growing with vocabulary, but that is not verified at 1 GB.
- **§4.1 trains on its own evaluation slice.** The held-out documents are the last 500 of
  the stream, but the corpus was built from the whole stream, so they are ~1.3% of the
  training data. The leak is identical for every pretokenizer, so the *gaps* in §4.1 are
  unaffected; absolute chars/token is optimistic for all four alike. Fixed for the digit
  axis (§4.4), which withholds them.
- **§4.1 ran with `digit_handling=None`**, so it pays the 2–3% digit-variant tax described
  in §2.2. Combining boundaries with digit splitting is implemented and tested but not yet
  measured at scale; it should only improve the reported figures.

## 7. Reproduction

```
marker_experiments/
  boundary_pretokenizer.py   # BoundaryScriptPretokenizer, boundary_targets config
  test_boundary.py           # 412 tests
  finewiki1gb_grid.py        # 4.1/4.2 grid: resumable, commits each cell with its tokenizer
  finewiki1gb_result.json    # 4.1/4.2 numbers
  multilang_grid.py          # earlier per-script 100M multilingual grid
  multilang_result.json
  prior_results.json         # 4.3 and 5 numbers
  tokenizers/                # every trained tokenizer
```

The three variants are one class differing only in `boundary_targets`, and produce distinct
`hash()` values so they cannot collide in the pretokenized-corpus cache — a trap the
earlier prototypes fell into, since `Pretokenizer.hash()` is config-derived and ignores
behaviour.

A note on running this environment: the container clears the working tree every ~30–60
minutes and caps disk, so the grid streams text rather than staging it, frees each
language's corpora when done, retries transient CDN failures, and commits and pushes every
finished cell. Reading only shard `000_00000` silently under-reads languages whose first
shard is smaller than the budget (Arabic 483M, Korean ~734M); the runner lists and reads
all shards.
