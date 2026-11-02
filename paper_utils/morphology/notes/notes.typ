#set document(title: "Morphological alignment: working notes")
#set page(margin: 2.2cm, numbering: "1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#show link: underline

#align(center)[
  #text(16pt, weight: "bold")[Morphological alignment of tokenizers: working notes]

  Branch `morphology`, `script_tok`. Internal, not for publication. Started 2026-11-02.
]

#outline(depth: 2)

= Where this started

Two observations from the boundary paper:

+ *Probing a word on its own is not neutral.* MorphScore and MorphAlign both tokenize the gold
  word without context. Under a leading-space vocabulary the bare string `leads` is not what
  the tokenizer sees in running text (` leads`), so it is split (`lead|s`) where in context it is
  one token. English plain BPE: 7.9% of MorphScore words are a single token on the bare probe,
  64.7% in context.
+ *How whole-word tokens are credited decides the sign of the comparison.* MorphScore either
  drops single-token words (`exclude_single_tok=True`) or gives them a full point. The credited
  variant tracks single-token share at r = 0.95 across the grid.

Both metrics are weak in other ways, and existing gold is sparse and inconsistent, which led to
the question in @silver: can we build our own silver gold for the most frequent words of a
corpus?

= Metrics in use

== MorphScore
Arnett & Bergen (COLING 2025) and Arnett, Hudspeth & O'Connor (TokShop 2025, 70 languages).
Gold comes from Universal Dependencies: the stem is the substring of the word form shared with
the lemma, the remainder is a prefix and/or suffix, so each item has one or two boundaries and
is mostly inflectional. Scoring is per word, macro-averaged; recall is 1 if all gold boundaries
are hit, 0.5 if one of two.

*Bug, upstream and in our port* (fixed in `catherinearnett/morphscore#3`, and in
`script_bpe/analysis/morphscore.py` on this branch): the end of the word was counted as a
predicted boundary, so a perfect `lead|s` scored precision 1/2 and an unsplit word (credited)
scored 1. Recall is unaffected. Re-scoring the English boundary grid moves precision and F1
up, mostly for `plain`, and changes no ranking.

== MorphAlign
Stephen & Libovický (arXiv 2601.18536), the `eval/morph-tok-eval` submodule. Gold-free with
respect to segmentation: an IBM Model 1 alignment between subwords and UD morpho-syntactic
features, scored by how well subwords predict features. No notion of boundary position; an
unsplit word can only score zero; the scale differs by about 30x between languages. Our
MinGram-paper usage also probes the bare word.

== Boundary P/R/F1 (Creutz & Lindén 2004)
Used by Uzan et al. (ACL 2024) and Gow-Smith et al. (EMNLP 2022). Micro precision and recall
over word-internal boundaries. An unsplit word loses recall on its gold boundaries and adds
nothing to precision, which is the behaviour we want without a credit switch. Gow-Smith argue
precision is the more meaningful number because storing frequent words whole is legitimate.

= Silver gold for frequent words <silver>

== Goal
A gold file of surface segmentations (morphs concatenate to the word) for the top 10k-100k
words of a web corpus, per language, with boundary types so a metric can choose its depth.

== Background: Morfessor
Morfessor (Creutz & Lagus 2002, 2007; Morfessor 2.0, Virpioja et al. 2013) is the standard
unsupervised morphological segmenter. It models a corpus as a sequence of morphs drawn from a
lexicon and picks the lexicon by a two-part cost, in MDL or MAP form:

$ L(theta, D) = underbrace(-log P(theta), "lexicon cost") + underbrace(-log P(D | theta), "corpus cost") $

The corpus cost is a unigram model over morphs, exactly as in the Unigram LM tokenizer. The
lexicon cost charges for every character of every morph type in the lexicon, so adding a morph
must pay for itself by shortening the corpus encoding. *Morfessor Baseline* optimizes this
greedily: for each word type it tries every binary split recursively and keeps the cheapest;
segmentation of new words is Viterbi over the learned lexicon.

Relation to our tokenizers: Unigram LM (SentencePiece) has the same corpus model but replaces
the lexicon prior with a hard vocabulary size and EM plus pruning. Morfessor has no target
vocabulary size; the trade-off is set by a corpus weight $alpha$ multiplying the corpus cost.
Trained on token frequencies it keeps frequent words whole (they are cheap to store); trained
on type counts, or log-dampened counts as here, it segments more.

Variants: *semi-supervised Morfessor* adds a likelihood term for annotated segmentations with
its own weight; *FlatCat* (Grönroos et al. 2014) adds an HMM over morph categories (prefix,
stem, suffix, non-morph) and is among the best on Morpho Challenge in semi-supervised mode.

Caveat raised during this work: silver gold learned with Morfessor from the same kind of corpus
would reward tokenizers for behaving like Morfessor, which favours Unigram and MinGram. Morfessor
is therefore only a baseline and a CRF feature here, never the source of gold.

== Existing gold and how it is built
- *Morpho Challenge 2005-2010*: about 1000 randomly sampled words per language (en, fi, de, tr),
  canonical analyses from CELEX and FINTWOL.
- *SIGMORPHON 2022*: canonical segmentation, Wiktionary-derived, 9 languages (no de, fi).
  Winner 97.3 macro F1 (char seq2seq); Morfessor baseline 25.6 on this canonical task.
- *UniSegments 1.0*: 38 harmonised datasets, 30 languages, licences per dataset.
- *MorphyNet*: Wiktionary, 15 languages, mixed surface and lemma-based segmentations.
- *CRF boundary taggers* (Ruokolainen et al. 2013, 2014): 1000 annotated words give 85.9 (en)
  and 83.8 (fi) boundary F1; adding Morfessor features gives 88.4 and 89.4.
- *LLM annotation*: thin evidence. Prompted frontier LLMs are "inconsistent" (Morozov et al.
  2026); Claude 3.7 under-segments Bribri. No large silver set exists.

== Data
Word frequencies: first 1 GB of each Goldfish corpus, lowercased letter runs
(`word_counts.py`). English: 168M tokens, 1.1M types.

Reference: SIGMORPHON 2022 English word gold, projected from canonical to surface by character
alignment (`gold.py`), surface-only letters attached to the left morph. 70% of English rows
concatenate exactly; Russian only 11% (lemma-based), so the projection is shaky there.

Coverage of the most frequent words is poor:

#table(columns: 3, align: (left, right, right),
  [top-k], [SIGMORPHON], [CELEX (morph-tok-eval)],
  [1k], [26%], [70%],
  [10k], [71%], [51%],
  [100k], [63%], [19%])

and the sources disagree: CELEX keeps `leads` and `understanding` whole, SIGMORPHON gives
`lead|s` and `under|stand|ing`; SIGMORPHON has no `running` or `walked`.

== Experiments
#include "results.typ"

= Open questions
- Which boundary depth should a tokenizer metric use: inflection only, all, or report each?
- Paradigm consistency (does ` lead+s` reuse ` lead`?) as a metric that needs no segmentation.
- In-context probing for every metric.
