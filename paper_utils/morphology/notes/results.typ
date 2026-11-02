=== Classical segmenters against the SIGMORPHON reference

English, Goldfish frequency bands. Cells are boundary F1 (precision/recall) against
SIGMORPHON 2022 projected to surface. Supervised methods train on SIGMORPHON words outside the
top 50k. `CELEX` is not a method: it is the second human-derived gold, scored against the first,
so its row is the agreement between two references.

#table(columns: 4, align: (left,right,right,right), stroke: 0.4pt,
  [], [top 1k], [1k-10k], [10k-50k],
  [CELEX vs SIGMORPHON], [0.54 (0.86/0.39)], [0.67 (0.72/0.62)], [0.74 (0.75/0.72)],
  [no split], [0.00 (1.00/0.00)], [0.00 (1.00/0.00)], [0.00 (1.00/0.00)],
  [Morfessor, $alpha = 1$], [0.12 (1.00/0.06)], [0.22 (0.96/0.12)], [0.31 (0.95/0.18)],
  [Morfessor, $alpha = 0.3$], [0.54 (0.81/0.40)], [0.66 (0.80/0.57)], [0.64 (0.64/0.65)],
  [Morfessor, $alpha = 0.1$], [0.58 (0.51/0.66)], [0.60 (0.50/0.75)], [0.53 (0.41/0.76)],
  [Morfessor, $alpha = 0.03$], [0.35 (0.23/0.76)], [0.41 (0.28/0.78)], [0.41 (0.28/0.79)],
  [Morfessor semi-sup. (1k)], [0.59 (0.73/0.50)], [0.59 (0.77/0.48)], [0.62 (0.79/0.51)],
  [CRF (1,000)], [0.67 (0.69/0.65)], [0.71 (0.75/0.67)], [0.70 (0.78/0.63)],
  [CRF + Morfessor (1,000)], [0.66 (0.72/0.61)], [0.71 (0.82/0.62)], [0.66 (0.83/0.55)],
  [CRF (10,000)], [0.73 (0.71/0.77)], [0.78 (0.78/0.78)], [0.78 (0.81/0.75)],
  [CRF + Morfessor (10,000)], [0.73 (0.73/0.72)], [0.78 (0.82/0.75)], [0.77 (0.84/0.71)],
  [CRF (50,000)], [0.78 (0.77/0.80)], [0.85 (0.84/0.85)], [0.84 (0.86/0.83)],
  [CRF + Morfessor (50,000)], [0.79 (0.78/0.80)], [0.85 (0.86/0.83)], [0.83 (0.87/0.80)])


Unsupervised Morfessor tops out at 0.67 F1 at its best corpus weight, which is the range the
literature reports. A CRF with Ruokolainen features learns from labelled words steadily (0.71
at 1k, 0.85 at 50k); Morfessor features buy precision for recall and no F1. The two human-derived
references agree with each other at only 0.67 F1 in the 1k-10k band, so 0.85 against one of them
is close to what any method can show on this yardstick.


=== LLM annotation

Prompt: typed surface segmentation (`llm_segment.TYPED_PROMPT`), boundaries marked `+`
inflection, `-` derivation, `=` compound, with explicit conventions for depth (split all the
way down), bound roots (split an affix only if the base is a real related word) and spelling
changes (affixes keep their dictionary form, the stem absorbs the change: `ignor+ed`,
`patch+es`). An earlier untyped prompt left depth open and Sonnet split only the outer layer
(F1 0.78 against SIGMORPHON, against 0.85 typed).

Pairwise boundary F1 on 290 words drawn from the top 10k and present in SIGMORPHON. F1 is
symmetric, so each cell reads as agreement. Claude models run through `claude -p`, the others
through the Copilot CLI.

#table(columns: 9, align: (left,right,right,right,right,right,right,right,right), stroke: 0.4pt,
  [], [Haiku], [Haiku (run 2)], [Sonnet], [Gemini 3.8 Flash], [GPT-6 Astra], [SIGMORPHON], [CRF (50k)], [CELEX],
  [Haiku], [--], [0.95], [0.93], [0.95], [0.93], [0.85], [0.83], [0.71],
  [Haiku (run 2)], [0.95], [--], [0.93], [0.96], [0.93], [0.86], [0.84], [0.74],
  [Sonnet], [0.93], [0.93], [--], [0.97], [0.92], [0.85], [0.82], [0.70],
  [Gemini 3.8 Flash], [0.95], [0.96], [0.97], [--], [0.94], [0.86], [0.83], [0.73],
  [GPT-6 Astra], [0.93], [0.93], [0.92], [0.94], [--], [0.86], [0.81], [0.74],
  [SIGMORPHON], [0.85], [0.86], [0.85], [0.86], [0.86], [--], [0.86], [0.72],
  [CRF (50k)], [0.83], [0.84], [0.82], [0.83], [0.81], [0.86], [--], [0.68],
  [CELEX], [0.71], [0.74], [0.70], [0.73], [0.74], [0.72], [0.68], [--])


LLMs agree with each other at 0.93-0.97, across model families as much as within one, and with
either human-derived reference at 0.70-0.86. Two runs of the same model agree at 0.95-0.97, so
sampling noise is a small part of the disagreement. By boundary type, inflection is almost
settled and derivation is where the remaining disagreement lives:

#table(columns: 2, align: (left,right), stroke: 0.4pt,
  [boundary type], [mean pairwise F1 (range)],
  [inflection], [0.98 (0.96-1.00)],
  [derivation], [0.88 (0.83-0.94)],
  [compound], [0.92 (0.90-1.00)])


Of the 256 words where all four models (Haiku, Sonnet, Gemini, GPT; three families) agree exactly, 29 differ from SIGMORPHON. Most
are reference errors (`alex|a`, `gen|ome`, `line|n`, `los|t`, `give|a|way`) or artefacts of
projecting canonical gold to surface (`ignore|d`, `patche|s`, `consume|r|s`, where the LLM
convention is `ignor+ed`, `patch+es`, `consum-er+s`). A few go the other way and are debatable
deep Latinate splits (`typ-ic-al`, `athlet-ic-s`). The 35 words where the LLMs disagree among
themselves are bound roots and depth (`cognit-ive`, `re-mind+ed`, `proto-type`), which a
per-boundary vote resolves.
