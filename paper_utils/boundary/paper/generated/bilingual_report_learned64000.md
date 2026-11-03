# English/Korean shared-vocabulary ablation

CPU BPE/MinGram ablation using prefixes of the local FineWeb quick caches. The 20M-character runs are exploratory pilots. All schemes share the same text within a mixture. Budgets and vocabulary sizes are recorded below. Evaluation uses fixed prefixes of the cached English and Korean Goldfish slices. Compression counts characters after the tokenizer's normalization. Mixture ratios count characters, not documents or UTF-8 bytes.

Allocation counts below cover learned pieces only. Latin/Hangul are script labels, not exclusive language ownership. Duplicate surplus counts use n−1 per equivalent group, retaining boundary markers. Space/case columns overlap and must not be added. These counts do not measure all possible positional redundancy. Space surplus includes whitespace-only pairs, such as newline versus space+newline; per-script counts are in the JSON. Individual SCRIPT atomics may be undecodable fragments, so usage in that category does not imply failed text round trips.

| Train chars | Trainer | Scheme | Latin | Hangul | Other scripts | Nonlexical | Atomics | English chars/token | Korean chars/token |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2,000,000,000 (50% en) | mingram | plain | 21,530 | 40,557 | 420 | 1,493 | 1,710 | 4.3675 | 2.4226 |
| 2,000,000,000 (50% en) | mingram | bnd_w | 21,146 | 41,163 | 378 | 1,313 | 1,711 | 3.9499 | 2.2121 |
| 2,000,000,000 (50% en) | mingram | bnd_wp | 21,076 | 41,020 | 377 | 1,527 | 1,711 | 4.2786 | 2.3416 |
| 2,000,000,000 (50% en) | mingram | bnd_wpd | 21,022 | 40,863 | 376 | 1,739 | 1,711 | 4.3583 | 2.3877 |
| 2,000,000,000 (50% en) | mingram | bnd_w_caps | 20,748 | 41,535 | 387 | 1,330 | 1,713 | 3.9674 | 2.2133 |
| 2,000,000,000 (50% en) | mingram | bnd_wp_caps | 20,691 | 41,388 | 386 | 1,535 | 1,713 | 4.2993 | 2.3433 |
| 2,000,000,000 (50% en) | mingram | bnd_wpd_caps | 20,627 | 41,234 | 381 | 1,758 | 1,713 | 4.3799 | 2.3899 |
| 5,000,000,000 (80% en) | mingram | plain | 34,065 | 28,028 | 318 | 1,589 | 1,710 | 4.5355 | 2.3214 |
| 5,000,000,000 (80% en) | mingram | bnd_w | 33,394 | 28,880 | 298 | 1,428 | 1,711 | 4.1054 | 2.1206 |
| 5,000,000,000 (80% en) | mingram | bnd_wpd | 33,171 | 28,622 | 292 | 1,915 | 1,711 | 4.5511 | 2.2812 |
| 5,000,000,000 (80% en) | mingram | bnd_wpd_caps | 32,691 | 29,062 | 305 | 1,942 | 1,713 | 4.5654 | 2.2847 |

## Paired scheme comparisons

### fineweb_enko_2000m_en50_local_v2, mingram, learned vocabulary 64,000

- `bnd_w` versus plain: English compression -9.563%, Korean -8.688%. Higher is better.
  Allocation versus plain: {'latin': -384, 'hangul': 606, 'other_script': -42, 'nonlexical': -180}.
- `bnd_wp` versus plain: English compression -2.037%, Korean -3.341%. Higher is better.
  Allocation versus plain: {'latin': -454, 'hangul': 463, 'other_script': -43, 'nonlexical': 34}.
- `bnd_wpd` versus plain: English compression -0.212%, Korean -1.438%. Higher is better.
  Allocation versus plain: {'latin': -508, 'hangul': 306, 'other_script': -44, 'nonlexical': 246}.
- `bnd_w_caps` versus plain: English compression -9.162%, Korean -8.638%. Higher is better.
  Allocation versus plain: {'latin': -782, 'hangul': 978, 'other_script': -33, 'nonlexical': -163}.
- `bnd_wp_caps` versus plain: English compression -1.563%, Korean -3.271%. Higher is better.
  Allocation versus plain: {'latin': -839, 'hangul': 831, 'other_script': -34, 'nonlexical': 42}.
- `bnd_wpd_caps` versus plain: English compression +0.284%, Korean -1.348%. Higher is better.
  Allocation versus plain: {'latin': -903, 'hangul': 677, 'other_script': -39, 'nonlexical': 265}.
- Adding case codes to `bnd_w` changes Latin allocation by -398 learned slots and Hangul by +372; Korean compression changes +0.055%.
- Adding case codes to `bnd_wp` changes Latin allocation by -385 learned slots and Hangul by +368; Korean compression changes +0.072%.
- Adding case codes to `bnd_wpd` changes Latin allocation by -395 learned slots and Hangul by +371; Korean compression changes +0.091%.

### fineweb_enko_5000m_en80_local_v2, mingram, learned vocabulary 64,000

- `bnd_w` versus plain: English compression -9.483%, Korean -8.649%. Higher is better.
  Allocation versus plain: {'latin': -671, 'hangul': 852, 'other_script': -20, 'nonlexical': -161}.
- `bnd_wpd` versus plain: English compression +0.345%, Korean -1.731%. Higher is better.
  Allocation versus plain: {'latin': -894, 'hangul': 594, 'other_script': -26, 'nonlexical': 326}.
- `bnd_wpd_caps` versus plain: English compression +0.659%, Korean -1.579%. Higher is better.
  Allocation versus plain: {'latin': -1374, 'hangul': 1034, 'other_script': -13, 'nonlexical': 353}.
- Adding case codes to `bnd_wpd` changes Latin allocation by -480 learned slots and Hangul by +440; Korean compression changes +0.154%.

## MinGram versus BPE allocation


Round-trip failures across completed runs: 0.

This experiment measures tokenizer allocation and compression. It does not test LM quality, establish generality across training samples, or isolate English-only changes: boundary markers affect both scripts. The case-code contrast is more specific to cased text, but Korean documents can contain Latin text.

Full counts, usage frequencies, sample hashes, vocabulary hashes, and timings are in `bilingual_results_learned64000.json`.
