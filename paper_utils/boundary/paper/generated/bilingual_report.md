# English/Korean shared-vocabulary ablation

CPU BPE/MinGram ablation using prefixes of the local FineWeb quick caches. The 20M-character runs are exploratory pilots. All schemes share the same text within a mixture. Budgets and vocabulary sizes are recorded below. Evaluation uses fixed prefixes of the cached English and Korean Goldfish slices. Compression counts characters after the tokenizer's normalization. Mixture ratios count characters, not documents or UTF-8 bytes.

Allocation counts below cover learned pieces only. Latin/Hangul are script labels, not exclusive language ownership. Duplicate surplus counts use n−1 per equivalent group, retaining boundary markers. Space/case columns overlap and must not be added. These counts do not measure all possible positional redundancy. Space surplus includes whitespace-only pairs, such as newline versus space+newline; per-script counts are in the JSON. Individual SCRIPT atomics may be undecodable fragments, so usage in that category does not imply failed text round trips.

| Train chars | Trainer | Scheme | Latin | Hangul | Other scripts | Nonlexical | Atomics | English chars/token | Korean chars/token |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2,000,000,000 (50% en) | bpe | plain | 12,443 | 19,673 | 116 | 743 | 1,710 | 4.0692 | 2.1929 |
| 2,000,000,000 (50% en) | bpe | bnd_w | 12,451 | 19,811 | 91 | 621 | 1,711 | 3.6943 | 1.9962 |
| 2,000,000,000 (50% en) | bpe | bnd_w_caps | 12,334 | 19,919 | 94 | 625 | 1,713 | 3.6959 | 1.9978 |
| 2,000,000,000 (50% en) | bpe | bnd_wp | 12,394 | 19,728 | 90 | 762 | 1,711 | 3.9801 | 2.1005 |
| 2,000,000,000 (50% en) | bpe | bnd_wp_caps | 12,289 | 19,823 | 92 | 768 | 1,713 | 3.9812 | 2.1017 |
| 2,000,000,000 (50% en) | bpe | bnd_wpd | 12,305 | 19,583 | 90 | 996 | 1,711 | 4.0442 | 2.1367 |
| 2,000,000,000 (50% en) | bpe | bnd_wpd_caps | 12,200 | 19,678 | 92 | 1,002 | 1,713 | 4.0462 | 2.1380 |
| 2,000,000,000 (50% en) | mingram | plain | 11,857 | 20,235 | 117 | 766 | 1,710 | 4.1055 | 2.2116 |
| 2,000,000,000 (50% en) | mingram | bnd_w | 11,738 | 20,487 | 99 | 650 | 1,711 | 3.7019 | 2.0126 |
| 2,000,000,000 (50% en) | mingram | bnd_w_caps | 11,513 | 20,694 | 108 | 657 | 1,713 | 3.7311 | 2.0152 |
| 2,000,000,000 (50% en) | mingram | bnd_wp | 11,691 | 20,377 | 99 | 807 | 1,711 | 3.9880 | 2.1180 |
| 2,000,000,000 (50% en) | mingram | bnd_wp_caps | 11,472 | 20,579 | 104 | 817 | 1,713 | 4.0214 | 2.1211 |
| 2,000,000,000 (50% en) | mingram | bnd_wpd | 11,619 | 20,228 | 99 | 1,028 | 1,711 | 4.0545 | 2.1549 |
| 2,000,000,000 (50% en) | mingram | bnd_wpd_caps | 11,413 | 20,422 | 104 | 1,033 | 1,713 | 4.0877 | 2.1580 |
| 20,000,000 (50% en) | bpe | plain | 12,572 | 19,542 | 66 | 795 | 1,710 | 4.0334 | 2.1667 |
| 20,000,000 (50% en) | bpe | bnd_wpd | 12,436 | 19,440 | 61 | 1,037 | 1,711 | 4.0016 | 2.1115 |
| 20,000,000 (50% en) | bpe | bnd_wpd_caps | 12,349 | 19,522 | 61 | 1,040 | 1,713 | 4.0062 | 2.1121 |
| 20,000,000 (90% en) | bpe | plain | 23,504 | 8,590 | 46 | 835 | 1,710 | 4.3276 | 1.9200 |
| 20,000,000 (90% en) | bpe | bnd_wpd | 23,053 | 8,779 | 44 | 1,098 | 1,711 | 4.3387 | 1.8566 |
| 20,000,000 (90% en) | bpe | bnd_wpd_caps | 22,977 | 8,832 | 50 | 1,113 | 1,713 | 4.3451 | 1.8580 |

## Paired scheme comparisons

### fineweb_enko_2000m_en50_local_v2, bpe, vocabulary 34,685

- `bnd_w` versus plain: English compression -9.211%, Korean -8.970%. Higher is better.
  Allocation versus plain: {'latin': 8, 'hangul': 138, 'other_script': -25, 'nonlexical': -122}.
- `bnd_w_caps` versus plain: English compression -9.173%, Korean -8.897%. Higher is better.
  Allocation versus plain: {'latin': -109, 'hangul': 246, 'other_script': -22, 'nonlexical': -118}.
- `bnd_wp` versus plain: English compression -2.188%, Korean -4.216%. Higher is better.
  Allocation versus plain: {'latin': -49, 'hangul': 55, 'other_script': -26, 'nonlexical': 19}.
- `bnd_wp_caps` versus plain: English compression -2.161%, Korean -4.161%. Higher is better.
  Allocation versus plain: {'latin': -154, 'hangul': 150, 'other_script': -24, 'nonlexical': 25}.
- `bnd_wpd` versus plain: English compression -0.614%, Korean -2.566%. Higher is better.
  Allocation versus plain: {'latin': -138, 'hangul': -90, 'other_script': -26, 'nonlexical': 253}.
- `bnd_wpd_caps` versus plain: English compression -0.564%, Korean -2.504%. Higher is better.
  Allocation versus plain: {'latin': -243, 'hangul': 5, 'other_script': -24, 'nonlexical': 259}.
- Adding case codes to `bnd_w` changes Latin allocation by -117 learned slots and Hangul by +108; Korean compression changes +0.080%.
- Adding case codes to `bnd_wp` changes Latin allocation by -105 learned slots and Hangul by +95; Korean compression changes +0.057%.
- Adding case codes to `bnd_wpd` changes Latin allocation by -105 learned slots and Hangul by +95; Korean compression changes +0.064%.

### fineweb_enko_2000m_en50_local_v2, mingram, vocabulary 34,685

- `bnd_w` versus plain: English compression -9.831%, Korean -8.997%. Higher is better.
  Allocation versus plain: {'latin': -119, 'hangul': 252, 'other_script': -18, 'nonlexical': -116}.
- `bnd_w_caps` versus plain: English compression -9.121%, Korean -8.882%. Higher is better.
  Allocation versus plain: {'latin': -344, 'hangul': 459, 'other_script': -9, 'nonlexical': -109}.
- `bnd_wp` versus plain: English compression -2.863%, Korean -4.232%. Higher is better.
  Allocation versus plain: {'latin': -166, 'hangul': 142, 'other_script': -18, 'nonlexical': 41}.
- `bnd_wp_caps` versus plain: English compression -2.050%, Korean -4.092%. Higher is better.
  Allocation versus plain: {'latin': -385, 'hangul': 344, 'other_script': -13, 'nonlexical': 51}.
- `bnd_wpd` versus plain: English compression -1.244%, Korean -2.567%. Higher is better.
  Allocation versus plain: {'latin': -238, 'hangul': -7, 'other_script': -18, 'nonlexical': 262}.
- `bnd_wpd_caps` versus plain: English compression -0.433%, Korean -2.426%. Higher is better.
  Allocation versus plain: {'latin': -444, 'hangul': 187, 'other_script': -13, 'nonlexical': 267}.
- Adding case codes to `bnd_w` changes Latin allocation by -225 learned slots and Hangul by +207; Korean compression changes +0.127%.
- Adding case codes to `bnd_wp` changes Latin allocation by -219 learned slots and Hangul by +202; Korean compression changes +0.147%.
- Adding case codes to `bnd_wpd` changes Latin allocation by -206 learned slots and Hangul by +194; Korean compression changes +0.144%.

### fineweb_enko_20m_en50_local_v1, bpe, vocabulary 34,685

- `bnd_wpd` versus plain: English compression -0.788%, Korean -2.545%. Higher is better.
  Allocation versus plain: {'latin': -136, 'hangul': -102, 'other_script': -5, 'nonlexical': 242}.
- `bnd_wpd_caps` versus plain: English compression -0.674%, Korean -2.517%. Higher is better.
  Allocation versus plain: {'latin': -223, 'hangul': -20, 'other_script': -5, 'nonlexical': 245}.
- Adding case codes to `bnd_wpd` changes Latin allocation by -87 learned slots and Hangul by +82; Korean compression changes +0.029%.

### fineweb_enko_20m_en90_local_v1, bpe, vocabulary 34,685

- `bnd_wpd` versus plain: English compression +0.257%, Korean -3.300%. Higher is better.
  Allocation versus plain: {'latin': -451, 'hangul': 189, 'other_script': -2, 'nonlexical': 263}.
- `bnd_wpd_caps` versus plain: English compression +0.406%, Korean -3.228%. Higher is better.
  Allocation versus plain: {'latin': -527, 'hangul': 242, 'other_script': 4, 'nonlexical': 278}.
- Adding case codes to `bnd_wpd` changes Latin allocation by -76 learned slots and Hangul by +53; Korean compression changes +0.075%.

## MinGram versus BPE allocation

- fineweb_enko_2000m_en50_local_v2, 34,685 tokens, `plain`: MinGram minus BPE learned slots {'latin': -586, 'hangul': 562, 'other_script': 1, 'nonlexical': 23}.
- fineweb_enko_2000m_en50_local_v2, 34,685 tokens, `bnd_w`: MinGram minus BPE learned slots {'latin': -713, 'hangul': 676, 'other_script': 8, 'nonlexical': 29}.
- fineweb_enko_2000m_en50_local_v2, 34,685 tokens, `bnd_w_caps`: MinGram minus BPE learned slots {'latin': -821, 'hangul': 775, 'other_script': 14, 'nonlexical': 32}.
- fineweb_enko_2000m_en50_local_v2, 34,685 tokens, `bnd_wp`: MinGram minus BPE learned slots {'latin': -703, 'hangul': 649, 'other_script': 9, 'nonlexical': 45}.
- fineweb_enko_2000m_en50_local_v2, 34,685 tokens, `bnd_wp_caps`: MinGram minus BPE learned slots {'latin': -817, 'hangul': 756, 'other_script': 12, 'nonlexical': 49}.
- fineweb_enko_2000m_en50_local_v2, 34,685 tokens, `bnd_wpd`: MinGram minus BPE learned slots {'latin': -686, 'hangul': 645, 'other_script': 9, 'nonlexical': 32}.
- fineweb_enko_2000m_en50_local_v2, 34,685 tokens, `bnd_wpd_caps`: MinGram minus BPE learned slots {'latin': -787, 'hangul': 744, 'other_script': 12, 'nonlexical': 31}.

Round-trip failures across completed runs: 0.

This experiment measures tokenizer allocation and compression. It does not test LM quality, establish generality across training samples, or isolate English-only changes: boundary markers affect both scripts. The case-code contrast is more specific to cased text, but Korean documents can contain Latin text.

Full counts, usage frequencies, sample hashes, vocabulary hashes, and timings are in `bilingual_results.json`.
