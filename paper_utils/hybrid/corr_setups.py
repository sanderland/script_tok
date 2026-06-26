"""Which train/eval setups (pairings) give correlated compression numbers?

For each pair of setups, and each language, compute R^2 (squared Pearson r) of
the per-method compression values, then average over languages. High mean-R^2 =
the two setups rank/scale the methods the same way => redundant, can be merged
or dropped to reduce the number of panels.
"""
import json
from itertools import combinations

import numpy as np

GRID = "results/hybrid/compression_train_eval_grid.json"
# methods present in EVERY setup (pathpiece/convextok are FineWiki-only)
CORE = ["bpe", "fsp", "bpe_init", "bpe_init_fsp", "mingram"]
LANGS = ["eng", "deu", "fin", "rus", "arb", "kor"]


def main() -> None:
    g = json.load(open(GRID))
    setups = list(g.keys())

    def vec(setup, lang):
        ser = g[setup]["series"].get(lang, {})
        v = [ser.get(m) for m in CORE]
        return None if any(x is None for x in v) else np.array(v, float)

    n = len(setups)
    R2 = np.full((n, n), np.nan)
    for i, j in combinations(range(n), 2):
        r2s = []
        for lang in LANGS:
            a, b = vec(setups[i], lang), vec(setups[j], lang)
            if a is None or b is None or a.std() == 0 or b.std() == 0:
                continue
            r = np.corrcoef(a, b)[0, 1]
            r2s.append(r * r)
        if r2s:
            R2[i, j] = R2[j, i] = float(np.mean(r2s))
    np.fill_diagonal(R2, 1.0)

    short = {s: s.replace("->", "→").replace("finewiki", "fwiki").replace("fineweb", "fweb") for s in setups}
    print("Mean R^2 over languages between setups (per-method compression):\n")
    hdr = "".join(f"{short[s][:13]:>14}" for s in setups)
    print(f"{'':>22}{hdr}")
    for i, s in enumerate(setups):
        row = "".join(f"{R2[i, j]:>14.3f}" if not np.isnan(R2[i, j]) else f"{'-':>14}" for j in range(n))
        print(f"{short[s]:>22}{row}")

    print("\nHighly correlated setup pairs (mean R^2 >= 0.9):")
    for i, j in combinations(range(n), 2):
        if not np.isnan(R2[i, j]) and R2[i, j] >= 0.9:
            print(f"  {short[setups[i]]:<22} ~ {short[setups[j]]:<22} R^2={R2[i, j]:.3f}")


if __name__ == "__main__":
    main()
