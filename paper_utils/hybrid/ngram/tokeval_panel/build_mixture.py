"""Build eval/train text approximating TokEval's std-1B LM training mixture.

Byte shares from the paper's Tables 7-8 (B.3.1): 36.9% FineWeb-Edu English, 33.4%
FineWeb2 across 30 languages at their exact per-language shares (Russian 18.1% of total
down to Tamil 0.1%), 16.1% FineMath-4plus, 10.0% StarCoderData Python, 3.6% JavaScript.
Eval and train are disjoint slices of each source stream (eval first, then train), so
both splits carry the same mixture. Budgets are in BYTES, matching how the paper states
its shares; documents are never split, so each source overshoots by at most one document.
"""
import json
import sys
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

EVAL_TOTAL, TRAIN_TOTAL = 6_000_000, 30_000_000
FW2 = "datasets/HuggingFaceFW/fineweb-2/data/{}/train/000_00000.parquet"
SHARES = {  # percent of total bytes; source path; text column
    "en": (36.9, "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/000_00000.parquet", "text"),
    "math": (16.1, "datasets/HuggingFaceTB/finemath/finemath-4plus/train-00000-of-00064.parquet", "text"),
    # StarCoderData lists publicly but gates file reads, so code comes from
    # codeparrot/github-code filtered by extension -- same domains (GitHub Python/JS),
    # different cleaning. Stated as an approximation in the writeup.
    "py": (10.0, "datasets/codeparrot/github-code/data/train-00000-of-01126.parquet#EXT:.py", "content"),
    "js": (3.6, "datasets/codeparrot/github-code/data/train-00000-of-01126.parquet#EXT:.js", "content"),
}
FW2_SHARES = {  # Table 8, percent of total bytes
    "rus_Cyrl": 18.1, "spa_Latn": 1.7, "fra_Latn": 1.5, "deu_Latn": 1.4, "cmn_Hani": 1.3,
    "ind_Latn": 0.9, "jpn_Jpan": 0.8, "por_Latn": 0.8, "ukr_Cyrl": 0.8, "ita_Latn": 0.7,
    "arb_Arab": 0.7, "tur_Latn": 0.6, "pol_Latn": 0.4, "ron_Latn": 0.4, "hun_Latn": 0.4,
    "vie_Latn": 0.4, "ell_Grek": 0.4, "nld_Latn": 0.3, "ces_Latn": 0.3, "tha_Thai": 0.3,
    "bul_Cyrl": 0.2, "fin_Latn": 0.1, "slk_Latn": 0.1, "kor_Hang": 0.1, "hrv_Latn": 0.1,
    "cat_Latn": 0.1, "hin_Deva": 0.1, "heb_Hebr": 0.1, "ben_Beng": 0.1, "tam_Taml": 0.1,
}
for lang, pct in FW2_SHARES.items():
    SHARES[lang] = (pct, FW2.format(lang), "text")


def main(out_eval, out_train):
    fs = HfFileSystem()
    t0 = time.time()
    failures = []
    with open(out_eval, "w", encoding="utf-8") as fe, open(out_train, "w", encoding="utf-8") as ft:
        for name, (pct, path, col) in SHARES.items():
          try:
            eb, tb = EVAL_TOTAL * pct / 100, TRAIN_TOTAL * pct / 100
            got_e = got_t = 0
            ext = None
            if "#EXT:" in path:
                path, ext = path.split("#EXT:")
            with fs.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                cols = [col] + (["path"] if ext else [])
                for batch in pf.iter_batches(batch_size=512, columns=cols):
                    paths = batch.column(1).to_pylist() if ext else None
                    for j, text in enumerate(batch.column(0).to_pylist()):
                        if not text or (ext and not paths[j].endswith(ext)):
                            continue
                        nb = len(text.encode("utf-8"))
                        if got_e < eb:
                            fe.write(json.dumps(text, ensure_ascii=False) + "\n")
                            got_e += nb
                        elif got_t < tb:
                            ft.write(json.dumps(text, ensure_ascii=False) + "\n")
                            got_t += nb
                        else:
                            break
                    else:
                        continue
                    break
            print(f"[{time.time()-t0:6.0f}s] {name:10s} eval={got_e/1e6:6.2f}MB train={got_t/1e6:6.2f}MB", flush=True)
          except Exception as e:
            failures.append(name)
            print(f"[{time.time()-t0:6.0f}s] {name:10s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    if failures:
        raise SystemExit(f"sources failed: {failures} -- mixture is incomplete, do not use")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
