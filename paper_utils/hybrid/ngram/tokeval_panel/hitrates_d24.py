"""Same seen-rate numbers for the matched d24 regime (34.7K vocab, FineWeb-en)."""
import numpy as np
from script_bpe.ngram.counts import build_stream, iter_gram_ids
from script_bpe.ngram.evaluate import VocabGeometry
from script_bpe.ngram.text import take_split
from script_bpe.tokenizers import load_tokenizer

SP = "/tmp/claude-0/-home-user-script-tok/7cda6375-6c3b-5955-885d-f6865119d2e6/scratchpad"
eval_docs, train_docs = take_split(f"parquet:{SP}/hfdata/sample/10BT/000_00000.parquet",
                                   eval_chars=5_000_000, train_chars=25_000_000)
tok = load_tokenizer("results/hybrid/fineweb_en_5gb/bpe_n32768.model.json.gz")
enc_e = [np.asarray(tok.encode(d), dtype=np.int32) for d in eval_docs]
enc_t = [np.asarray(tok.encode(d), dtype=np.int32) for d in train_docs]
geom = VocabGeometry.of(tok)
ts, tm, tp = build_stream(enc_t, 3, geom.bos_id, geom.eos_id)
tables = None
for _, _, tables in iter_gram_ids(ts, tm, tp, 3, geom.radix):
    pass
es, em, ep = build_stream(enc_e, 3, geom.bos_id, geom.eos_id)
idx = np.flatnonzero(em)
tt = int(sum(len(a) for a in enc_t))
out = [f"d24 bpe arm: train_tokens={tt/1e6:.1f}M per_vocab_entry={tt/34684:.0f}"]
for k, gid, _ in iter_gram_ids(es, em, ep, 3, geom.radix, tables=list(tables)):
    out.append(f"n={k} seen={float((gid[idx] >= 0).mean())*100:.1f}%")
print("  ".join(out))
