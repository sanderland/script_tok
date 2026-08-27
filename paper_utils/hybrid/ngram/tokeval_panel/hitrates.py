"""Fraction of held-out n-grams seen in training, per tokenizer -- the mechanism number."""
import json
import numpy as np
from script_bpe.ngram.counts import build_stream, iter_gram_ids
from script_bpe.ngram.evaluate import VocabGeometry
from script_bpe.ngram.hf_adapter import HFTokenizerAdapter
from script_bpe.ngram.text import iter_documents

SP = "/tmp/claude-0/-home-user-script-tok/7cda6375-6c3b-5955-885d-f6865119d2e6/scratchpad"
CAP = 100_000
eval_docs = [d[:CAP] for d in iter_documents(f"file:{SP}/mix_eval.jsonl") if d]
train_docs = [d[:CAP] for d in iter_documents(f"file:{SP}/mix_train.jsonl") if d]
panel = json.load(open(f"{SP}/panel_tokenizers.json"))

for run in ("full-128k-gpt4o-balanced-bpe", "full-128k-gpt4o-english-bpe"):
    tok = HFTokenizerAdapter(panel[run]["path"])
    enc_e = [np.asarray(tok.encode(d), dtype=np.int32) for d in eval_docs]
    enc_t = [np.asarray(tok.encode(d), dtype=np.int32) for d in train_docs]
    geom = VocabGeometry.of(tok)
    ts, tm, tp = build_stream(enc_t, 3, geom.bos_id, geom.eos_id)
    _, tables = zip(*[(k, t) for k, _, t in iter_gram_ids(ts, tm, tp, 3, geom.radix)])
    tables = tables[-1]
    es, em, ep = build_stream(enc_e, 3, geom.bos_id, geom.eos_id)
    idx = np.flatnonzero(em)
    train_tokens = int(sum(len(a) for a in enc_t))
    line = [run.replace("full-128k-", ""), f"train_tokens={train_tokens/1e6:.1f}M",
            f"per_vocab_entry={train_tokens/128256:.0f}"]
    for k, gid, _ in iter_gram_ids(es, em, ep, 3, geom.radix, tables=list(tables)):
        hit = float((gid[idx] >= 0).mean())
        line.append(f"n={k} seen={hit*100:.1f}%")
    print("  ".join(line), flush=True)

# a Russian word under both, to show the fragmentation
w = " государство"  # 'state', common word
for run in ("full-128k-gpt4o-balanced-bpe", "full-128k-gpt4o-english-bpe"):
    tok = HFTokenizerAdapter(panel[run]["path"])
    ids = tok.encode(w)
    pieces = [tok.decode([i]) for i in ids]
    print(f"{run.replace('full-128k-','')}: {len(ids)} tokens -> {pieces}")
