#!/usr/bin/env python3
"""Pre-tokenize the ClimbMix corpus for one tokenizer to token-id shards.

The encode is identical across training seeds, so for slow pure-Python tokenizers
we encode the corpus once here and let each seed train GPU-bound by reading these
shards through `pynanochat.pretok_dataloader`.

Usage:
  uv run python paper_utils/hybrid/downstream/pretokenize.py \
    --tokenizer-path <.json.gz> --tokenizer-class <dotted> \
    --base-dir /fsx/sander/nanochat_d24_bpe32k --out-dir /fsx/sander/nc_runs/pretok/<key> \
    --workers 90
"""

import importlib
from pathlib import Path

import cyclopts
import numpy as np
import pyarrow.parquet as pq

app = cyclopts.App()


@app.default
def main(
    tokenizer_path: str,
    tokenizer_class: str,
    base_dir: str,
    out_dir: str,
    workers: int = 90,
    batch: int = 256,
    max_shards: int = -1,
) -> None:
    from pynanochat.tokenizer_adapter import ScriptBPETokenizerAdapter, init_encode_pool

    mod, _, cls = tokenizer_class.rpartition(".")
    adapter = ScriptBPETokenizerAdapter(getattr(importlib.import_module(mod), cls).load(tokenizer_path))
    bos = adapter.get_bos_token_id()
    assert adapter.get_vocab_size() <= 65535, "uint16 overflow"
    init_encode_pool(adapter, workers)

    data_dir = Path(base_dir) / "base_data_climbmix"
    shards = sorted(data_dir.glob("shard_*.parquet"))
    assert shards, f"no shards in {data_dir}"
    if max_shards > 0:
        shards = shards[:max_shards]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[pretok] {len(shards)} shards -> {out}  (bos={bos}, vocab={adapter.get_vocab_size()})", flush=True)

    for shard in shards:
        dest = out / (shard.stem + ".npz")
        if dest.exists():
            print(f"[pretok] skip {shard.name} (exists)", flush=True)
            continue
        pf = pq.ParquetFile(shard)
        tok_chunks, doc_lengths, rg_counts = [], [], []
        for rg in range(pf.num_row_groups):
            texts = pf.read_row_group(rg).column("text").to_pylist()
            n = 0
            for i in range(0, len(texts), batch):
                enc = adapter.encode(texts[i:i + batch], prepend=bos)
                for ids in enc:
                    arr = np.asarray(ids, dtype=np.uint16)
                    tok_chunks.append(arr)
                    doc_lengths.append(len(arr))
                    n += 1
            rg_counts.append(n)
        tokens = np.concatenate(tok_chunks) if tok_chunks else np.zeros(0, np.uint16)
        np.savez(
            dest,
            tokens=tokens,
            doc_lengths=np.asarray(doc_lengths, np.int32),
            rg_doc_counts=np.asarray(rg_counts, np.int32),
        )
        print(
            f"[pretok] {shard.name}  rgs={pf.num_row_groups} docs={len(doc_lengths)} "
            f"toks={tokens.size / 1e6:.1f}M -> {dest.name}",
            flush=True,
        )
    print("[pretok] DONE", flush=True)


if __name__ == "__main__":
    app()
