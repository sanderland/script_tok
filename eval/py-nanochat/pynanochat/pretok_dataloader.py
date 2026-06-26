"""Pre-tokenized drop-in for nanochat's bos_bestfit dataloader.

Reads token-id shards written by paper_utils/hybrid/downstream/pretokenize.py instead of
tokenizing parquet text on the fly. The doc source mirrors nanochat's _document_batches
EXACTLY (same train/val split, same DDP row-group striding, same doc order), and the
best-fit packing body is copied verbatim — so training is bit-identical to on-the-fly,
just without the (slow, pure-Python) encode in the hot loop.

Activated by bootstrap when PYNANOCHAT_PRETOK_DIR is set: it monkeypatches
nanochat.dataloader.tokenizing_distributed_data_loader_bos_bestfit to this.
"""
import os
from pathlib import Path

import numpy as np
import torch

from nanochat.common import get_dist_info


def _pretok_files(split):
    pretok_dir = Path(os.environ["PYNANOCHAT_PRETOK_DIR"])
    files = sorted(pretok_dir.glob("shard_*.npz"))
    assert files, f"no pretok shards in {pretok_dir}"
    return files[:-1] if split == "train" else files[-1:]


def _load_shard_rows(npz_path):
    """Return list (over row-groups) of list (over docs) of token-id lists, in original order."""
    d = np.load(npz_path)
    tokens, doc_lengths, rg_counts = d["tokens"], d["doc_lengths"], d["rg_doc_counts"]
    # split flat tokens into per-doc lists
    offs = np.zeros(len(doc_lengths) + 1, dtype=np.int64)
    np.cumsum(doc_lengths, out=offs[1:])
    docs = [tokens[offs[i]:offs[i + 1]].tolist() for i in range(len(doc_lengths))]
    # regroup docs into row-groups
    rows, p = [], 0
    for c in rg_counts:
        rows.append(docs[p:p + c])
        p += c
    return rows


def _pretok_document_batches(split, resume_state_dict, tokenizer_batch_size):
    """Mirror nanochat._document_batches: infinite, DDP row-group striding, but pre-encoded.
    Yields in tokenizer_batch_size chunks (CRITICAL: same chunking => same bestfit buffer state)."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    paths = _pretok_files(split)
    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    epoch = resume_epoch
    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(paths):
            rows = _load_shard_rows(paths[pq_idx])  # list[rg] -> list[doc] -> token list
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size + 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= len(rows):
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < len(rows):
                docs = rows[rg_idx]
                for i in range(0, len(docs), tokenizer_batch_size):
                    yield docs[i:i + tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def pretok_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None, buffer_size=1000,
):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _pretok_document_batches(split, resume_state_dict, tokenizer_batch_size)
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        rg_docs, (pq_idx, rg_idx, epoch) = next(batches)
        for tokens in rg_docs:          # already bos-prepended, already token ids
            doc_buffer.append(tokens)

    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    dl = len(doc)
                    if dl <= remaining and dl > best_len:
                        best_idx, best_len = i, dl
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    dl = len(doc)
                    row_buffer[row_idx, pos:pos + dl] = torch.tensor(doc, dtype=torch.long)
                    pos += dl
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def pretok_data_loader_bos_bestfit(*args, **kwargs):
    for inputs, targets, _ in pretok_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
