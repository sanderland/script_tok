"""Re-inject a script_bpe tokenizer inside a nanochat subprocess.

`inject()` only patches the *calling* Python process, but `run_experiment` shells
nanochat's `scripts.base_train` / `scripts.base_eval` out as **subprocesses**, so
the parent's inject never reaches them. The child therefore imports this module
*before* nanochat's argparse runs:

    python -c "import pynanochat.bootstrap; import runpy;
               runpy.run_module('scripts.base_train', run_name='__main__')"

It rebuilds the adapter from two env vars, monkeypatches
`nanochat.tokenizer.get_tokenizer`, and writes the `token_bytes.pt` the bpb metric
needs into nanochat's tokenizer dir.

Env contract (set by runner.run_experiment):
    PYNANOCHAT_TOKENIZER   path to a script_bpe ``.json.gz`` model
    PYNANOCHAT_TOK_CLASS   dotted path to the script_bpe tokenizer class to load with
                          (e.g. ``script_bpe.tokenizers.bpe.BPETokenizer``)
    NANOCHAT_BASE_DIR     (read by nanochat) where tokenizer/ + checkpoints live
"""

import importlib
import os

from .tokenizer import inject, write_token_bytes
from .tokenizer_adapter import ScriptBPETokenizerAdapter, init_encode_pool


def load_adapter():
    path = os.environ["PYNANOCHAT_TOKENIZER"]
    dotted = os.environ["PYNANOCHAT_TOK_CLASS"]
    module_name, _, cls_name = dotted.rpartition(".")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return ScriptBPETokenizerAdapter(cls.load(path))


def _override_seed():
    """nanochat hardcodes torch.manual_seed(42) in compute_init (no --seed flag).
    Patch torch's seed setters to use PYNANOCHAT_SEED instead, so we can run replicates."""
    seed = os.environ.get("PYNANOCHAT_SEED")
    if seed is None:
        return
    import torch
    s = int(seed)
    _m, _c = torch.manual_seed, torch.cuda.manual_seed
    torch.manual_seed = lambda *_a, **_k: _m(s)
    torch.cuda.manual_seed = lambda *_a, **_k: _c(s)


def _enable_core_context_crop():
    """Make CORE's own over-length truncation actually run.

    `core_eval.evaluate_example` crops prompts longer than the trained context, but only
    `if hasattr(model, 'max_seq_len')`, and nanochat's GPT defines no such attribute, so
    the branch is dead. Over-long prompts are then forwarded at rotary positions the model
    never saw in training. That is not neutral across arms: a tokenizer emitting more
    tokens per byte crosses the limit more often. Measured over 22 CORE tasks, prompts past
    2048 tokens are 3.4% for plain and 5.9% for bnd_w, concentrated in squad and boolq.
    Exposing the attribute evaluates every arm inside its trained context.

    This changes absolute CORE values, so they are not comparable to numbers produced
    without it.
    """
    import nanochat.gpt as gpt_module

    if not hasattr(gpt_module.GPT, "max_seq_len"):
        gpt_module.GPT.max_seq_len = property(lambda self: self.config.sequence_len)


def _shuffle_data_order():
    """Make the seed move the data order, not only the weight initialization.

    `_document_batches` contains no RNG: parquet files in sorted order, row groups in
    rank-strided index order. Every seed of an arm therefore reads the identical document
    sequence and ends at the identical position, so the spread across seeds measures
    initialization variance alone and understates run-to-run variance.

    The permutation is derived from the seed only, never from the arm, so seed s draws the
    same order in every arm. That keeps the arms paired, which is what makes a paired
    comparison across arms valid, while letting the seed spread include order variance.
    """
    seed = os.environ.get("PYNANOCHAT_SEED")
    if seed is None:
        return
    import random

    import nanochat.dataloader as dl

    original = dl._document_batches

    def shuffled(split, resume_state_dict, tokenizer_batch_size):
        # Only the training stream is permuted. The validation split must stay fixed, or
        # each seed would be scored on a different sample of text.
        if split != "train":
            yield from original(split, resume_state_dict, tokenizer_batch_size)
            return
        rng = random.Random(int(seed))
        pf_list = dl.list_parquet_files()[:-1]
        order = list(range(len(pf_list)))
        rng.shuffle(order)
        real_list = dl.list_parquet_files

        def permuted_list(*a, **k):
            files = real_list(*a, **k)
            train, val = files[:-1], files[-1:]
            return [train[i] for i in order] + val

        dl.list_parquet_files = permuted_list
        try:
            yield from original(split, resume_state_dict, tokenizer_batch_size)
        finally:
            dl.list_parquet_files = real_list

    dl._document_batches = shuffled


def _maybe_patch_pretok_loader():
    """If PYNANOCHAT_PRETOK_DIR is set, swap base_train's TRAINING loader for the pre-tokenized
    one (eval's bpb loader stays on-the-fly = canonical). Must run before base_train imports it."""
    pretok = os.environ.get("PYNANOCHAT_PRETOK_DIR")
    if not pretok:
        return
    import nanochat.dataloader as _dl
    from .pretok_dataloader import pretok_data_loader_with_state_bos_bestfit
    _dl.tokenizing_distributed_data_loader_with_state_bos_bestfit = pretok_data_loader_with_state_bos_bestfit
    print(f"[pynanochat] PRETOK training loader active: {pretok}", flush=True)


def main():
    _override_seed()
    _shuffle_data_order()
    _enable_core_context_crop()
    adapter = load_adapter()
    inject(adapter)
    _maybe_patch_pretok_loader()
    # Set up the parallel-encode pool HERE, before nanochat touches CUDA: this module is
    # imported at the very top of the train/eval `python -c`, so forking now is safe.
    workers = int(os.environ.get("PYNANOCHAT_ENCODE_WORKERS", "1"))
    if workers > 1:
        init_encode_pool(adapter, workers)
    # nanochat reads token_bytes.pt from <base_dir>/tokenizer (get_token_bytes()).
    from nanochat.common import get_base_dir
    write_token_bytes(adapter, os.path.join(get_base_dir(), "tokenizer"))


# Run on import so the `python -c "import pynanochat.bootstrap; ..."` pattern works.
main()
