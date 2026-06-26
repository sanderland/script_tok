"""Entry point for multi-GPU (torchrun) nanochat runs.

torchrun launches `python _torchrun_entry.py <flags>` once per rank. We import
`pynanochat.bootstrap` FIRST (injects the script_bpe tokenizer + sets up the pre-CUDA
encode pool, per rank), then run the target nanochat module as __main__ so its argparse
sees <flags> via sys.argv. The target (scripts.base_train / scripts.base_eval) is passed
in `PYNANOCHAT_TARGET`. The single-GPU path uses `python -c` instead (see runner.py).
"""

import os
import runpy

import pynanochat.bootstrap  # noqa: F401  -- runs on import: inject + token_bytes + encode pool

runpy.run_module(os.environ["PYNANOCHAT_TARGET"], run_name="__main__")
