"""Training data for nanochat.

We use nanochat's *built-in* dataset downloader (ClimbMix shards) rather than
converting script_bpe corpora: nanochat's dataloader reads parquet shards via
`nanochat.dataset.list_parquet_files` and tokenizes text on the fly, and the last
shard is the val split. `download_nanochat_data` just shells that downloader with
the right `NANOCHAT_BASE_DIR`; `run_experiment` calls it for you.

`write_parquet_shards` (corpus -> custom parquet shards) is intentionally left
unimplemented — only needed if we ever train on a script_bpe corpus directly.
"""

import os
import subprocess
import sys
from pathlib import Path


def download_nanochat_data(num_train_shards: int, base_dir: str, nanochat_repo: str | Path) -> Path:
    """Download `num_train_shards` train shards (+ the always-included val shard).

    Returns the data directory. No-op if at least 2 shards already exist.
    """
    base_dir = os.path.abspath(base_dir)
    data_dir = Path(base_dir) / "base_data_climbmix"
    existing = sorted(data_dir.glob("shard_*.parquet")) if data_dir.exists() else []
    if len(existing) >= 2:
        return data_dir
    env = dict(os.environ, NANOCHAT_BASE_DIR=base_dir)
    subprocess.run(
        [sys.executable, "-m", "nanochat.dataset", "-n", str(num_train_shards)],
        cwd=str(nanochat_repo), env=env, check=True,
    )
    return data_dir


def write_parquet_shards(texts, out_dir, shard_size=100_000):
    raise NotImplementedError(
        "unused: we use nanochat's built-in downloader (download_nanochat_data). "
        "Implement only if training on a script_bpe corpus directly."
    )
