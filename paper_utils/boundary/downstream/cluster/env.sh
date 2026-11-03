#!/usr/bin/env bash
# Shared environment for the boundary-marker downstream runs on CSCS Clariden.
#
# Source before any step, interactively or inside a job:
#     source paper_utils/boundary/downstream/cluster/env.sh
#
# Why these are not the defaults: home on Clariden has a 50 GB quota that is close to full,
# and `uv sync --extra downstream` alone needs about 10 GB (torch plus the CUDA wheels), so
# every cache the pipeline writes is redirected to scratch.
#
# Paths are written out rather than derived from $SCRATCH: $SCRATCH points at iopsstor in
# this account's shell, and inheriting it once split the layout across two filesystems by
# accident. capstor is the faster of the two for this work; measured 2026-07-31 with a
# 200 MiB O_DIRECT dd, capstor wrote at 781 MB/s and iopsstor at 84.2 MB/s, and imports of
# the environment stalled for minutes on iopsstor.
#
# capstor deletes files not accessed for 14 days, and it did exactly that to this project
# between 2026-08-04 and 2026-09-17: the environment, the data shards, every checkpoint and
# every log were gone, leaving the directory tree standing. Re-touch anything worth keeping
# inside each 14-day window:
#
#     find "${DATA_ROOT}" -exec touch -a -m {} +

DATA_ROOT="/capstor/scratch/cscs/${USER}/marker_downstream"

# uv: environment and wheel cache on the same filesystem, so uv can hardlink between them.
export UV_CACHE_DIR="${DATA_ROOT}/uv_cache"
export UV_PROJECT_ENVIRONMENT="${DATA_ROOT}/venv"
# Compute nodes must not re-resolve dependencies; the environment is built on the login node.
export UV_NO_SYNC=1

export HF_HOME="${DATA_ROOT}/hf_cache"

# Respects a caller's value. The August version assigned this unconditionally, so a sweep
# pointed at another shard directory silently trained on this one instead. That matters more
# now than it did then: each language has its own base directory holding its own text, and
# the two are not interchangeable.
export NANOCHAT_BASE="${NANOCHAT_BASE:-${DATA_ROOT}/nanochat_base}"

# Triton's default is ~/.triton/cache, which every node of a sweep would share over the home
# filesystem. Node-local is both faster and quieter.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}_triton_cache}"
