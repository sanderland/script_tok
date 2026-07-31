#!/usr/bin/env bash
# Shared environment for the boundary-marker downstream runs on CSCS Clariden.
#
# Source this before any step of the README's flow, interactively or inside a job:
#     source marker_experiments/downstream/cluster/env.sh
#
# Why none of these are the README's defaults: home on Clariden
# (/vast/users/cscs/$USER) has a 50 GB quota that is already near-full, and
# `uv sync --extra downstream` alone needs ~10 GB (torch plus the CUDA wheels).
# Every cache the pipeline writes is therefore redirected to scratch.
#
# Everything lives on capstor scratch, written out rather than derived from $SCRATCH:
# $SCRATCH is already set to the iopsstor path in this account's shell, and inheriting it
# put the environment on iopsstor by accident.
#
# The venv was on iopsstor at first, on the theory that an IO-optimized mount suits the
# many small reads of a Python import. Measured on 2026-07-31 with 200 MiB O_DIRECT dd:
# capstor 781 MB/s write and 1.1 GB/s read, iopsstor 84.2 MB/s write. A test process sat
# in Lustre cl_sync_io_wait for minutes partway through importing wcwidth. A job whose
# workers each import this environment cannot absorb that, so the venv moved to capstor.
#
# capstor purges files not accessed for 14 days. Re-touch these paths if the experiment
# goes dormant that long.

DATA_ROOT="/capstor/scratch/cscs/${USER}/marker_downstream"

# uv: environment and wheel cache on the same filesystem, so uv can hardlink between them.
export UV_CACHE_DIR="${DATA_ROOT}/uv_cache"
export UV_PROJECT_ENVIRONMENT="${DATA_ROOT}/venv"
# Compute nodes must not re-resolve dependencies; the environment is built on the login node.
export UV_NO_SYNC=1

# Dataset caches: FineWiki/FineWeb via huggingface, ClimbMix shards via nanochat.
export HF_HOME="${DATA_ROOT}/hf_cache"
export NANOCHAT_BASE="${DATA_ROOT}/nanochat_base"

# Results land under the repo's `results` symlink, which already points at capstor scratch.
export OUT="${OUT:-results/marker_downstream}"

mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${NANOCHAT_BASE}"
