"""Shared paths for the boundary-marker paper's scripts.

Every script here writes into one artifact directory, `paper/generated/`, and reads its
inputs from the same place. Computing that path with `os.path.dirname` chains in each
script made the depth a thing to get right per file and silently wrong after a move, so
it is stated once.

Mirrors `paper_utils/hybrid/utils.py`, except that this paper's artifacts are small enough
to track: the JSON and TSV caches under `paper/generated/` are what let the tables
regenerate on a machine with no GPU and no trained tokenizers.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# The one artifact directory: caches in, LaTeX and figures out.
GENERATED = os.path.join(HERE, "paper", "generated")
MANIFEST_PARTS = os.path.join(GENERATED, "manifest_parts")
MANIFEST = os.path.join(GENERATED, "manifest.json")
EVAL_GOLDFISH = os.path.join(GENERATED, "eval_goldfish.json")

# Trained tokenizers. Gitignored: the matched grid is about 100 MB, and it rebuilds from
# the corpus. The caches above are what the paper actually needs.
TOKENIZERS = os.path.join(HERE, "downstream", "tokenizers")
EVAL_TEXTS = os.path.join(HERE, "eval_texts")


def rel(path: str) -> str:
    """A repo-relative path, for provenance headers in generated files."""
    return os.path.relpath(path, REPO_ROOT)
