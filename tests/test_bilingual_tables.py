"""The bilingual tables must refuse input that would print a partial or unpaired table.

Every cell but the duplicate shares is a difference against the block's `plain` run, so a
missing run, or a scheme trained on other text, would still produce a table that looks
complete. These cases alter the committed extract one field at a time and expect the
check to stop with an error instead.
"""

import copy
import json

import pytest

from paper_utils.boundary.make_bilingual_tables import EXTRACT, TABLES, check


def _main_runs():
    return json.load(open(EXTRACT))["tables"]["main"]


def test_committed_extract_passes():
    # The guard cases below are only meaningful if the unaltered input is accepted.
    assert len(check("main", TABLES["main"], _main_runs())) == len(_main_runs())


def _drop_a_scheme(runs):
    runs.pop(next(i for i, r in enumerate(runs) if r["arm"] == "bnd_wpd"))


def _other_training_text(runs):
    next(r for r in runs if r["arm"] == "bnd_w")["train_sample_sha256"]["ko"] = "0" * 64


def _hidden_category(runs):
    next(r for r in runs if r["arm"] == "plain")["learned_categories"]["mixed"] = 1


def _roundtrip_failure(runs):
    next(r for r in runs if r["arm"] == "bnd_wp")["evaluation"]["en"]["roundtrip_failures"] = 1


def _other_vocabulary(runs):
    next(r for r in runs if r["arm"] == "bnd_w_caps")["total_vocab"] = 32_768


@pytest.mark.parametrize("alter", [_drop_a_scheme, _other_training_text, _hidden_category,
                                   _roundtrip_failure, _other_vocabulary])
def test_check_refuses(alter):
    runs = copy.deepcopy(_main_runs())
    alter(runs)
    with pytest.raises(SystemExit):
        check("main", TABLES["main"], runs)
