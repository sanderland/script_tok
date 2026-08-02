"""The sampled-text cache must save the scan without changing the sample.

`create_streaming_sampled_corpus` reservoir-samples a fixed number of characters out of a
much larger source. The corpus it writes is keyed by pretokenizer hash, so a grid that
trains several pretokenizers on one corpus name used to rescan the whole source per
pretokenizer -- on FineWeb sample-10BT that is ~45 GB each, and the scan is single-stream.
The selected text does not depend on the pretokenizer, so it is cached and reused.

The property that matters is that reuse is invisible: a corpus built from the cache must
be identical to one built by scanning.
"""

import hashlib
import tempfile

import pytest

import script_bpe.corpus.registry as R
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.utils import create_logger

DOCS = [{"text": f"doc {i} some  text about {i % 7} things. " * 20} for i in range(400)]


@pytest.fixture
def counting_source(monkeypatch):
    """Stand in for the streamed dataset and count how often it is opened."""
    calls = {"n": 0}

    def fake_load_dataset(name, streaming=True, columns=None, **kw):
        calls["n"] += 1
        return iter(DOCS)

    monkeypatch.setattr(R, "load_dataset", fake_load_dataset)
    return calls


def _build(base, pretokenizer, **kw):
    return R.create_streaming_sampled_corpus(
        dataset_name="fake/ds",
        corpus_name="fineweb_xx_5gb",
        pretokenizer=pretokenizer,
        base_dir=base,
        logger=create_logger("test"),
        source_max_chars=10**9,
        sample_max_chars=20_000,
        text_transform=R.normalize_whitespace,
        block_max_chars=5_000,
        num_workers=2,
        **kw,
    )


def _signature(corpus):
    """Hash the corpus contents, not its metadata counts.

    The counts are too coarse to be a fingerprint: two different script encodings over the
    same ASCII text produce the same number of chunks and atomic tokens and differ only in
    which token ids those are, so a counts-only signature cannot tell a correct build from
    one that reused another pretokenizer's corpus.
    """
    h = hashlib.sha256()
    for chunk, count in sorted(corpus):
        h.update(bytes(chunk))
        h.update(str(count).encode())
    return h.hexdigest()[:16]


def _two_pretokenizers():
    """Two pretokenizers that genuinely encode the same text differently.

    Different script encodings, not merely different flags: `enforce_inherited` gives a
    distinct config hash but chunks plain ASCII identically, so the corpora would coincide
    and the test could not tell a correct build from one that reused the wrong corpus.
    """
    return get_pretokenizer("scriptenc3_cb"), get_pretokenizer("scriptenc_cb")


def test_second_pretokenizer_does_not_rescan(counting_source):
    first, second = _two_pretokenizers()
    with tempfile.TemporaryDirectory() as base:
        _build(base, first)
        assert counting_source["n"] == 1
        _build(base, second)
        assert counting_source["n"] == 1, "the second pretokenizer rescanned the source"


def test_cache_off_rescans(counting_source):
    first, second = _two_pretokenizers()
    with tempfile.TemporaryDirectory() as base:
        _build(base, first, sample_cache=False)
        _build(base, second, sample_cache=False)
        assert counting_source["n"] == 2


def test_cached_corpus_matches_scanned_corpus(counting_source):
    first, second = _two_pretokenizers()
    with tempfile.TemporaryDirectory() as cached, tempfile.TemporaryDirectory() as scanned:
        from_cache = (_build(cached, first), _build(cached, second))
        from_scan = (_build(scanned, first, sample_cache=False),
                     _build(scanned, second, sample_cache=False))
        for a, b in zip(from_cache, from_scan):
            assert _signature(a) == _signature(b)
        assert _signature(from_cache[0]) != _signature(from_cache[1]), (
            "the two pretokenizers should not produce the same corpus"
        )


def test_incomplete_cache_is_not_trusted(tmp_path):
    """A scan killed partway must never be mistaken for a finished sample."""
    cache = tmp_path / "corpus_abc"
    cache.mkdir()
    assert R._read_sample_cache(str(cache)) is None  # no manifest
    R._write_text_batch(str(cache / "block_000000.jsonl"), ["a"])
    (cache / "manifest.json").write_text(
        '{"blocks": ["block_000000.jsonl", "block_000001.jsonl"]}'
    )
    assert R._read_sample_cache(str(cache)) is None  # manifest names a missing block


def test_cache_key_covers_what_selects_the_text():
    fields = dict(
        dataset_name="d", sample_max_chars=1, block_max_chars=2, seed=3,
        text_transform=R.normalize_whitespace, kwargs={"name": "x"},
    )
    base = R._sample_cache_key(**fields)
    assert base == R._sample_cache_key(**fields)
    for field, other in [
        ("dataset_name", "e"), ("sample_max_chars", 2), ("block_max_chars", 3),
        ("seed", 4), ("text_transform", None), ("kwargs", {"name": "y"}),
    ]:
        assert R._sample_cache_key(**{**fields, field: other}) != base, field
