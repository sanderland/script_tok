"""The quick sampler must read only what it keeps, and never collide with the full one."""

import tempfile

import pytest

import script_bpe.corpus.registry as R
from script_bpe.pretokenize import get_pretokenizer
from script_bpe.utils import create_logger

DOCS = [{"text": f"doc {i} some  text about {i % 7} things. " * 20} for i in range(4000)]


class _CountingIterable:
    """Stands in for a streaming IterableDataset and counts documents actually consumed."""

    def __init__(self, docs, seen):
        self.docs, self.seen = docs, seen

    def shuffle(self, seed=None, buffer_size=None):
        # Order changes, length does not; enough to check the caller shuffles and that the
        # sample is not simply the head of the stream.
        rotated = self.docs[buffer_size or 0:] + self.docs[: buffer_size or 0]
        return _CountingIterable(rotated, self.seen)

    def __iter__(self):
        for doc in self.docs:
            self.seen[0] += 1
            yield doc


@pytest.fixture
def counting_source(monkeypatch):
    seen = [0]
    monkeypatch.setattr(R, "load_dataset", lambda *a, **k: _CountingIterable(DOCS, seen))
    return seen


def _quick(base, pt, seen=None, **kw):
    return R.create_streaming_quick_corpus(
        dataset_name="fake/ds", corpus_name="fineweb_xx_5gb_quick", pretokenizer=pt,
        base_dir=base, logger=create_logger("test"), sample_max_chars=20_000,
        text_transform=R.normalize_whitespace, block_max_chars=5_000, num_workers=2,
        shuffle_buffer=64, **kw,
    )


def test_reads_only_what_it_keeps(counting_source):
    """The point of the quick path: stop at the sample size instead of draining the source."""
    pt = get_pretokenizer("scriptenc3_cb")
    with tempfile.TemporaryDirectory() as base:
        _quick(base, pt)
    assert counting_source[0] < len(DOCS) / 4, (
        f"read {counting_source[0]} of {len(DOCS)} documents for a sample worth far fewer"
    )


def test_second_pretokenizer_reuses_the_sample(counting_source):
    pt1 = get_pretokenizer("scriptenc3_cb")
    pt2 = get_pretokenizer("scriptenc_cb")
    with tempfile.TemporaryDirectory() as base:
        _quick(base, pt1)
        after_first = counting_source[0]
        _quick(base, pt2)
    assert counting_source[0] == after_first, "the second pretokenizer re-read the source"


def test_quick_and_reservoir_do_not_share_a_cache():
    """Different samples, so a shared cache key would hand one method the other's text."""
    common = dict(dataset_name="d", sample_max_chars=1, block_max_chars=2, seed=3,
                  text_transform=None, kwargs={})
    reservoir = R._sample_cache_key(**common)
    assert reservoir == R._sample_cache_key(**common, method="reservoir"), (
        "adding the method argument must not change keys already published"
    )
    assert R._sample_cache_key(**common, method="quick10000") != reservoir
    assert R._sample_cache_key(**common, method="quick64") != R._sample_cache_key(
        **common, method="quick10000"
    )


def test_corpus_name_routes_to_the_right_sampler(monkeypatch):
    """fineweb_<lang>_5gb_quick must reach the quick sampler, plain _5gb the reservoir."""
    called = []
    monkeypatch.setattr(R, "create_streaming_quick_corpus",
                        lambda **kw: called.append(("quick", kw["corpus_name"])))
    monkeypatch.setattr(R, "create_streaming_sampled_corpus",
                        lambda **kw: called.append(("reservoir", kw["corpus_name"])))
    pt = get_pretokenizer("scriptenc3_cb")
    with tempfile.TemporaryDirectory() as base:
        for name in ("fineweb_en_5gb", "fineweb_en_5gb_quick"):
            try:
                R.load_corpus_by_name(name, pt, base_dir=base)
            except FileNotFoundError:
                pass
    assert called == [("reservoir", "fineweb_en_5gb"), ("quick", "fineweb_en_5gb_quick")]
