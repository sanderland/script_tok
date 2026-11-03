from collections import Counter

import pytest

from paper_utils.boundary.downstream.train_matched import make_pretokenizer
from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.disk_builder import build_disk_corpus


def test_disk_counts_match_direct_encoding_and_resume(tmp_path):
    pretokenizer = make_pretokenizer("plain")
    batches = [["Hello hello 한국어", "another document"], ["Hello hello 한국어", "끝"]]

    def interrupted():
        yield batches[0]
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        build_disk_corpus("test", interrupted(), pretokenizer, tmp_path, workers=1)
    corpus_dir = tmp_path / "test" / pretokenizer.hash()
    assert not (corpus_dir / "metadata.json").exists()
    assert (corpus_dir / "building.sqlite3").exists()
    corpus = build_disk_corpus("test", iter(batches), pretokenizer, tmp_path, workers=1)
    expected = Counter()
    for texts in batches:
        counts, _ = PretokenizedCorpus.encode_texts(texts, pretokenizer, corpus.DEFAULT_MAX_LENGTH)
        expected.update(counts)
    actual = {chunk.tobytes(): count for chunk, count in corpus}
    assert actual == expected
    assert corpus.metadata["docs"] == 4
    assert corpus.metadata["characters"] == sum(len(text) for texts in batches for text in texts)
    assert not (corpus_dir / "building.sqlite3").exists()
