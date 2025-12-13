from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.pretokenize import get_pretokenizer


def test_load_swift(tmp_path):
    pt = get_pretokenizer("scriptenc_cb")
    corpus = load_corpus_by_name("swift", pt, base_dir=str(tmp_path))
    # basic metadata and iteration smoke
    assert corpus.metadata["docs"] >= 1
    it = list(corpus.worker_iterate(worker_id=0, num_workers=1))
    assert len(it) > 0
