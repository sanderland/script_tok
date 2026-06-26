import pytest

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.registry import (
    FINEWEB_HYBRID6_CORPORA,
    FINEWIKI_HYBRID6_CORPORA,
    load_corpus_by_name,
)
from script_bpe.pretokenize import get_pretokenizer


def _chunk_counts(corpus: PretokenizedCorpus) -> dict[bytes, int]:
    return {chunk.tobytes(): count for chunk, count in corpus}


@pytest.mark.parametrize(
    ("alias_name", "source_corpora"),
    [
        ("finewiki:hybrid6", FINEWIKI_HYBRID6_CORPORA),
        ("fineweb:hybrid6", FINEWEB_HYBRID6_CORPORA),
    ],
)
def test_load_hybrid6_from_cached_corpora(tmp_path, alias_name, source_corpora):
    pt = get_pretokenizer("scriptenc_cb")
    source_texts = {
        corpus_name: [f"{corpus_name} alpha beta", "shared token"]
        for corpus_name in source_corpora
    }

    combined_texts: list[str] = []
    for corpus_name in source_corpora:
        texts = source_texts[corpus_name]
        PretokenizedCorpus.from_texts(
            name=corpus_name,
            texts=texts,
            pretokenizer=pt,
            base_path=str(tmp_path),
        )
        combined_texts.extend(texts)

    merged = load_corpus_by_name(alias_name, pt, base_dir=str(tmp_path))
    expected = PretokenizedCorpus.from_texts(
        name=f"expected_{alias_name.replace(':', '_')}",
        texts=combined_texts,
        pretokenizer=pt,
        base_path=str(tmp_path),
    )

    assert merged.metadata["docs"] == expected.metadata["docs"]
    assert merged.metadata["chunks"] == expected.metadata["chunks"]
    assert merged.metadata["atomic_tokens"] == expected.metadata["atomic_tokens"]
    assert _chunk_counts(merged) == _chunk_counts(expected)
