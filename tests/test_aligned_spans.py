"""Integration checks with the locally downloaded published MinGram tokenizers."""
from pathlib import Path

import pytest

from paper_utils.boundary.eval_aligned_spans import (
    VARIANTS, aligned_rows, load_tokenizer, non_roundtripping_variants,
)


@pytest.fixture(scope='module', params=['en', 'ko', 'ru'])
def tokenizers(request):
    result = {}
    for variant in VARIANTS:
        folder = Path('models/boundary-marker-lms') / request.param / f'boundary-markers-{request.param}-d12-{variant}-mingram/tokenizer'
        paths = list(folder.glob('*.json.gz'))
        if len(paths) != 1:
            pytest.skip('Published tokenizer not downloaded')
        result[variant] = load_tokenizer(str(paths[0]))
    return result


@pytest.mark.parametrize('text', [
    'The NASA report, 2024. MixedCase and words.',
    'Привет Мир! РУССКИЙ текст, 12345 слов.',
    '한국어 텍스트입니다. 공백 없는단어와 숫자 12345!',
    '  leading  spaces\tand\nnewlines ',
    'x = 1; f(a,b) != 3.14\n',
    'naïve café 😀 中文 русский 한국어',
    'a', ' ', '123', '!!!',
])
def test_common_partition_conserves_all_tokens_and_bytes(tokenizers, text):
    encodings = {v:t.encode(text) for v,t in tokenizers.items()}
    assert all(t.decode(encodings[v]) == text for v,t in tokenizers.items())
    rows = aligned_rows(text, encodings, tokenizers, {}, plain_forms={})
    assert ''.join(text[r['start']:r['end']] for r in rows) == text
    assert sum(r['bytes'] for r in rows) == len(text.encode('utf-8'))
    for v,ids in encodings.items():
        positions = [i for r in rows for i in range(*r['slices'][v])]
        assert positions == list(range(len(ids)))


def test_non_roundtrip_is_identified_before_alignment(tokenizers):
    # Published tokenizers normalize this combining-mark spelling. It cannot
    # enter an exact raw-source comparison, even if all models change it alike.
    text = 'naïve café e\u0301 😀 中文 русский 한국어'
    encodings = {v:t.encode(text) for v,t in tokenizers.items()}
    failed = non_roundtripping_variants(tokenizers, encodings, text)
    assert failed
    assert set(failed) == {v for v,t in tokenizers.items() if t.decode(encodings[v]) != text}
