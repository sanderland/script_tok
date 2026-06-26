from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.corpus.registry import load_corpus_by_name
from script_bpe.pretokenize import get_pretokenizer
import script_bpe.corpus.registry as registry


class FakeStreamingDataset:
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __iter__(self):
        for text in self.texts:
            yield {"text": text}


class SequenceRandom:
    def __init__(self, _seed: int, values: list[float]):
        self.values = iter(values)

    def random(self) -> float:
        return next(self.values)


def _chunk_counts(corpus: PretokenizedCorpus) -> dict[bytes, int]:
    return {chunk.tobytes(): count for chunk, count in corpus}


def _set_fineweb_budgets(
    monkeypatch,
    *,
    sample_chars: int,
    source_chars: int,
    block_chars: int,
) -> None:
    monkeypatch.setattr(registry, "FINEWEB_5GB_MAX_CHARS", sample_chars)
    monkeypatch.setattr(registry, "FINEWEB_SOURCE_MAX_CHARS", source_chars)
    monkeypatch.setattr(registry, "FINEWEB_BLOCK_MAX_CHARS", block_chars)
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 1)


def test_load_fineweb_en_5gb_uses_sample10bt(tmp_path, monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(dataset_name: str, **kwargs):
        calls.append((dataset_name, kwargs))
        return FakeStreamingDataset(["alpha", "beta", "gamma", "delta"])

    _set_fineweb_budgets(monkeypatch, sample_chars=32, source_chars=64, block_chars=16)
    monkeypatch.setattr(registry, "load_dataset", fake_load_dataset)

    pt = get_pretokenizer("scriptenc_cb")
    corpus = load_corpus_by_name("fineweb_en_5gb", pt, base_dir=str(tmp_path))

    assert calls == [
        (
            "HuggingFaceFW/fineweb",
            {"streaming": True, "columns": ["text"], "name": "sample-10BT", "split": "train"},
        )
    ]
    assert corpus.metadata["docs"] == 4


def test_load_fineweb_ru_5gb_uses_fineweb2_config(tmp_path, monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(dataset_name: str, **kwargs):
        calls.append((dataset_name, kwargs))
        return FakeStreamingDataset(["пример", "текста"])

    _set_fineweb_budgets(monkeypatch, sample_chars=32, source_chars=64, block_chars=16)
    monkeypatch.setattr(registry, "load_dataset", fake_load_dataset)

    pt = get_pretokenizer("scriptenc_cb")
    corpus = load_corpus_by_name("fineweb_ru_5gb", pt, base_dir=str(tmp_path))

    assert calls == [
        (
            "HuggingFaceFW/fineweb-2",
            {"streaming": True, "columns": ["text"], "name": "rus_Cyrl", "split": "train"},
        )
    ]
    assert corpus.metadata["docs"] == 2


def test_load_flores_plus_uses_language_config_and_eval_splits(tmp_path, monkeypatch):
    calls: list[tuple[str, dict]] = []

    def fake_load_dataset(dataset_name: str, **kwargs):
        calls.append((dataset_name, kwargs))
        return FakeStreamingDataset(["eins zwei", "drei vier"])

    monkeypatch.setattr(registry, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(registry.os, "cpu_count", lambda: 1)

    pt = get_pretokenizer("scriptenc_cb")
    corpus = load_corpus_by_name("flores_plus_deu_latn", pt, base_dir=str(tmp_path))

    assert calls == [
        (
            "openlanguagedata/flores_plus",
            {"name": "deu_Latn", "split": "dev+devtest"},
        )
    ]
    assert corpus.metadata["docs"] == 2


def test_load_fineweb_5gb_sampler_replaces_early_blocks(tmp_path, monkeypatch):
    def fake_load_dataset(_dataset_name: str, **_kwargs):
        return FakeStreamingDataset(["aa", "bb", "cc", "dd", "ee", "ff"])

    _set_fineweb_budgets(monkeypatch, sample_chars=8, source_chars=64, block_chars=4)
    monkeypatch.setattr(registry, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(registry.random, "Random", lambda seed: SequenceRandom(seed, [0.9, 0.8, 0.1]))

    pt = get_pretokenizer("scriptenc_cb")
    sampled = load_corpus_by_name("fineweb_en_5gb", pt, base_dir=str(tmp_path))
    expected = PretokenizedCorpus.from_texts(
        name="expected_replaced_blocks",
        texts=["cc", "dd", "ee", "ff"],
        pretokenizer=pt,
        base_path=str(tmp_path),
        num_workers=1,
    )

    assert sampled.metadata["docs"] == expected.metadata["docs"]
    assert sampled.metadata["chunks"] == expected.metadata["chunks"]
    assert _chunk_counts(sampled) == _chunk_counts(expected)


def test_load_fineweb_5gb_respects_source_scan_budget(tmp_path, monkeypatch):
    def fake_load_dataset(_dataset_name: str, **_kwargs):
        return FakeStreamingDataset(["aa", "bb", "cc", "dd", "ee", "ff"])

    _set_fineweb_budgets(monkeypatch, sample_chars=8, source_chars=8, block_chars=4)
    monkeypatch.setattr(registry, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(registry.random, "Random", lambda seed: SequenceRandom(seed, [0.9, 0.8, 0.1]))

    pt = get_pretokenizer("scriptenc_cb")
    sampled = load_corpus_by_name("fineweb_en_5gb", pt, base_dir=str(tmp_path))
    expected = PretokenizedCorpus.from_texts(
        name="expected_source_capped_blocks",
        texts=["aa", "bb", "cc", "dd"],
        pretokenizer=pt,
        base_path=str(tmp_path),
        num_workers=1,
    )

    assert sampled.metadata["docs"] == expected.metadata["docs"]
    assert sampled.metadata["chunks"] == expected.metadata["chunks"]
    assert _chunk_counts(sampled) == _chunk_counts(expected)
