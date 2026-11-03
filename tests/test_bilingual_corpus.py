import json

import pytest

from script_bpe.corpus.bilingual import bounded_documents, mixture_budgets, prepare_mixture, text_batches


def test_exact_budgets_and_invalid_names():
    assert mixture_budgets("fineweb_enko_20m_en50_local_v1") == {"en": 10_000_000, "ko": 10_000_000}
    assert mixture_budgets("fineweb_enko_20m_en90_local_v1") == {"en": 18_000_000, "ko": 2_000_000}
    with pytest.raises(ValueError):
        mixture_budgets("fineweb_enko_20m_en100_local_v1")


def test_bounded_documents_preserve_unicode_and_boundaries(tmp_path):
    path = tmp_path / "block.jsonl"
    path.write_text('\n'.join(json.dumps(t) for t in ["hello", "한국어입니다", "unused"]))
    assert list(bounded_documents([path], 8)) == ["hello", "한국어"]
    with pytest.raises(ValueError, match="short"):
        list(bounded_documents([path], 100))


def test_prepare_reuses_identical_sample_and_batches(tmp_path):
    for lang in ("en", "ko"):
        source = tmp_path / "_sampled_text" / f"fineweb_{lang}_5gb_quick_test"
        source.mkdir(parents=True)
        (source / "block.jsonl").write_text(json.dumps("a" * 600_000) + "\n")
        (source / "manifest.json").write_text(json.dumps({"method": "quick", "blocks": ["block.jsonl"]}))
    name = "fineweb_enko_1m_en50_local_v1"
    target, manifest = prepare_mixture(name, tmp_path)
    assert sum(len(t) for batch in text_batches(target) for t in batch) == 1_000_000
    assert prepare_mixture(name, tmp_path)[1] == manifest
    path = target / "ko.jsonl"
    path.write_text(path.read_text().replace("a", "b"))
    with pytest.raises(ValueError, match="checksum"):
        prepare_mixture(name, tmp_path)
    (target / "ko.jsonl").write_text("")
    with pytest.raises(ValueError, match="Incomplete"):
        prepare_mixture(name, tmp_path)


def test_reference_sample_matches_copied_sample_without_copying(tmp_path):
    for lang in ("en", "ko"):
        source = tmp_path / "_sampled_text" / f"fineweb_{lang}_5gb_quick_test"
        source.mkdir(parents=True)
        (source / "block.jsonl").write_text(json.dumps("한a" * 300_000) + "\n")
        (source / "manifest.json").write_text(json.dumps({"method": "quick", "blocks": ["block.jsonl"]}))
    copied, copy_manifest = prepare_mixture("fineweb_enko_1m_en50_local_v1", tmp_path)
    target, manifest = prepare_mixture("fineweb_enko_1m_en50_local_v2", tmp_path)
    assert not list(target.glob("*.jsonl"))
    assert list(text_batches(target)) == list(text_batches(copied))
    for lang in ("en", "ko"):
        assert manifest["sources"][lang]["sample_sha256"] == copy_manifest["sources"][lang]["sample_sha256"]
    assert prepare_mixture("fineweb_enko_1m_en50_local_v2", tmp_path)[1]["budgets"] == manifest["budgets"]
