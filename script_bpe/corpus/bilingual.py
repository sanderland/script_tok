"""Bounded bilingual samples from existing FineWeb quick text caches, no downloads.

Names encode total characters in millions and the English percentage, for example
fineweb_enko_20m_en50_local_v1. Source order is the cached quick sample's order,
not a new uniform sample. Documents stay separate; only the last is truncated.
"""

import hashlib
import json
from pathlib import Path
import re
import contextlib

from script_bpe.corpus.base import PretokenizedCorpus

NAME = re.compile(r"fineweb_enko_([1-9][0-9]*)m_en([1-9][0-9]?)_local_v[12]")


def mixture_budgets(name):
    match = NAME.fullmatch(name)
    if not match:
        raise ValueError(f"Invalid bilingual corpus name: {name}")
    millions, english = map(int, match.groups())
    total = millions * 1_000_000
    return {"en": total * english // 100, "ko": total * (100 - english) // 100}


def source_cache(base_dir, lang):
    candidates = sorted((Path(base_dir) / "_sampled_text").glob(f"fineweb_{lang}_5gb_quick_*/manifest.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one complete local {lang} quick sample, found {len(candidates)}")
    manifest_path = candidates[0]
    manifest = json.loads(manifest_path.read_text())
    paths = [manifest_path.parent / name for name in manifest["blocks"]]
    if manifest.get("method") != "quick" or not paths or not all(p.is_file() for p in paths):
        raise ValueError(f"Incomplete or non-quick sample: {manifest_path}")
    return manifest_path, paths


def bounded_documents(paths, budget):
    remaining = budget
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line)
                if not isinstance(text, str):
                    raise ValueError(f"Expected a JSON string in {path}")
                text = text[:remaining]
                if text:
                    yield text
                    remaining -= len(text)
                if remaining == 0:
                    return
    if remaining:
        raise ValueError(f"Source short by {remaining} characters")


def prepare_mixture(name, base_dir):
    """Cache document-preserving JSONL with an exact character budget per language."""
    budgets = mixture_budgets(name)
    target = Path(base_dir) / "_bilingual_text" / name
    manifest_path = target / "manifest.json"
    reference_only = name.endswith("_v2")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["budgets"] != budgets:
            raise ValueError("Bilingual sample budget mismatch")
        for lang, info in manifest["sources"].items():
            if reference_only:
                source, paths = source_cache(base_dir, lang)
                if hashlib.sha256(source.read_bytes()).hexdigest() != info["manifest_sha256"]:
                    raise ValueError(f"Source manifest changed: {source}")
                if [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths] != [tuple(row) for row in info["block_stats"]]:
                    raise ValueError(f"Source blocks changed: {source}")
                continue
            path = target / f"{lang}.jsonl"
            if not path.is_file() or path.stat().st_size != info["jsonl_bytes"]:
                raise ValueError(f"Incomplete bilingual sample: {path}")
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != info["sample_sha256"]:
                    raise ValueError(f"Bilingual sample checksum mismatch: {path}")
        return target, manifest
    sources = {lang: source_cache(base_dir, lang) for lang in budgets}
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "budgets": budgets, "sampling": "prefix of cached quick sample; final document truncated", "sources": {}}
    for lang, budget in budgets.items():
        source, paths = sources[lang]
        digest = hashlib.sha256()
        count = 0
        partial = target / f"{lang}.jsonl.partial"
        output_context = contextlib.nullcontext(None) if reference_only else partial.open("wb")
        byte_count = 0
        with output_context as handle:
            for text in bounded_documents(paths, budget):
                encoded = (json.dumps(text, ensure_ascii=False) + "\n").encode("utf-8")
                if handle is not None:
                    handle.write(encoded)
                digest.update(encoded)
                byte_count += len(encoded)
                count += 1
        output = target / f"{lang}.jsonl"
        if not reference_only:
            partial.replace(output)
        manifest["sources"][lang] = {"manifest": str(source), "manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "sample_sha256": digest.hexdigest(), "documents": count, "characters": budget, "jsonl_bytes": byte_count}
        if reference_only:
            manifest["sources"][lang]["block_stats"] = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
        print(f"Prepared {lang}: {budget:,} characters in {count:,} documents", flush=True)
    partial_manifest = target / "manifest.json.partial"
    partial_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    partial_manifest.replace(manifest_path)
    return target, manifest


def text_batches(target, batch_chars=1_000_000):
    manifest = json.loads((target / "manifest.json").read_text())
    for lang in ("en", "ko"):
        batch, size = [], 0
        if manifest["name"].endswith("_v2"):
            source = Path(manifest["sources"][lang]["manifest"])
            paths = [source.parent / row[0] for row in manifest["sources"][lang]["block_stats"]]
        else:
            paths = [target / f"{lang}.jsonl"]
        for text in bounded_documents(paths, manifest["budgets"][lang]):
            batch.append(text)
            size += len(text)
            if size >= batch_chars:
                yield batch
                batch, size = [], 0
        if batch:
            yield batch


def create_bilingual_corpus(name, pretokenizer, base_dir, num_workers=None):
    target, _ = prepare_mixture(name, base_dir)
    if name.endswith("_v2"):
        from script_bpe.corpus.disk_builder import build_disk_corpus

        return build_disk_corpus(name, text_batches(target), pretokenizer, base_dir, num_workers or 2)
    return PretokenizedCorpus.from_text_batches(
        name=name, text_batches=text_batches(target), pretokenizer=pretokenizer,
        base_path=base_dir, num_workers=num_workers or 2,
    )
