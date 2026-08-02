import json
import heapq
import os
import random
import re
import tempfile
from typing import Callable

import polars as pl
from datasets import load_dataset

from script_bpe.corpus.base import PretokenizedCorpus
from script_bpe.utils import create_logger
import dotenv

dotenv.load_dotenv()  # support .env with HF_TOKEN

# TokSuite pretraining data subsets (excludes stack_edu which is broken)
TOKSUITE_SUBSETS = [
    "cmn_Hani",  # Chinese
    "fas_Arab",  # Persian/Farsi
    "fw_edu",    # FineWeb educational
    "ita_Latn",  # Italian
    "tur_Latn",  # Turkish
]

FINEWIKI_1GB_MAX_CHARS = 1_000_000_000
FINEWIKI_HYBRID6_CORPORA = [
    "finewiki_en_1gb",
    "finewiki_de_1gb",
    "finewiki_fi_1gb",
    "finewiki_ru_1gb",
    "finewiki_ar_1gb",
    "finewiki_ko_1gb",
]
FINEWEB_HYBRID6_CORPORA = [
    "fineweb_en_5gb",
    "fineweb_de_5gb",
    "fineweb_fi_5gb",
    "fineweb_ru_5gb",
    "fineweb_ar_5gb",
    "fineweb_ko_5gb",
]
FINEWEB_5GB_MAX_CHARS = 5_000_000_000
FINEWEB_SOURCE_MAX_CHARS = 500_000_000_000
FINEWEB_BLOCK_MAX_CHARS = 10_000_000
FINEWEB_SAMPLE_SEED = 42
# Worker processes used to pretokenize a corpus when the caller names no count.
# Pretokenization is pure Python and scales with cores, so this cap is what makes a
# 5 GB build take 8.6 hours on a 288-core node. It stays 16 to leave the behaviour of
# existing callers unchanged; pass `num_workers` to use the machine you actually have.
CORPUS_BUILD_DEFAULT_WORKERS = 16
FINEWEB2_LANGUAGE_CONFIGS = {
    "de": "deu_Latn",
    "nl": "nld_Latn",
    "fi": "fin_Latn",
    "hu": "hun_Latn",
    "kn": "kan_Knda",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "zh": "cmn_Hani",
}
FLORES_PLUS_LANGUAGE_CONFIGS = {
    "eng_latn": "eng_Latn",
    "deu_latn": "deu_Latn",
    "fin_latn": "fin_Latn",
    "rus_cyrl": "rus_Cyrl",
    "arb_arab": "arb_Arab",
    "kor_hang": "kor_Hang",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text)


def _write_text_batch(path: str, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            json.dump(text, f, ensure_ascii=False)
            f.write("\n")


def _read_text_batches(paths: list[str]):
    for path in paths:
        texts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line))
        if texts:
            yield texts


def create_merged_corpus(
    corpus_name: str,
    source_corpus_names: list[str],
    pretokenizer,
    base_dir: str,
    logger,
    num_workers: int | None = None,
) -> PretokenizedCorpus:
    corpus = PretokenizedCorpus(corpus_name, base_dir, pretokenizer, dummy=True)
    if os.path.exists(corpus.metadata_path()):
        raise FileExistsError(
            f"Corpus {corpus_name} already exists at {corpus.metadata_path()}. Use a different name or delete the existing corpus."
        )

    source_corpora = [
        load_corpus_by_name(name, pretokenizer, base_dir=base_dir, num_workers=num_workers)
        for name in source_corpus_names
    ]
    logger.info(f"Merging {len(source_corpora)} corpora into {corpus_name}")

    merged = (
        pl.concat(
            [pl.read_parquet(os.path.join(source.dir_path(), "*.parquet")) for source in source_corpora],
            how="vertical_relaxed",
        )
        .group_by("chunk")
        .agg(pl.col("count").sum().alias("count"))
        .sort("chunk")
        .with_row_index("row_idx")
    )

    metadata: dict[str, int | str] = {
        "version": PretokenizedCorpus.VERSION,
        "max_length": PretokenizedCorpus.DEFAULT_MAX_LENGTH,
        "pretokenizer_hash": pretokenizer.hash(),
        "docs": 0,
        "atomic_tokens": 0,
        "chunks": 0,
        "chunks_skipped": 0,
    }
    for source in source_corpora:
        for key in ("docs", "atomic_tokens", "chunks", "chunks_skipped"):
            metadata[key] += int(source.metadata[key])
    metadata["unique_chunks"] = merged.height

    with open(corpus.metadata_path(), "w") as f:
        json.dump(metadata, f, indent=4)

    for partition in range(PretokenizedCorpus.DEFAULT_PARTITIONS):
        partition_df = merged.filter((pl.col("row_idx") % PretokenizedCorpus.DEFAULT_PARTITIONS) == partition).drop("row_idx")
        partition_df.write_parquet(corpus.partition_path(partition), compression=PretokenizedCorpus.PARQUET_COMPRESSION)

    logger.info(f"Created merged corpus {corpus_name} in {corpus.dir_path()}")
    return PretokenizedCorpus(corpus_name, base_dir, pretokenizer)


def create_streaming_sampled_corpus(
    dataset_name: str,
    corpus_name: str,
    pretokenizer,
    base_dir: str,
    logger,
    source_max_chars: int,
    sample_max_chars: int,
    text_transform: Callable | None = None,
    block_max_chars: int | None = None,
    seed: int | None = None,
    num_workers: int | None = None,
    **kwargs,
) -> PretokenizedCorpus:
    num_cpus = num_workers or min(os.cpu_count() or 4, CORPUS_BUILD_DEFAULT_WORKERS)
    if block_max_chars is None:
        block_max_chars = FINEWEB_BLOCK_MAX_CHARS
    if seed is None:
        seed = FINEWEB_SAMPLE_SEED
    dataset = load_dataset(
        dataset_name,
        streaming=True,
        columns=["text"],
        **kwargs,
    )
    rng = random.Random(seed)
    selected_blocks: list[tuple[float, int, int, str]] = []
    selected_chars = 0
    scanned_chars = 0
    scanned_blocks = 0
    accepted_blocks = 0
    current_texts: list[str] = []
    current_chars = 0

    temp_root = os.path.join(base_dir, "_tmp")
    os.makedirs(temp_root, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{corpus_name.replace(':', '_')}_", dir=temp_root) as temp_dir:
        def finalize_block() -> None:
            nonlocal accepted_blocks
            nonlocal current_chars
            nonlocal current_texts
            nonlocal scanned_blocks
            nonlocal selected_chars

            if not current_texts:
                return

            scanned_blocks += 1
            priority = rng.random()
            if selected_chars < sample_max_chars or not selected_blocks or priority < -selected_blocks[0][0]:
                path = os.path.join(temp_dir, f"block_{scanned_blocks:06d}.jsonl")
                _write_text_batch(path, current_texts)
                heapq.heappush(selected_blocks, (-priority, scanned_blocks, current_chars, path))
                selected_chars += current_chars
                accepted_blocks += 1
                while selected_chars > sample_max_chars:
                    _, _, removed_chars, removed_path = heapq.heappop(selected_blocks)
                    selected_chars -= removed_chars
                    os.remove(removed_path)

            current_texts = []
            current_chars = 0

        for doc in dataset:
            text = doc["text"]
            if text_transform is not None:
                text = text_transform(text)
            if not text:
                continue
            current_texts.append(text)
            text_chars = len(text)
            current_chars += text_chars
            scanned_chars += text_chars
            if current_chars >= block_max_chars:
                finalize_block()
            if scanned_chars >= source_max_chars:
                break
        finalize_block()

        block_paths = [path for _, _, _, path in sorted(selected_blocks, key=lambda block: block[1])]
        logger.info(
            f"Scanned {scanned_chars:,} chars across {scanned_blocks:,} blocks; "
            f"accepted {accepted_blocks:,} candidate blocks, selected {len(block_paths):,} blocks "
            f"for {selected_chars:,} chars"
        )
        corpus = PretokenizedCorpus.from_text_batches(
            name=corpus_name,
            base_path=base_dir,
            pretokenizer=pretokenizer,
            text_batches=_read_text_batches(block_paths),
            num_workers=num_cpus,
        )

    logger.info(f"Created streaming corpus {corpus_name} with pretokenizer {pretokenizer.hash()} in {corpus.dir_path()}")
    return corpus


def create_huggingface_corpus(
    dataset_name: str,
    corpus_name: str,
    pretokenizer,
    base_dir: str,
    logger,
    subsample: int | None = None,
    max_chars: int | None = None,
    text_transform: Callable | None = None,
    num_workers: int | None = None,
    **kwargs,
) -> PretokenizedCorpus:
    num_cpus = num_workers or min(os.cpu_count() or 4, CORPUS_BUILD_DEFAULT_WORKERS)
    dataset = load_dataset(dataset_name, **kwargs)
    if subsample:
        dataset = dataset.select(range(0, len(dataset), subsample))

    texts = [doc["text"] for doc in dataset]

    # Apply text transform if provided (e.g., normalize whitespace)
    if text_transform is not None:
        texts = [text_transform(t) for t in texts]

    if max_chars:  # Shuffle and truncate if requested
        random.seed(42)
        random.shuffle(texts)
        total_chars = 0
        for i in range(len(texts)):
            if total_chars >= max_chars:
                break
            total_chars += len(texts[i])
        texts = texts[:i]
        logger.info(f"Truncated to {i} documents ({total_chars:,} chars)")

    logger.info(f"Loaded dataset {dataset_name} with args {kwargs}, pretokenizing on {num_cpus} CPUs.")
    corpus = PretokenizedCorpus.from_texts(
        name=corpus_name,
        base_path=base_dir,
        pretokenizer=pretokenizer,
        texts=texts,
        num_workers=num_cpus,
    )
    logger.info(f"Created corpus {corpus_name} with pretokenizer {pretokenizer.hash()} in {corpus.dir_path()}")
    return corpus


def load_corpus_by_name(
    corpus_name,
    pretokenizer,
    base_dir: str = PretokenizedCorpus.DEFAULT_BASE_PATH,
    num_workers: int | None = None,
) -> PretokenizedCorpus:  # little hardcoded dataset registry
    logger = create_logger("corpus")
    try:
        corpus = PretokenizedCorpus(
            name=corpus_name,
            base_path=base_dir,
            pretokenizer=pretokenizer,
        )
        return corpus
    except FileNotFoundError as e:
        logger.warning(
            f"Corpus {corpus_name} with pretokenizer {pretokenizer.hash()} not found in cache, creating it: {e}"
        )

    if corpus_name.endswith("300mb"):
        # LEGACY (kept for older caches): undocumented 300MB multilingual sample
        # at sanderland/monolingual-tokenizer-data. The hybrid paper has migrated
        # eval to *_fishfood (Goldfish/LREC 2026) which is the published, citable
        # equivalent. New experiments should use *_fishfood; results match within
        # ~0.02pp per method/lang on this old 300MB set.
        corpus = create_huggingface_corpus(
            "sanderland/monolingual-tokenizer-data",
            corpus_name=corpus_name,
            base_dir=base_dir,
            pretokenizer=pretokenizer,
            num_workers=num_workers,
            logger=logger,
            split="train",
            data_files=[f"{corpus_name.removeprefix('smol_')}.txt"],
            subsample=10 if corpus_name.startswith("smol_") else None,
        )
        return corpus
    elif corpus_name.endswith("_fishfood"):
        # Goldfish "fish-food" multilingual eval (Chang et al., LREC 2026,
        # arXiv 2408.10441; https://huggingface.co/datasets/goldfish-models/fish-food).
        # Files named <lang>_<script>.txt, ~7-15 GB per lang.
        # Our corpus_name "<lang>_<script>_fishfood" -> data_file "<lang>_<script>.txt".
        lang_script = corpus_name.removesuffix("_fishfood")
        return create_huggingface_corpus(
            "goldfish-models/fish-food",
            corpus_name=corpus_name,
            base_dir=base_dir,
            pretokenizer=pretokenizer,
            num_workers=num_workers,
            logger=logger,
            split="train",
            data_files=[f"{lang_script}.txt"],
        )
    elif corpus_name.startswith("flores_plus_"):
        lang_key = corpus_name.removeprefix("flores_plus_")
        return create_huggingface_corpus(
            "openlanguagedata/flores_plus",
            corpus_name=corpus_name,
            base_dir=base_dir,
            pretokenizer=pretokenizer,
            num_workers=num_workers,
            logger=logger,
            name=FLORES_PLUS_LANGUAGE_CONFIGS[lang_key],
            split="dev+devtest",
            text_transform=normalize_whitespace,
        )
    elif "OSCAR" in corpus_name or "CulturaX" in corpus_name:
        return create_huggingface_corpus(
            f"sanderland/{corpus_name}",
            corpus_name=corpus_name,
            base_dir=base_dir,
            logger=logger,
            pretokenizer=pretokenizer,
            split="train",
            num_workers=num_workers,
        )
    elif corpus_name.startswith("finewiki_"):
        # Format: finewiki_{lang}_1gb
        # Always normalize whitespace (collapse multiple spaces to single space)
        lang_code = corpus_name.removeprefix("finewiki_").removesuffix("_1gb")
        max_chars = FINEWIKI_1GB_MAX_CHARS if "_1gb" in corpus_name else None

        return create_huggingface_corpus(
            "HuggingFaceFW/finewiki",
            corpus_name=corpus_name,
            base_dir=base_dir,
            logger=logger,
            pretokenizer=pretokenizer,
            name=lang_code,
            split="train",
            max_chars=max_chars,
            text_transform=normalize_whitespace,
            num_workers=num_workers,
        )
    elif corpus_name.startswith("fineweb_") and (corpus_name.endswith("_5gb") or corpus_name.endswith("_1gb")):
        size_suffix = corpus_name.rsplit("_", 1)[1]  # "5gb" or "1gb" (1gb is for wiki-vs-web size control)
        lang_code = corpus_name.removeprefix("fineweb_").removesuffix(f"_{size_suffix}")
        sample_max = FINEWEB_5GB_MAX_CHARS if size_suffix == "5gb" else FINEWIKI_1GB_MAX_CHARS
        if lang_code == "en":
            dataset_name = "HuggingFaceFW/fineweb"
            dataset_kwargs = {"name": "sample-10BT", "split": "train"}
        else:
            dataset_name = "HuggingFaceFW/fineweb-2"
            dataset_kwargs = {"name": FINEWEB2_LANGUAGE_CONFIGS[lang_code], "split": "train"}
        return create_streaming_sampled_corpus(
            dataset_name=dataset_name,
            corpus_name=corpus_name,
            base_dir=base_dir,
            logger=logger,
            pretokenizer=pretokenizer,
            source_max_chars=FINEWEB_SOURCE_MAX_CHARS,
            sample_max_chars=sample_max,
            text_transform=normalize_whitespace,
            num_workers=num_workers,
            **dataset_kwargs,
        )
    elif corpus_name == "finewiki:hybrid6":
        return create_merged_corpus(
            corpus_name=corpus_name,
            source_corpus_names=FINEWIKI_HYBRID6_CORPORA,
            pretokenizer=pretokenizer,
            base_dir=base_dir,
            logger=logger,
            num_workers=num_workers,
        )
    elif corpus_name == "fineweb:hybrid6":
        return create_merged_corpus(
            corpus_name=corpus_name,
            source_corpus_names=FINEWEB_HYBRID6_CORPORA,
            pretokenizer=pretokenizer,
            base_dir=base_dir,
            logger=logger,
            num_workers=num_workers,
        )
    elif corpus_name == "swift":
        with open("tests/data/taylorswift.txt", "r") as f:
            return PretokenizedCorpus.from_texts(
                corpus_name,
                pretokenizer=pretokenizer,
                texts=[f.read()],
                base_path=base_dir,
            )
    elif corpus_name.startswith("toksuite_"):
        # Format: toksuite_{subset}_1pct or toksuite_all_1pct
        # Dataset: toksuite/toksuite_pretraining_data
        num_cpus = num_workers or min(os.cpu_count() or 4, CORPUS_BUILD_DEFAULT_WORKERS)
        suffix = corpus_name.removeprefix("toksuite_")

        if suffix == "all_1pct":
            # Combined: load 1% from each subset, concatenate
            all_texts = []
            for subset in TOKSUITE_SUBSETS:
                dataset = load_dataset("toksuite/toksuite_pretraining_data", name=subset, split="train")
                dataset = dataset.select(range(0, len(dataset), 100))  # 1% subsample
                all_texts.extend([doc["text"] for doc in dataset])
                logger.info(f"Loaded {len(dataset)} docs from toksuite subset {subset}")
            logger.info(f"Total {len(all_texts)} docs for toksuite_all_1pct, pretokenizing on {num_cpus} CPUs.")
            return PretokenizedCorpus.from_texts(
                name=corpus_name,
                base_path=base_dir,
                pretokenizer=pretokenizer,
                texts=all_texts,
                num_workers=num_cpus,
            )
        else:
            # Per-language: toksuite_{subset}_1pct -> subset name (convert to HF format)
            subset_name = suffix.removesuffix("_1pct")
            # Convert lowercase to HF naming (e.g., cmn_hani -> cmn_Hani)
            subset_lookup = {s.lower(): s for s in TOKSUITE_SUBSETS}
            hf_subset = subset_lookup[subset_name]
            return create_huggingface_corpus(
                "toksuite/toksuite_pretraining_data",
                corpus_name=corpus_name,
                base_dir=base_dir,
                logger=logger,
                pretokenizer=pretokenizer,
                name=hf_subset,
                split="train",
                subsample=100,  # 1% subsample
                num_workers=num_workers,
            )
    else:
        raise ValueError(f"Unknown dataset: {corpus_name}")


MONOLINGUAL_DATASETS = [  # in order of average number of bytes/char in dataset
    "eng_latn_300mb",  # ~1B/char: ASCII only, very efficient
    "deu_latn_300mb",  # ~1B/char: mostly ASCII, occasional umlauts (2B)
    "vie_latn_300mb",  # ~1.3B/char: Latin with many combining diacritics (2–3B per char possible)
    "heb_hebr_300mb",  # ~2B/char: Hebrew uses 2-byte characters in UTF-8
    "arb_arab_300mb",  # ~2/char: Arabic base chars = 2B; some diacritics & shaping
    "rus_cyrl_300mb",  # ~2B/char: Cyrillic mostly 2-byte characters
    "kor_hang_300mb",  # ~3B/char: Hangul syllables are full 3B in UTF-8
    "hin_deva_300mb",  # ~3B/char: Devanagari has many combining marks, 3B typical
    "tha_thai_300mb",  # ~3B/char: Thai script, including tone marks and vowels
    "zho_hans_300mb",  # ~3B/char: Simplified Chinese, each Han character is 3B
    "jpn_jpan_300mb",  # ~3B/char: Mix of of 3B scripts
    "pan_guru_300mb",  # ~3B/char: Gurmukhi script, mostly 3B characters
]
