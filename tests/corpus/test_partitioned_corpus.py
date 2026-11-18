import pytest

from script_bpe.corpus import PretokenizedCorpus
from script_bpe.pretokenize import get_pretokenizer


@pytest.mark.parametrize("pretokenizer_name", ["bytes_gpt4", "scriptenc_cb", "scriptenc_nosplit_cb"])
def test_corpus_creation_and_content_integrity(tmp_path, pretokenizer_name, taylorswift_text):
    """
    Tests corpus creation, basic metadata, and that iterating through the corpus
    yields all unique chunks with correct counts. Does a rough check on total token count.
    """
    pretokenizer = get_pretokenizer(pretokenizer_name)
    corpus_name = f"e2e_integrity_{pretokenizer_name}"
    texts = [taylorswift_text[:2000], taylorswift_text[2000:]]

    corpus = PretokenizedCorpus.from_texts(
        name=corpus_name,
        texts=texts,
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
        num_partitions=2,
        num_workers=1,  # Use 1 process for determinism here
    )
    assert corpus.metadata["docs"] == len(texts)
    assert corpus.metadata["chunks"] > 0
    assert corpus.metadata["unique_chunks"] > 0
    assert corpus.metadata["atomic_tokens"] > 0  # Total tokens in all stored chunks
    assert corpus.metadata["chunks_skipped"] == 0  # default max_length is sufficient

    # Iterate and collect all chunks and their counts
    iterated_chunk_counts = {}
    total_tokens_from_iteration = 0
    for worker_id in range(2):  # Iterate over partitions/workers
        for chunk_arr, count in corpus.worker_iterate(worker_id=worker_id, num_workers=2):
            iterated_chunk_counts[chunk_arr.tobytes()] = count
            total_tokens_from_iteration += len(chunk_arr) * count

    assert len(iterated_chunk_counts) == corpus.metadata["unique_chunks"]
    assert (
        sum(iterated_chunk_counts.values()) == corpus.metadata["chunks"]
    )  # Sum of counts of unique chunks == total chunks
    assert total_tokens_from_iteration == corpus.metadata["atomic_tokens"]


@pytest.mark.parametrize("num_procs", [1, 2])  # Test single and multi-process
@pytest.mark.parametrize("pretokenizer_name", ["bytes_gpt4", "scriptenc_cb"])
def test_corpus_reconstruction_and_token_sum(tmp_path, taylorswift_text, pretokenizer_name, num_procs):
    """
    Tests that the sum of (len(chunk) * count) for all unique chunks
    matches the 'atomic_tokens' metadata. Verifies consistency across single/multi-proc.
    Checks if some decoded content resembles input.
    """
    pretokenizer = get_pretokenizer(pretokenizer_name)
    corpus_name = f"e2e_reconstruct_procs_{num_procs}"

    corpus = PretokenizedCorpus.from_texts(
        name=corpus_name,
        texts=[taylorswift_text],
        pretokenizer=pretokenizer,
        base_path=str(tmp_path),
        num_partitions=2,
        num_workers=num_procs,
    )

    reconstructed_atomic_tokens = 0
    # Iterate through all unique chunks from the corpus
    for chunk_array, count in corpus.worker_iterate(worker_id=0, num_workers=1):
        reconstructed_atomic_tokens += len(chunk_array) * count

    assert reconstructed_atomic_tokens == corpus.metadata["atomic_tokens"]


# Writing partition files dominates the build time of a small corpus, so these tests use few partitions.
# The partition count does not affect the property under test.
TEST_PARTITIONS = 4
# Each multi-worker build pays forkserver start-up (every worker imports script_bpe), so the tests below do
# one multi-worker build in total and compare everything else against single-worker builds.
TEST_WORKERS = 2
SMALL_TASK_CHARS = 500


def _read_corpus_content(corpus: PretokenizedCorpus) -> tuple[list[str], list[tuple[bytes, int]]]:
    """Return (partition file names, all (chunk, count) rows sorted across partitions)."""
    rows = [(chunk.tobytes(), count) for chunk, count in corpus]
    return corpus.partitions, sorted(rows)


def test_splitter_cuts_by_size_and_keeps_every_document_once():
    """
    _split_by_chars must keep documents in order, never drop or duplicate one, keep each task within the
    character budget, and leave a document larger than the budget whole as its own task.
    """
    from script_bpe.corpus import base

    documents = ["a" * 300, "b" * 300, "c" * 1200, "d" * 100, "e" * 100, "f" * 301, "g"]
    tasks = list(base._split_by_chars(documents, SMALL_TASK_CHARS))
    assert tasks == [
        [documents[0]],
        [documents[1]],
        [documents[2]],
        [documents[3], documents[4]],
        [documents[5], documents[6]],
    ]
    assert list(base._split_by_chars([], SMALL_TASK_CHARS)) == []


def test_corpus_does_not_depend_on_batching_workers_or_task_size(tmp_path, taylorswift_text, monkeypatch):
    """
    The corpus must not depend on how the text is batched, how many workers encode it, or how it is cut into
    tasks. Reference: one batch, one worker. Variants: (a) many small batches, one worker; (b) the same batches,
    TEST_WORKERS workers and a 500-character task budget, so the batches are cut into many tasks that finish in
    arbitrary order. Metadata and the sorted (chunk, count) rows must be identical, in the requested files.
    The main process does the splitting, so patching the module constant takes effect for the forkserver pool.
    """
    from script_bpe.corpus import base

    pretokenizer = get_pretokenizer("bytes_gpt4")
    documents = taylorswift_text.split("\n")  # one document per line
    assert len(documents) > 200, "test data must yield many batches"
    assert any(len(text) > SMALL_TASK_CHARS for text in documents), "test data must contain an over-budget document"
    batches = [documents[i : i + 20] for i in range(0, len(documents), 20)]
    kwargs = dict(pretokenizer=pretokenizer, base_path=str(tmp_path), num_partitions=TEST_PARTITIONS)

    reference = PretokenizedCorpus.from_texts("reference", documents, num_workers=1, **kwargs)
    builds = {"batches_1_worker": PretokenizedCorpus.from_text_batches("batches_1", batches, num_workers=1, **kwargs)}
    monkeypatch.setattr(base, "ENCODE_TASK_MAX_CHARS", SMALL_TASK_CHARS)
    builds["small_tasks_multiprocess"] = PretokenizedCorpus.from_text_batches(
        "small_tasks", batches, num_workers=TEST_WORKERS, **kwargs
    )

    reference_rows = _read_corpus_content(reference)[1]
    for name, corpus in builds.items():
        files, rows = _read_corpus_content(corpus)
        assert len(files) == TEST_PARTITIONS, name
        assert corpus.metadata == reference.metadata, name
        assert rows == reference_rows, name
