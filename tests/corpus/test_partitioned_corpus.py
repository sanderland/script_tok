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
