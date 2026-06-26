"""Tests for FastScriptTokenizer C++ implementation."""

import pytest
import tempfile
import shutil

# Check if fast C++ backend is available
try:
    from script_bpe.fast import BPE_FAST_AVAILABLE
    if BPE_FAST_AVAILABLE:
        from script_bpe.fast.bpe_tokenizer import FastScriptTokenizer
    else:
        FastScriptTokenizer = None
except ImportError:
    BPE_FAST_AVAILABLE = False
    FastScriptTokenizer = None

from script_bpe.pretokenize import get_pretokenizer
from script_bpe.tokenizers.bpe.trainer import BPETrainer, BPETrainerConfig
from script_bpe.corpus import PretokenizedCorpus

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(
    not BPE_FAST_AVAILABLE,
    reason="C++ fast tokenizer extension not built"
)


@pytest.fixture(scope="module")
def temp_dir_module():
    """Create a temporary directory for the module for efficiency."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def train_bpe_simple(pretokenizer, corpus, additional_vocab_size, num_workers=1, verbose=False):
    """Simple BPE training helper for tests."""
    config = BPETrainerConfig(additional_vocab_size=additional_vocab_size, num_workers=num_workers)
    trainer = BPETrainer(pretokenizer, corpus, config)
    return trainer.train()


@pytest.fixture(scope="module")
def comprehensive_tokenizer(temp_dir_module):
    """
    Creates a single, comprehensive tokenizer for all tests.
    The corpus is specifically designed to create a rich set of conditions:
    - 'a' is pre-merged into a single char token (T_a).
    - 'z' remains an unmerged base pair.
    - A merge rule for (T_a, T_a) is created to test determinism.
    - Multilingual characters are included for broad compatibility.
    """
    pretokenizer = get_pretokenizer("scriptenc_cb")

    # This corpus creates all the test conditions we need in one tokenizer.
    corpus_text = ("a" * 500) + "z" + "世"

    corpus = PretokenizedCorpus.from_texts(
        name="test_corpus_comprehensive_final",
        texts=[corpus_text], pretokenizer=pretokenizer, base_path=temp_dir_module
    )
    # Learn enough rules to merge 'a' into a single token, and then merge that token with itself.
    tokenizer = train_bpe_simple(
        pretokenizer, corpus, additional_vocab_size=3, num_workers=1, verbose=False
    )
    return tokenizer, pretokenizer


@pytest.fixture
def fast_tokenizer(comprehensive_tokenizer):
    """Instantiates the FastScriptTokenizer from the comprehensive fixture."""
    tokenizer, pretokenizer = comprehensive_tokenizer
    return FastScriptTokenizer.from_bpe_tokenizer(tokenizer), tokenizer


# Comprehensive list of test cases for the main validation function
TEST_CASES = [
    # Basic cases
    pytest.param("", "Empty string", id="empty_string"),
    pytest.param(" ", "Single space", id="single_space"),
    
    # Determinism test (the critical bug fix)
    pytest.param("aaaaa", "Sequence requiring deterministic merge order", id="determinism"),
    
    # Merged & Unmerged character tests
    pytest.param("a", "Single pre-merged char", id="single_merged"),
    pytest.param("z", "Single unmerged char", id="single_unmerged"),
    pytest.param("az", "Merged followed by unmerged", id="merged_unmerged"),
    pytest.param("za", "Unmerged followed by merged", id="unmerged_merged"),
    pytest.param("aza", "Unmerged between two merged", id="unmerged_between_merged"),
    pytest.param("zaz", "Merged between two unmerged", id="merged_between_unmerged"),
    
    # Multilingual and other characters
    pytest.param("世", "Single Han character", id="han_char"),
    pytest.param("a世z", "Mixed Latin, Han, and unmerged Latin", id="mixed_scripts"),
    pytest.param("123", "Sequence of numbers", id="numbers"),
    pytest.param("a z a z a", "Complex sequence with spaces", id="complex_string"),
]


@pytest.mark.parametrize("test_text, description", TEST_CASES)
def test_fast_tokenizer_is_identical(fast_tokenizer, test_text, description):
    """
    Tests that the C++ tokenizer produces IDENTICAL results to the Python reference
    across a wide range of critical edge cases using a single, comprehensive tokenizer.
    """
    cpp_tok, python_tok = fast_tokenizer

    python_tokens = python_tok.encode(test_text)
    cpp_tokens = cpp_tok.encode(test_text)

    assert list(cpp_tokens) == list(python_tokens), (
        f"Mismatch for case: '{description}'\n"
        f"String: {repr(test_text)}\n"
        f"Python (correct): {list(python_tokens)}\n"
        f"C++    (buggy):   {list(cpp_tokens)}"
    )

    # Also verify that decoding produces identical, correct results
    python_decoded = python_tok.decode(python_tokens)
    cpp_decoded = cpp_tok.decode(cpp_tokens)
    assert cpp_decoded == python_decoded
    assert cpp_decoded == python_tok.pretokenizer.normalize(test_text)


def test_fast_tokenizer_error_handling():
    """Test error handling for incorrect pre-tokenizer types."""
    # FastScriptTokenizer only supports ScriptPretokenizer
    pretokenizer = get_pretokenizer("bytes_gpt4")
    with pytest.raises(RuntimeError, match="FastScriptTokenizer only supports ScriptPretokenizer"):
        FastScriptTokenizer(merge_rules=[], pretokenizer=pretokenizer, metadata={})
    
    # It must also have enforce_char_boundaries=True
    pretokenizer_no_cb = get_pretokenizer("scriptenc")
    with pytest.raises(RuntimeError, match="enforce_char_boundaries"):
        FastScriptTokenizer(merge_rules=[], pretokenizer=pretokenizer_no_cb, metadata={})


@pytest.fixture(scope="module")
def determinism_tokenizer(temp_dir_module):
    """
    Creates a tokenizer with the simplest possible case to test for non-determinism:
    a single merge rule 'a,a -> T_aa' that can be applied in multiple places.
    """
    pretokenizer = get_pretokenizer("scriptenc_cb")
    corpus_text = "a" * 500  # A long string of 'a's to ensure 'a,a' is the top merge.

    corpus = PretokenizedCorpus.from_texts(
        name="test_corpus_determinism",
        texts=[corpus_text], pretokenizer=pretokenizer, base_path=temp_dir_module
    )
    # We only need to learn one merge rule.
    tokenizer = train_bpe_simple(
        pretokenizer, corpus, additional_vocab_size=2, num_workers=1, verbose=False
    )
    return tokenizer, pretokenizer


def test_deterministic_merge_order_failure(determinism_tokenizer):
    """
    This test is designed to FAIL with the non-deterministic C++ implementation and PASS
    with the corrected one. It validates that when multiple occurrences of the SAME
    merge rule are possible, the leftmost one is always chosen.

    - The Buggy C++ `priority_queue` had no tie-breaker for merges of equal priority,
      leading to arbitrary (non-deterministic) merge order.
    - The Corrected C++ `priority_queue` uses `std::tie(priority, from_a)` to mimic
      Python's deterministic behavior, always choosing the merge with the lowest index first.
    """
    python_tok, pretokenizer = determinism_tokenizer
    cpp_tok = FastScriptTokenizer.from_bpe_tokenizer(python_tok)

    # This input has many possible aa merges
    test_string = "a"*100
    
    # Python's output is the deterministic "ground truth." It will always merge left-to-right.
    # Expected logic: aaaaa -> (T_aa)aaa -> (T_aa)a(T_aa)
    python_tokens = python_tok.encode(test_string)

    # The C++ output will be different if its arbitrary choice of merge order differs from Python's.
    cpp_tokens = cpp_tok.encode(test_string)

    # This assertion will fail if the C++ implementation is non-deterministic.
    assert list(cpp_tokens) == list(python_tokens), (
        f"\n\n*** FAILURE DETECTED: Non-Deterministic Merge Order ***\n"
        f"String: {repr(test_string)}\n"
        f"Python (Correct & Deterministic): {list(python_tokens)}\n"
        f"C++    (Incorrect & Arbitrary):  {list(cpp_tokens)}\n"
        f"Reason: For multiple possible merges of the same rule, the C++ implementation did not\n"
        f"        deterministically choose the leftmost one, leading to a different tokenization."
    )
