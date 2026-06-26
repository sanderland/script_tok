# Fast C++ backend for tokenization algorithms
# This module provides optional C++ acceleration for BPE inference and Unigram algorithms

# BPE fast backend - check if C++ module is available
BPE_FAST_AVAILABLE = False
try:
    from script_bpe.fast.fast_tokenizer_cpp import FastTokenizer, CharSCRIPTEnc
    BPE_FAST_AVAILABLE = True
except ImportError:
    FastTokenizer = None
    CharSCRIPTEnc = None

# Unigram fast backend  
UNIGRAM_FAST_AVAILABLE = False
FastTrie = None
FastLattice = None

try:
    from script_bpe.fast.fast_unigram_cpp import FastTrie, FastLattice
    UNIGRAM_FAST_AVAILABLE = True
except ImportError:
    pass

# Overall fast availability (at least one backend available)
FAST_AVAILABLE = BPE_FAST_AVAILABLE or UNIGRAM_FAST_AVAILABLE

# Lazy import for FastScriptTokenizer to avoid circular dependency
# BPETokenizer imports fast_tokenizer_cpp, which triggers this module,
# which would try to import bpe_tokenizer which imports BPETokenizer
_FastScriptTokenizer = None

def get_fast_script_tokenizer():
    """Get FastScriptTokenizer class, lazy-loaded to avoid circular import."""
    global _FastScriptTokenizer
    if _FastScriptTokenizer is None and BPE_FAST_AVAILABLE:
        from script_bpe.fast.bpe_tokenizer import FastScriptTokenizer
        _FastScriptTokenizer = FastScriptTokenizer
    return _FastScriptTokenizer

# For backwards compatibility, expose FastScriptTokenizer as property-like access
# Users should import via: from script_bpe.fast.bpe_tokenizer import FastScriptTokenizer
# Or use: script_bpe.fast.get_fast_script_tokenizer()
FastScriptTokenizer = None  # Will be None until lazy loaded
