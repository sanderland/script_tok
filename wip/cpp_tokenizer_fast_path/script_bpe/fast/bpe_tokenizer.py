"""Fast C++ implementation of BPE inference for SCRIPT encoding with character boundaries."""

import unicodedata

import numpy as np
from script_bpe.tokenizers.bpe.tokenizer import BPETokenizer
from script_bpe.pretokenize import Pretokenizer, ScriptPretokenizerConfig
from script_bpe.fast.fast_tokenizer_cpp import FastTokenizer, CharSCRIPTEnc


class FastScriptTokenizer(BPETokenizer):
    """Fast C++ implementation of BPE inference, for SCRIPT encoded with character boundaries.
    
    This tokenizer wraps a standard BPETokenizer and accelerates the encode() method
    using a C++ backend. It only supports ScriptPretokenizer with enforce_char_boundaries=True.
    """
    
    def __init__(self, merge_rules, pretokenizer: Pretokenizer, metadata=None, tokens=None):
        # Only support ScriptPretokenizer for full C++ implementation
        if not isinstance(pretokenizer.config, ScriptPretokenizerConfig):
            raise RuntimeError("FastScriptTokenizer only supports ScriptPretokenizer")
        if not pretokenizer.config.enforce_char_boundaries:
            raise RuntimeError("FastScriptTokenizer requires enforce_char_boundaries=True")
        
        super().__init__(merge_rules, pretokenizer, metadata, tokens)
        self._setup_cpp_backend()

    def _setup_cpp_backend(self):
        """Initialize the C++ tokenizer with character encoding and merge rules."""
        # Find max codepoint to size the vector
        max_cp = max(ord(c) for c in self.pretokenizer.char_encoding)
        cpp_script_encoding = [CharSCRIPTEnc(-1, -1, -1, -1) for _ in range(max_cp + 1)]
        
        for c, char_enc in self.pretokenizer.char_encoding.items():
            token_pair = char_enc.atomic_token_ids
            # Look up if this character pair has been merged into a single token
            token_id = self._merge_rules_dict.get(token_pair, (0, -1))[1]
            cpp_script_encoding[ord(c)] = CharSCRIPTEnc(
                token_id, 
                char_enc.script_id, 
                token_pair[0], 
                token_pair[1]
            )

        self._cpp_fast_tokenizer = FastTokenizer(
            cpp_script_encoding,
            {k: v[1] for k, v in self._merge_rules_dict.items()},
        )
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text using the fast C++ backend.
        
        Args:
            text: The text to encode
            
        Returns:
            A numpy array of token IDs
        """
        normalized = unicodedata.normalize('NFC', text)
        return self._cpp_fast_tokenizer.encode(normalized)
    
    @classmethod
    def from_bpe_tokenizer(cls, tokenizer: BPETokenizer) -> "FastScriptTokenizer":
        """Create a FastScriptTokenizer from an existing BPETokenizer.
        
        Args:
            tokenizer: A BPETokenizer instance with ScriptPretokenizer
            
        Returns:
            A FastScriptTokenizer that uses C++ for inference
        """
        return cls(
            merge_rules=tokenizer.merge_rules,
            pretokenizer=tokenizer.pretokenizer,
            metadata=tokenizer.metadata,
            tokens=tokenizer.tokens,
        )

