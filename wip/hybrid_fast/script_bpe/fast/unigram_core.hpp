#pragma once

/**
 * C++ acceleration for Unigram tokenization algorithms.
 * 
 * This module provides fast implementations of:
 * - Trie: Prefix tree for token lookup
 * - Lattice: Dynamic programming for Viterbi and forward-backward
 * - Suffix array construction and LCP-based pattern extraction for initialization
 * 
 * These are the hot paths in Unigram training and inference.
 */

#include <vector>
#include <unordered_map>
#include <string>
#include <tuple>
#include <cstdint>
#include <cmath>
#include <limits>

#ifndef NO_PYTHON_BINDINGS
    #include <pybind11/pybind11.h>
    #include <pybind11/stl.h>
    #include <pybind11/numpy.h>
    namespace py = pybind11;
#endif

namespace script_bpe {
namespace unigram {

using token_id_t = int32_t;
using atomic_token_t = int32_t;
using log_prob_t = double;

constexpr log_prob_t NEG_INF = -std::numeric_limits<double>::infinity();

/**
 * Token in the Unigram vocabulary.
 */
struct UnigramToken {
    token_id_t id;
    std::vector<atomic_token_t> atomic_tokens;
    log_prob_t log_prob;
    bool required;
};

/**
 * Trie node for fast prefix matching.
 */
struct TrieNode {
    std::unordered_map<atomic_token_t, size_t> children;  // token -> child node index
    token_id_t token_id = -1;  // -1 if not a terminal node
    log_prob_t log_prob = NEG_INF;
};

/**
 * Trie for fast prefix lookup of tokens.
 */
class FastTrie {
public:
    FastTrie() {
        nodes_.emplace_back();  // root node
    }
    
    void insert(token_id_t token_id, const std::vector<atomic_token_t>& atomic_tokens, log_prob_t log_prob);
    
    /**
     * Find all tokens that are prefixes of the given sequence starting at offset.
     * Returns vector of (token_id, length, log_prob) tuples.
     */
    std::vector<std::tuple<token_id_t, size_t, log_prob_t>> find_prefixes(
        const std::vector<atomic_token_t>& seq, size_t offset) const;

private:
    std::vector<TrieNode> nodes_;
};

/**
 * Lattice for dynamic programming on token segmentations.
 */
class FastLattice {
public:
    FastLattice(const std::vector<atomic_token_t>& atomic_tokens,
                const std::vector<std::vector<std::tuple<token_id_t, size_t, log_prob_t>>>& tokens_from_pos,
                log_prob_t token_bias = 0.0);
    
    /**
     * Viterbi algorithm: find the best segmentation.
     * Returns (best_path_token_ids, best_path_score).
     * If allow_single_token is false, disallows paths that use a single token spanning the entire sequence.
     */
    std::pair<std::vector<token_id_t>, log_prob_t> viterbi(bool allow_single_token = true) const;
    
    /**
     * Forward-backward algorithm for computing marginal probabilities.
     * Returns (log_partition, token_id -> expected_count).
     */
    std::pair<log_prob_t, std::unordered_map<token_id_t, double>> calc_marginal() const;

private:
    std::vector<atomic_token_t> atomic_tokens_;
    std::vector<std::vector<std::tuple<token_id_t, size_t, log_prob_t>>> tokens_from_pos_;
    log_prob_t token_bias_;
    
    std::pair<std::vector<log_prob_t>, std::vector<log_prob_t>> forward_backward() const;
};

// Stable log(exp(a) + exp(b))
inline log_prob_t logaddexp(log_prob_t a, log_prob_t b) {
    if (a < b) std::swap(a, b);
    if (b == NEG_INF) return a;
    return a + std::log1p(std::exp(b - a));
}

}  // namespace unigram
}  // namespace script_bpe

