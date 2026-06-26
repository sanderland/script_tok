/**
 * C++ implementation of Unigram tokenization algorithms.
 * 
 * TODO: Implement these algorithms for fast Unigram training and inference.
 * Currently these are stubs that will be filled in.
 */

#include "unigram_core.hpp"
#include <algorithm>
#include <cassert>

namespace script_bpe {
namespace unigram {

void FastTrie::insert(token_id_t token_id, const std::vector<atomic_token_t>& atomic_tokens, log_prob_t log_prob) {
    size_t node_idx = 0;  // start at root
    for (atomic_token_t tok : atomic_tokens) {
        auto& node = nodes_[node_idx];
        auto it = node.children.find(tok);
        if (it == node.children.end()) {
            size_t new_idx = nodes_.size();
            nodes_.emplace_back();
            node.children[tok] = new_idx;
            node_idx = new_idx;
        } else {
            node_idx = it->second;
        }
    }
    nodes_[node_idx].token_id = token_id;
    nodes_[node_idx].log_prob = log_prob;
}

std::vector<std::tuple<token_id_t, size_t, log_prob_t>> FastTrie::find_prefixes(
    const std::vector<atomic_token_t>& seq, size_t offset) const {
    
    std::vector<std::tuple<token_id_t, size_t, log_prob_t>> results;
    size_t node_idx = 0;  // start at root
    
    for (size_t i = offset; i < seq.size(); ++i) {
        atomic_token_t tok = seq[i];
        const auto& node = nodes_[node_idx];
        auto it = node.children.find(tok);
        if (it == node.children.end()) {
            break;  // no more matches
        }
        node_idx = it->second;
        const auto& next_node = nodes_[node_idx];
        if (next_node.token_id != -1) {
            size_t length = i - offset + 1;
            results.emplace_back(next_node.token_id, length, next_node.log_prob);
        }
    }
    
    return results;
}

FastLattice::FastLattice(
    const std::vector<atomic_token_t>& atomic_tokens,
    const std::vector<std::vector<std::tuple<token_id_t, size_t, log_prob_t>>>& tokens_from_pos,
    log_prob_t token_bias)
    : atomic_tokens_(atomic_tokens)
    , tokens_from_pos_(tokens_from_pos)
    , token_bias_(token_bias) {}

std::pair<std::vector<log_prob_t>, std::vector<log_prob_t>> FastLattice::forward_backward() const {
    size_t n = atomic_tokens_.size();
    
    // alpha[i] = total log-prob of paths reaching position i
    std::vector<log_prob_t> alpha(n + 1, NEG_INF);
    alpha[0] = 0.0;
    
    // Forward pass
    for (size_t pos = 0; pos < n; ++pos) {
        if (alpha[pos] == NEG_INF) continue;
        for (const auto& [token_id, length, log_prob] : tokens_from_pos_[pos]) {
            size_t end_pos = pos + length;
            log_prob_t effective_prob = log_prob - token_bias_;
            alpha[end_pos] = logaddexp(alpha[end_pos], alpha[pos] + effective_prob);
        }
    }
    
    // beta[i] = total log-prob of paths from position i to end
    std::vector<log_prob_t> beta(n + 1, NEG_INF);
    beta[n] = 0.0;
    
    // Backward pass
    for (size_t pos = n; pos-- > 0;) {
        for (const auto& [token_id, length, log_prob] : tokens_from_pos_[pos]) {
            size_t end_pos = pos + length;
            if (beta[end_pos] == NEG_INF) continue;
            log_prob_t effective_prob = log_prob - token_bias_;
            beta[pos] = logaddexp(beta[pos], beta[end_pos] + effective_prob);
        }
    }
    
    return {alpha, beta};
}

std::pair<std::vector<token_id_t>, log_prob_t> FastLattice::viterbi(bool allow_single_token) const {
    size_t n = atomic_tokens_.size();
    
    // best_at_pos[i] = (best_token_to_reach_here, best_score, came_from_pos)
    std::vector<std::tuple<token_id_t, log_prob_t, size_t>> best_at_pos(n + 1, {-1, NEG_INF, 0});
    std::get<1>(best_at_pos[0]) = 0.0;
    
    for (size_t pos = 0; pos < n; ++pos) {
        log_prob_t current_score = std::get<1>(best_at_pos[pos]);
        if (current_score == NEG_INF) continue;
        
        for (const auto& [token_id, length, log_prob] : tokens_from_pos_[pos]) {
            size_t end_pos = pos + length;
            
            // Skip single-token paths if not allowed
            if (!allow_single_token && pos == 0 && end_pos == n) continue;
            
            log_prob_t effective_prob = log_prob - token_bias_;
            log_prob_t score = current_score + effective_prob;
            
            if (score > std::get<1>(best_at_pos[end_pos])) {
                best_at_pos[end_pos] = {token_id, score, pos};
            }
        }
    }
    
    // Backtrack to get path
    std::vector<token_id_t> path;
    size_t pos = n;
    while (pos > 0) {
        auto [token_id, score, came_from] = best_at_pos[pos];
        if (token_id == -1) break;
        path.push_back(token_id);
        pos = came_from;
    }
    std::reverse(path.begin(), path.end());
    
    return {path, std::get<1>(best_at_pos[n])};
}

std::pair<log_prob_t, std::unordered_map<token_id_t, double>> FastLattice::calc_marginal() const {
    auto [alpha, beta] = forward_backward();
    log_prob_t z = alpha.back();  // partition function
    
    std::unordered_map<token_id_t, double> token_prob;
    
    for (size_t pos = 0; pos < atomic_tokens_.size(); ++pos) {
        for (const auto& [token_id, length, log_prob] : tokens_from_pos_[pos]) {
            size_t end_pos = pos + length;
            log_prob_t effective_prob = log_prob - token_bias_;
            log_prob_t token_logprob = alpha[pos] + effective_prob + beta[end_pos] - z;
            
            // Avoid underflow
            double prob = std::exp(std::max(-100.0, token_logprob));
            token_prob[token_id] += prob;
        }
    }
    
    return {z, token_prob};
}

}  // namespace unigram
}  // namespace script_bpe

