#include "bpe_core.hpp"
#include <iostream>
#include <iomanip>

// Helper macro for stringification
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace script_bpe {

    FastTokenizer::FastTokenizer(const std::vector<CharSCRIPTEnc>& char_script_enc,
                                     const std::unordered_map<std::pair<int, int>, merge_value_t>& merge_rules)
                : char_script_enc_(char_script_enc) {

        merge_rules_.reserve(merge_rules.size() * 2); // work well for all variants
        for (const auto& rule : merge_rules) {
            merge_rules_[make_merge_key(rule.first.first, rule.first.second)] = rule.second;
        }
        std::cout << "Created " << MERGE_MAP_NAME << ", type: " << typeid(merge_rules_).name() << std::endl;
        std::cout << "Load factor: " << merge_rules_.load_factor() << " size " << merge_rules_.size() << std::endl;

        // Assume special indices are provided by convention on the Python side
        whitespace_script_id_ = char_script_enc_[static_cast<size_t>(U' ')].script_id;
        inherited_script_id_ = char_script_enc_[static_cast<size_t>(U'ー')].script_id; // example for Japanese long vowel mark
        auto inherited_c_script_id_ = char_script_enc_[static_cast<size_t>(U'\u200d')].script_id; // example for zero-width joiner
        auto han_script_id_ = char_script_enc_[static_cast<size_t>(U'漢')].script_id;
        auto hiragana_script_id_ = char_script_enc_[static_cast<size_t>(U'ひ')].script_id;
        // re-code hiragana to han and inherited(c) to inherited(lm)
        for (auto& it : char_script_enc_) {
            if (it.script_id == hiragana_script_id_) {
                it.script_id = han_script_id_;
            }
            if (it.script_id == inherited_c_script_id_) it.script_id = inherited_script_id_;
        }

        worker_state_ = WorkerState(); // Initialize merge heap and token array
        worker_state_.token_array.resize(4096); // Reserve initial size
        reserve_heap(worker_state_.merge_heap, 4096); // Reserve initial size for heap
    }

    encode_return_t FastTokenizer::encode(const std::u32string& text) {
        if (text.empty()) {
            return encode_return_t(); // Return empty array for empty input
        }
        size_t start = 0, end = 0;
        int last_script_id = -1;

        auto& token_array = worker_state_.token_array;
        size_t required_capacity = text.length() * 2;
        if(token_array.size() < required_capacity) {
            token_array.resize(2 * required_capacity);
        }

        for(char32_t ch: text) {
            if (static_cast<size_t>(ch) >= char_script_enc_.size()) {
                continue; // invalid character, skip
            }
            auto& enc = char_script_enc_[static_cast<size_t>(ch)];
            auto& script_id = enc.script_id;
            if (script_id == -1) {
                continue; // invalid character, skip
            }
            if(script_id != last_script_id && script_id != inherited_script_id_) { // new pretoken
                if(last_script_id == whitespace_script_id_ && end-start==1) {
                    last_script_id = script_id; // single space, include, but set script id to non-space
                }
                else {
                    apply_bpe_merging(worker_state_, start, end); // apply BPE merging to previous pretoken
                    start = end; // reset start for new pretoken
                    last_script_id = script_id;
                }
            }
            if (enc.char_token_id == -1) { // still pair, never merged
                token_array[end++] = enc.block_id;
                token_array[end++] = enc.index_id;
            }
            else { // has token id, check script and maybe tokenize pretoken
                token_array[end++] = enc.char_token_id;
            }
        }
        apply_bpe_merging(worker_state_, start, end); // last pretoken
        int final_size = remove_gaps(token_array, end);
#ifndef NO_PYTHON_BINDINGS
        return py::array_t<int>(final_size, token_array.data());
#else
        return std::vector<int>(token_array.begin(), token_array.begin() + final_size);
#endif
    }

    inline void FastTokenizer::try_push_merge(pq_t& merge_heap,
                   int a, int b, const token_arr_t& token_array) {
        merge_key_t merge_key = make_merge_key(token_array[a], token_array[b]);
        auto it = merge_rules_.find(merge_key);
        if (it != merge_rules_.end()) {
            push_heap(merge_heap, {
                it->second,
                a,
                b,
                token_array[a],
                token_array[b],
            });
        }
    }

    int FastTokenizer::remove_gaps(std::vector<int>& token_array, int end) {
        int write_pos = 0;
        for (int read_pos = 0; read_pos < end; ++read_pos) {
            if (token_array[read_pos] != -1) {
                token_array[write_pos++] = token_array[read_pos];
            }
        }
        return write_pos; // Return new size
    }

    void FastTokenizer::apply_bpe_merging(WorkerState& worker_state, int start, int end) {
        if (end - start < 2) {
            return; // Need at least 2 tokens to merge
        }

        // Find all possible merges in this chunk - use consecutive individual tokens
        auto& token_array = worker_state.token_array;
        auto& merge_heap = worker_state.merge_heap;
        for (int i = start; i < end - 1; ++i) {
            this->try_push_merge(merge_heap, i, i+1, token_array);
        }

        // Apply merges in priority order
        while (!empty_heap(merge_heap)) {
            MergeItem item = top_heap(merge_heap);
            pop_heap(merge_heap);
            // Verify merge is still valid - close to 50-50
            if (token_array[item.from_a] != item.val_a ||
                token_array[item.from_b] != item.val_b) continue;

            // Perform merge - replace first token with merged token, mark second token as deleted
            token_array[item.from_a] = item.to_id;
            token_array[item.from_b] = -1;  // Mark as deleted

            // Add new potential merges
            {
                auto& tokens = worker_state.token_array;
                auto& merge_heap = worker_state.merge_heap;
                // Find next valid token after merge
                int next_pos = item.from_b + 1;
                while (next_pos < end && tokens[next_pos] == -1) {
                    next_pos++;
                }
                // Check merge with next token
                if (next_pos < end) {
                    this->try_push_merge(merge_heap, item.from_a, next_pos, tokens);
                }
                // Find previous valid token before merge
                int prev_pos = item.from_a - 1;
                while (prev_pos >= start && tokens[prev_pos] == -1) {
                    prev_pos--;
                }
                // Check merge with previous token
                if (prev_pos >= start) {
                    this->try_push_merge(merge_heap, prev_pos, item.from_a, tokens);
                }
            }
        }
    }

    FastTokenizer::~FastTokenizer() {
    }

} // namespace script_bpe
