#pragma once

#include <vector>
#include <queue>
#include <string>
#include <utility>
#include <functional>
#include <cstdint>
#include <tuple>

#ifndef NO_PYTHON_BINDINGS
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    namespace py = pybind11;
    typedef py::array_t<int> encode_return_t;
#else
    typedef std::vector<int> encode_return_t;
#endif

namespace std {
    template<>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            size_t h1 = hash<int>{}(p.first);
            size_t h2 = hash<int>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };
}


using token_t = int32_t;
using base_token_t = int16_t;
using token_arr_t = std::vector<token_t>;
// All map-specific configuration in a single block
// typedef std::pair<int, int> merge_key_t;
using merge_key_t = uint64_t;
using merge_value_t = token_t;
inline merge_key_t make_merge_key(int a, int b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
}

struct MergeItem {
    int32_t to_id;
    int32_t from_a;
    int32_t from_b;
    int32_t val_a;
    int32_t val_b;
    
    bool operator<(const MergeItem& other) const {
        return std::tie(to_id, from_a) < std::tie(other.to_id, other.from_a);
    }
    bool operator>(const MergeItem& other) const {
        return std::tie(to_id, from_a) > std::tie(other.to_id, other.from_a);
    }
};

#ifdef MERGE_MAP_STD
    #include <unordered_map>
    #define MERGE_MAP_NAME "std::unordered_map"
    using merge_map_t = std::unordered_map<merge_key_t, merge_value_t>;
#elif defined(MERGE_MAP_ABSL)
    #include "absl/container/flat_hash_map.h"
    #define MERGE_MAP_NAME "absl::flat_hash_map"
    using merge_map_t = absl::flat_hash_map<merge_key_t, merge_value_t>;
#elif defined(MERGE_MAP_GTL)
    #include "gtl/phmap.hpp"
    #define MERGE_MAP_NAME "gtl::flat_hash_map"
    using merge_map_t = gtl::flat_hash_map<merge_key_t, merge_value_t>;
#elif defined(MERGE_MAP_ROBINHOOD)
    #include "robin_hood.h"
    #define MERGE_MAP_NAME "robin_hood::unordered_flat_map"
    using merge_map_t = robin_hood::unordered_flat_map<merge_key_t, merge_value_t>;
#elif defined(MERGE_MAP_SKA)
    #include "flat_hash_map.hpp"
    #define MERGE_MAP_NAME "ska::flat_hash_map"
    using merge_map_t = ska::flat_hash_map<merge_key_t, merge_value_t>;
#else
    #error "One of MERGE_MAP_STD, MERGE_MAP_ABSL, MERGE_MAP_GTL, MERGE_MAP_ROBINHOOD, or MERGE_MAP_SKA must be defined"
#endif


#ifdef PQ_STD_HEAP
    using pq_t = std::vector<MergeItem>;
    inline void reserve_heap(pq_t& heap, size_t size) {
        heap.reserve(size);
    }
    inline void push_heap(pq_t& heap, const MergeItem& item) {
        heap.push_back(item);
        std::push_heap(heap.begin(), heap.end(), std::greater<MergeItem>());
    }
    inline const MergeItem& top_heap(const pq_t& heap) {
        return heap.front();
    }
    inline void pop_heap(pq_t& heap) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<MergeItem>());
        heap.pop_back();
    }
    inline bool empty_heap(const pq_t& heap) {
        return heap.empty();
    }
#elif defined(PQ_BOOST_4ARY)
    #include <boost/heap/d_ary_heap.hpp>
    using pq_t = boost::heap::d_ary_heap<MergeItem, boost::heap::arity<4>, boost::heap::compare<std::greater<MergeItem>>>;
    inline void reserve_heap(pq_t& heap, size_t size) {
        heap.reserve(size);
    }
    inline void push_heap(pq_t& heap, const MergeItem& item) {
        heap.push(item);
    }
    inline const MergeItem& top_heap(const pq_t& heap) {
        return heap.top();
    }
    inline void pop_heap(pq_t& heap) {
        heap.pop();
    }
    inline bool empty_heap(const pq_t& heap) {
        return heap.empty();
    }
#else
    #error "Define one of: PQ_STD_HEAP, PQ_BOOST_4ARY, PQ_BOOST_3ARY, PQ_SKA_MIN_MAX, PQ_SKA_4ARY"
#endif


namespace script_bpe {

    struct CharSCRIPTEnc {
        token_t char_token_id; // -1 if character is raw pair
        base_token_t script_id;
        base_token_t block_id;
        base_token_t index_id;
    };

    struct WorkerState {
        pq_t merge_heap;
        token_arr_t token_array;
    };

    class FastTokenizer {
    public:
        FastTokenizer(const std::vector<CharSCRIPTEnc>& char_script_enc,
                     const std::unordered_map<std::pair<int, int>, merge_value_t>& merge_rules);
        ~FastTokenizer();
        encode_return_t encode(const std::u32string& text);
        
    private:
        std::vector<CharSCRIPTEnc> char_script_enc_;
        merge_map_t merge_rules_;
        token_t whitespace_script_id_, inherited_script_id_;
        WorkerState worker_state_;

        void try_push_merge(pq_t& merge_heap,
                          int a, int b, const token_arr_t& token_array);
        void apply_bpe_merging(WorkerState& worker_state, int start, int end);
        int remove_gaps(token_arr_t& token_array, int end);
    };
}
