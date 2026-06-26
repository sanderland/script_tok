/**
 * Python bindings for Unigram C++ acceleration.
 * 
 * TODO: Complete bindings when unigram_core.cpp is implemented.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "unigram_core.hpp"

namespace py = pybind11;
using namespace script_bpe::unigram;

PYBIND11_MODULE(fast_unigram_cpp, m) {
    m.doc() = "Fast C++ Unigram tokenizer implementation (Trie, Lattice, Viterbi, Forward-Backward)";
    
    py::class_<FastTrie>(m, "FastTrie")
        .def(py::init<>())
        .def("insert", &FastTrie::insert,
             py::arg("token_id"), py::arg("atomic_tokens"), py::arg("log_prob"),
             "Insert a token into the trie")
        .def("find_prefixes", &FastTrie::find_prefixes,
             py::arg("seq"), py::arg("offset"),
             "Find all tokens that are prefixes of seq starting at offset");
    
    py::class_<FastLattice>(m, "FastLattice")
        .def(py::init<const std::vector<atomic_token_t>&,
                      const std::vector<std::vector<std::tuple<token_id_t, size_t, log_prob_t>>>&,
                      log_prob_t>(),
             py::arg("atomic_tokens"),
             py::arg("tokens_from_pos"),
             py::arg("token_bias") = 0.0)
        .def("viterbi", &FastLattice::viterbi,
             py::arg("allow_single_token") = true,
             "Run Viterbi algorithm to find best segmentation")
        .def("calc_marginal", &FastLattice::calc_marginal,
             "Run forward-backward to compute marginal token probabilities");
}

