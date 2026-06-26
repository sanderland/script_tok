
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <iostream>
#include "bpe_core.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fast_tokenizer_cpp, m) {
    m.doc() = "Fast C++ SCRIPT tokenizer implementation";
    
    // Expose CharSCRIPTEnc struct to Python
    py::class_<script_bpe::CharSCRIPTEnc>(m, "CharSCRIPTEnc")
        .def(py::init<int, base_token_t, base_token_t, base_token_t>());

    py::class_<script_bpe::FastTokenizer>(m, "FastTokenizer")
        .def(py::init<const std::vector<script_bpe::CharSCRIPTEnc>&,
                     const std::unordered_map<std::pair<int, int>, int>&>(),
              py::arg("char_script_enc"), py::arg("merge_rules"))
        .def("encode", &script_bpe::FastTokenizer::encode);
}