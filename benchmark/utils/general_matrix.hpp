/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_
#define GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_


#include <ginkgo/ginkgo.hpp>


#include <gflags/gflags.h>


#include "benchmark/utils/general.hpp"
#include "benchmark/utils/generator.hpp"


std::string reordering_algorithm_desc =
    "Reordering algorithm to apply to the input matrices:\n"
    "    none - no reordering\n"
    "    amd - Approximate Minimum Degree reordering algorithm\n"
#if GKO_HAVE_METIS
    "    nd - Nested Dissection reordering algorithm\n"
#endif
    "    rcm - Reverse Cuthill-McKee reordering algorithm";


DEFINE_string(input_matrix, "",
              "Filename of a matrix to be used as the single input. Overwrites "
              "the value of the -input flag");

DEFINE_string(reorder, "none", reordering_algorithm_desc.c_str());


template <typename ValueType, typename IndexType>
std::unique_ptr<gko::matrix::Permutation<IndexType>> reorder(
    gko::matrix_data<ValueType, IndexType>& data, json& test_case,
    bool is_distributed = false)
{
    if (FLAGS_reorder == "none" || is_distributed) {
        return nullptr;
    }
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    auto ref = gko::ReferenceExecutor::create();
    auto mtx = gko::share(Csr::create(ref));
    mtx->read(data);
    std::unique_ptr<gko::matrix::Permutation<IndexType>> perm;
    if (FLAGS_reorder == "amd") {
        perm = gko::experimental::reorder::Amd<IndexType>::build()
                   .on(ref)
                   ->generate(mtx);
#if GKO_HAVE_METIS
    } else if (FLAGS_reorder == "nd") {
        perm = gko::experimental::reorder::NestedDissection<ValueType,
                                                            IndexType>::build()
                   .on(ref)
                   ->generate(mtx);
#endif
    } else if (FLAGS_reorder == "rcm") {
        perm = gko::reorder::Rcm<ValueType, IndexType>::build()
                   .on(ref)
                   ->generate(mtx)
                   ->get_permutation()
                   ->clone();
    } else {
        throw std::runtime_error{"Unknown reordering algorithm " +
                                 FLAGS_reorder};
    }
    auto perm_arr =
        gko::array<IndexType>::view(ref, data.size[0], perm->get_permutation());
    gko::as<Csr>(mtx->permute(&perm_arr))->write(data);
    test_case["reordered"] = FLAGS_reorder;
    return perm;
}


template <typename ValueType, typename IndexType>
void permute(std::unique_ptr<gko::matrix::Dense<ValueType>>& vec,
             gko::matrix::Permutation<IndexType>* perm)
{
    auto perm_arr = gko::array<IndexType>::view(
        perm->get_executor(), perm->get_size()[0], perm->get_permutation());
    vec = gko::as<gko::matrix::Dense<ValueType>>(vec->row_permute(&perm_arr));
}


template <typename ValueType, typename IndexType>
void permute(
    std::unique_ptr<gko::experimental::distributed::Vector<ValueType>>& vec,
    gko::matrix::Permutation<IndexType>* perm)
{}


/**
 * @copydoc initialize_argument_parsing
 * @param additional_matrix_file_json  text to be appended to the
 *                                     `{"filename":"..."}` JSON object that
 *                                     will be used as input for the benchmark
 *                                     if the `-input_matrix` flag is used.
 */
void initialize_argument_parsing_matrix(
    int* argc, char** argv[], std::string& header, std::string& format,
    std::string additional_matrix_file_json = "", bool do_print = true)
{
    initialize_argument_parsing(argc, argv, header, format, do_print);
    std::string input_matrix_str{FLAGS_input_matrix};
    if (!input_matrix_str.empty()) {
        if (input_stream) {
            std::cerr
                << "-input and -input_matrix cannot be used simultaneously\n";
            std::exit(1);
        }
        // create JSON for the filename via nlohmann_json to ensure the string
        // is correctly escaped
        auto json_template =
            R"([{"filename":"")" + additional_matrix_file_json + "}]";
        auto doc = json::parse(json_template);
        doc[0]["filename"] = input_matrix_str;
        input_stream = std::make_unique<std::stringstream>(doc.dump());
    }
}


#endif  // GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_
