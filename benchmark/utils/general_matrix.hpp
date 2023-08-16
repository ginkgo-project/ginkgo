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


DEFINE_string(input_matrix, "",
              "Filename of a matrix to be used as the single input. Overwrites "
              "the value of the -input flag");


/**
 * @copydoc initialize_argument_parsing
 * @param additional_matrix_file_json  text to be appended to the
 *                                     `{"filename":"..."}` JSON object that
 *                                     will be used as input for the benchmark
 *                                     if the `-input_matrix` flag is used.
 */
void initialize_argument_parsing_matrix(
    int* argc, char** argv[], std::string& header, std::string& format,
    std::string additional_matrix_file_json = "")
{
    initialize_argument_parsing(argc, argv, header, format);
    std::string input_matrix_str{FLAGS_input_matrix};
    if (!input_matrix_str.empty()) {
        auto input_json = "[{\"filename\":\"" + input_matrix_str + "\"" +
                          additional_matrix_file_json + "}]";
        input_stream = std::make_unique<std::stringstream>(input_json);
    }
}


#endif  // GKO_BENCHMARK_UTILS_GENERAL_MATRIX_HPP_