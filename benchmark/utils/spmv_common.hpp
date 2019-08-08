/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/ginkgo.hpp>


#include <cstdlib>
#include <iostream>


#include <rapidjson/document.h>


// some shortcuts
using hybrid = gko::matrix::Hybrid<>;
using csr = gko::matrix::Csr<>;

/**
 * Function which outputs the input format for benchmarks similar to the spmv.
 */
[[noreturn]] void print_config_error_and_exit()
{
    std::cerr << "Input has to be a JSON array of matrix configurations:\n"
              << "  [\n"
              << "    { \"filename\": \"my_file.mtx\" },\n"
              << "    { \"filename\": \"my_file2.mtx\" }\n"
              << "  ]" << std::endl;
    std::exit(1);
}


/**
 * Validates whether the input format is correct for spmv-like benchmarks.
 *
 * @param value  the JSON value to test.
 */
void validate_option_object(const rapidjson::Value &value)
{
    if (!value.IsObject() || !value.HasMember("filename") ||
        !value["filename"].IsString()) {
        print_config_error_and_exit();
    }
}


/**
 * Creates a Ginkgo matrix from the intermediate data representation format
 * gko::matrix_data.
 *
 * @param exec  the executor where the matrix will be put
 * @param data  the data represented in the intermediate representation format
 *
 * @tparam MatrixType  the Ginkgo matrix type (such as `gko::matrix::Csr<>`)
 *
 * @return a `unique_pointer` to the created matrix
 */
template <typename MatrixType>
std::unique_ptr<MatrixType> read_matrix_from_data(
    std::shared_ptr<const gko::Executor> exec, const gko::matrix_data<> &data)
{
    auto mat = MatrixType::create(std::move(exec));
    mat->read(data);
    return mat;
}

/**
 * Creates a Ginkgo matrix from the intermediate data representation format
 * gko::matrix_data with support for variable arguments.
 *
 * @param MATRIX_TYPE  the Ginkgo matrix type (such as `gko::matrix::Csr<>`)
 */
#define READ_MATRIX(MATRIX_TYPE, ...)                                    \
    [](std::shared_ptr<const gko::Executor> exec,                        \
       const gko::matrix_data<> &data) -> std::unique_ptr<MATRIX_TYPE> { \
        auto mat = MATRIX_TYPE::create(std::move(exec), __VA_ARGS__);    \
        mat->read(data);                                                 \
        return mat;                                                      \
    }
