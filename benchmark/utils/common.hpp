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

#ifndef GKO_BENCHMARK_UTILS_COMMON_HPP_
#define GKO_BENCHMARK_UTILS_COMMON_HPP_


#include <ginkgo/ginkgo.hpp>


#include <map>


#ifdef HAS_CUDA
#include "cuda_linops.hpp"
#endif  // HAS_CUDA


// some shortcuts
using hybrid = gko::matrix::Hybrid<>;
using csr = gko::matrix::Csr<>;


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


const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>,
                                const gko::matrix_data<> &)>>
    matrix_factory{
        {"csr", READ_MATRIX(csr, std::make_shared<csr::automatical>())},
        {"csri", READ_MATRIX(csr, std::make_shared<csr::load_balance>())},
        {"csrm", READ_MATRIX(csr, std::make_shared<csr::merge_path>())},
        {"csrc", READ_MATRIX(csr, std::make_shared<csr::classical>())},
        {"coo", read_matrix_from_data<gko::matrix::Coo<>>},
        {"ell", read_matrix_from_data<gko::matrix::Ell<>>},
#ifdef HAS_CUDA
        {"cusp_csr", read_matrix_from_data<cusp_csr>},
        {"cusp_csrmp", read_matrix_from_data<cusp_csrmp>},
        {"cusp_csrex", read_matrix_from_data<cusp_csrex>},
        {"cusp_csrmm", read_matrix_from_data<cusp_csrmm>},
        {"cusp_hybrid", read_matrix_from_data<cusp_hybrid>},
        {"cusp_coo", read_matrix_from_data<cusp_coo>},
        {"cusp_ell", read_matrix_from_data<cusp_ell>},
#endif  // HAS_CUDA
        {"hybrid", read_matrix_from_data<hybrid>},
        {"hybrid0",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0))},
        {"hybrid25",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.25))},
        {"hybrid33",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_limit>(1.0 / 3.0))},
        {"hybrid40",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.4))},
        {"hybrid60",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.6))},
        {"hybrid80",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_limit>(0.8))},
        {"hybridlimit0",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_bounded_limit>(0))},
        {"hybridlimit25",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::imbalance_bounded_limit>(0.25))},
        {"hybridlimit33",
         READ_MATRIX(hybrid, std::make_shared<hybrid::imbalance_bounded_limit>(
                                 1.0 / 3.0))},
        {"hybridminstorage",
         READ_MATRIX(hybrid,
                     std::make_shared<hybrid::minimal_storage_limit>())},
        {"sellp", read_matrix_from_data<gko::matrix::Sellp<>>}};


#endif  // GKO_BENCHMARK_UTILS_COMMON_HPP_