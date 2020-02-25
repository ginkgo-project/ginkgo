/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_BENCHMARK_UTILS_FORMATS_HPP_
#define GKO_BENCHMARK_UTILS_FORMATS_HPP_


#include <ginkgo/ginkgo.hpp>


#include <map>
#include <string>


#include <gflags/gflags.h>


#ifdef HAS_CUDA
#include "cuda_linops.hpp"
#endif  // HAS_CUDA
#ifdef HAS_HIP
#include "hip_linops.hip.hpp"
#endif  // HAS_HIP


namespace formats {


std::string available_format =
    "coo, csr, ell, sellp, hybrid, hybrid0, hybrid25, hybrid33, hybrid40, "
    "hybrid60, hybrid80, hybridlimit0, hybridlimit25, hybridlimit33, "
    "hybridminstorage"
#ifdef HAS_CUDA
    ", cusp_csr, cusp_csrex, cusp_csrmp, cusp_csrmm, cusp_coo, cusp_ell, "
    "cusp_hybrid"
#endif  // HAS_CUDA
#ifdef HAS_HIP
    ", hipsp_csr, hipsp_csrmm, hipsp_coo, hipsp_ell, hipsp_hybrid"
#endif  // HAS_HIP
    ".\n";

std::string format_description =
    "coo: Coordinate storage. The CUDA kernel uses the load-balancing approach "
    "suggested in Flegar et al.: Overcoming Load Imbalance for Irregular "
    "Sparse Matrices.\n"
    "csr: Compressed Sparse Row storage. Ginkgo implementation with automatic "
    "strategy.\n"
    "csrc: Ginkgo's CSR implementation with automatic stategy.\n"
    "csri: Ginkgo's CSR implementation with inbalance strategy.\n"
    "csrm: Ginkgo's CSR implementation with merge_path strategy.\n"
    "ell: Ellpack format according to Bell and Garland: Efficient Sparse "
    "Matrix-Vector Multiplication on CUDA.\n"
    "sellp: Sliced Ellpack uses a default block size of 32.\n"
    "hybrid: Hybrid uses ell and coo to represent the matrix.\n"
    "hybrid0, hybrid25, hybrid33, hybrid40, hybrid60, hybrid80: Hybrid uses "
    "the row distribution to decide the partition.\n"
    "hybridlimit0, hybridlimit25, hybrid33: Add the upper bound on the ell "
    "part of hybrid0, hybrid25, hybrid33.\n"
    "hybridminstorage: Hybrid uses the minimal storage to store the matrix."
#ifdef HAS_CUDA
    "\n"
    "cusp_hybrid: benchmark CuSPARSE spmv with cusparseXhybmv and an automatic "
    "partition.\n"
    "cusp_coo: use cusparseXhybmv with a CUSPARSE_HYB_PARTITION_USER "
    "partition.\n"
    "cusp_ell: use cusparseXhybmv with CUSPARSE_HYB_PARTITION_MAX partition.\n"
    "cusp_csr: benchmark CuSPARSE with the cusparseXcsrmv function.\n"
    "cusp_csrex: benchmark CuSPARSE with the cusparseXcsrmvEx function.\n"
    "cusp_csrmp: benchmark CuSPARSE with the cusparseXcsrmv_mp function.\n"
    "cusp_csrmm: benchmark CuSPARSE with the cusparseXcsrmv_mm function."
#endif  // HAS_CUDA
#ifdef HAS_HIP
    "hipsp_csr: benchmark HipSPARSE with the hipsparseXcsrmv function.\n"
    "hipsp_csrmm: benchmark HipSPARSE with the hipsparseXcsrmv_mm function.\n"
    "hipsp_hybrid: benchmark HipSPARSE spmv with hipsparseXhybmv and an "
    "automatic partition.\n"
    "hipsp_coo: use hipsparseXhybmv with a HIPSPARSE_HYB_PARTITION_USER "
    "partition.\n"
    "hipsp_ell: use hipsparseXhybmv with HIPSPARSE_HYB_PARTITION_MAX partition."
#endif  // HAS_HIP
    ;

std::string format_command =
    "A comma-separated list of formats to run. Supported values are: " +
    available_format + format_description;


}  // namespace formats


// the formats command-line argument
DEFINE_string(formats, "coo", formats::format_command.c_str());


namespace formats {


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
    matrix_factory
{
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
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 10010)
        {"cusp_gcsr", read_matrix_from_data<cusp_gcsr>},
#endif  // defined(CUDA_VERSION) && (CUDA_VERSION >= 10010)
#endif  // HAS_CUDA
#ifdef HAS_HIP
        {"hipsp_csr", read_matrix_from_data<hipsp_csr>},
        {"hipsp_csrmm", read_matrix_from_data<hipsp_csrmm>},
        {"hipsp_hybrid", read_matrix_from_data<hipsp_hybrid>},
        {"hipsp_coo", read_matrix_from_data<hipsp_coo>},
        {"hipsp_ell", read_matrix_from_data<hipsp_ell>},
#endif  // HAS_HIP
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
    {
        "sellp", read_matrix_from_data<gko::matrix::Sellp<>>
    }
};


}  // namespace formats

#endif  // GKO_BENCHMARK_UTILS_FORMATS_HPP_