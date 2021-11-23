/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include <algorithm>
#include <map>
#include <string>


#include <gflags/gflags.h>


#ifdef HAS_CUDA
#include "benchmark/utils/cuda_linops.hpp"
#endif  // HAS_CUDA
#ifdef HAS_HIP
#include "benchmark/utils/hip_linops.hip.hpp"
#endif  // HAS_HIP


#include "benchmark/utils/types.hpp"


namespace formats {


std::string available_format =
    "batch_csr,batch_ell,coo, csr, ell, ell-mixed, sellp, hybrid, hybrid0, "
    "hybrid25, "
    "hybrid33, "
    "hybrid40, "
    "hybrid60, hybrid80, hybridlimit0, hybridlimit25, hybridlimit33, "
    "hybridminstorage"
#ifdef HAS_CUDA
    ", cusp_csr, cusp_csrex, cusp_coo"
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
    ", cusp_csrmp, cusp_csrmm, cusp_ell, cusp_hybrid"
#endif  // defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
#if defined(CUDA_VERSION) &&  \
    (CUDA_VERSION >= 11000 || \
     ((CUDA_VERSION >= 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))
    ", cusp_gcsr, cusp_gcsr2, cusp_gcoo"
#endif  // defined(CUDA_VERSION) && (CUDA_VERSION >= 11000 || ((CUDA_VERSION >=
        // 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))
#endif  // HAS_CUDA
#ifdef HAS_HIP
    ", hipsp_csr, hipsp_csrmm, hipsp_coo, hipsp_ell, hipsp_hybrid"
#endif  // HAS_HIP
    ".\n";

std::string format_description =
    "batch_csr: An optimized storage format for batch matrices with the same "
    "sparsity pattern\n"
    "batch_ell: An ELLPACK storage format optimized for batch matrices with "
    "the same sparsity pattern\n"
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
    "ell-mixed: Mixed Precision Ellpack format according to Bell and Garland: "
    "Efficient Sparse Matrix-Vector Multiplication on CUDA.\n"
    "sellp: Sliced Ellpack uses a default block size of 32.\n"
    "hybrid: Hybrid uses ell and coo to represent the matrix.\n"
    "hybrid0, hybrid25, hybrid33, hybrid40, hybrid60, hybrid80: Hybrid uses "
    "the row distribution to decide the partition.\n"
    "hybridlimit0, hybridlimit25, hybrid33: Add the upper bound on the ell "
    "part of hybrid0, hybrid25, hybrid33.\n"
    "hybridminstorage: Hybrid uses the minimal storage to store the matrix."
#ifdef HAS_CUDA
    "\n"
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
    "cusp_coo: use cusparseXhybmv with a CUSPARSE_HYB_PARTITION_USER "
    "partition.\n"
    "cusp_csr: benchmark CuSPARSE with the cusparseXcsrmv function.\n"
    "cusp_ell: use cusparseXhybmv with CUSPARSE_HYB_PARTITION_MAX partition.\n"
    "cusp_csrmp: benchmark CuSPARSE with the cusparseXcsrmv_mp function.\n"
    "cusp_csrmm: benchmark CuSPARSE with the cusparseXcsrmv_mm function.\n"
    "cusp_hybrid: benchmark CuSPARSE spmv with cusparseXhybmv and an automatic "
    "partition.\n"
#else  // CUDA_VERSION >= 11000
    "cusp_csr: is an alias of cusp_gcsr.\n"
    "cusp_coo: is an alias of cusp_gcoo.\n"
#endif
    "cusp_csrex: benchmark CuSPARSE with the cusparseXcsrmvEx function."
#if defined(CUDA_VERSION) &&  \
    (CUDA_VERSION >= 11000 || \
     ((CUDA_VERSION >= 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))
    "\n"
    "cusp_gcsr: benchmark CuSPARSE with the generic csr with default "
    "algorithm.\n"
    "cusp_gcsr2: benchmark CuSPARSE with the generic csr with "
    "CUSPARSE_CSRMV_ALG2.\n"
    "cusp_gcoo: benchmark CuSPARSE with the generic coo with default "
    "algorithm.\n"
#endif  // defined(CUDA_VERSION) && (CUDA_VERSION >= 11000 || ((CUDA_VERSION >=
        // 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))
#endif  // HAS_CUDA
#ifdef HAS_HIP
    "\n"
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
DEFINE_string(formats, "batch_csr", formats::format_command.c_str());

DEFINE_int64(ell_imbalance_limit, 100,
             "Maximal storage overhead above which ELL benchmarks will be "
             "skipped. Negative values mean no limit.");


namespace formats {


// some shortcuts
using hybrid = gko::matrix::Hybrid<etype, itype>;
using csr = gko::matrix::Csr<etype, itype>;
using batch_csr = gko::matrix::BatchCsr<etype>;
using batch_ell = gko::matrix::BatchEll<etype>;
using batch_dense = gko::matrix::BatchDense<etype>;

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
std::unique_ptr<MatrixType> read_batch_matrix_from_data(
    std::shared_ptr<const gko::Executor> exec, const int num_duplications,
    const gko::matrix_data<etype>& data)
{
    using FormatBaseType = typename MatrixType::unbatch_type;
    auto out_mat = FormatBaseType::create(exec);
    out_mat->read(data);
    auto mat = MatrixType::create(exec, num_duplications, out_mat.get());
    return mat;
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
std::unique_ptr<MatrixType> read_batch_matrix_from_batch_data(
    std::shared_ptr<const gko::Executor> exec, const int num_duplications,
    const std::vector<gko::matrix_data<etype>>& data)
{
    auto single_batch = MatrixType::create(exec);
    single_batch->read(data);
    auto mat = MatrixType::create(exec, num_duplications, single_batch.get());
    return mat;
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
    std::shared_ptr<const gko::Executor> exec,
    const gko::matrix_data<etype, itype>& data)
{
    auto mat = MatrixType::create(std::move(exec));
    mat->read(data);
    return mat;
}


/**
 * Creates a CSR strategy of the given type for the given executor if possible,
 * falls back to csr::classical for executors without support for this strategy.
 *
 * @tparam Strategy  one of csr::automatical or csr::load_balance
 */
template <typename Strategy>
std::shared_ptr<csr::strategy_type> create_gpu_strategy(
    std::shared_ptr<const gko::Executor> exec)
{
    if (auto cuda = dynamic_cast<const gko::CudaExecutor*>(exec.get())) {
        return std::make_shared<Strategy>(cuda->shared_from_this());
    } else if (auto hip = dynamic_cast<const gko::HipExecutor*>(exec.get())) {
        return std::make_shared<Strategy>(hip->shared_from_this());
    } else {
        return std::make_shared<csr::classical>();
    }
}


/**
 * Checks whether the given matrix data exceeds the ELL imbalance limit set by
 * the --ell_imbalance_limit flag
 *
 * @throws gko::Error if the imbalance limit is exceeded
 */
void check_ell_admissibility(const gko::matrix_data<etype, itype>& data)
{
    if (data.size[0] == 0 || FLAGS_ell_imbalance_limit < 0) {
        return;
    }
    std::vector<gko::size_type> row_lengths(data.size[0]);
    for (auto nz : data.nonzeros) {
        row_lengths[nz.row]++;
    }
    auto max_len = *std::max_element(row_lengths.begin(), row_lengths.end());
    auto avg_len = data.nonzeros.size() / std::max<double>(data.size[0], 1);
    if (max_len / avg_len > FLAGS_ell_imbalance_limit) {
        throw gko::Error(__FILE__, __LINE__,
                         "Matrix exceeds ELL imbalance limit");
    }
}


/**
 * Creates a Ginkgo matrix from the intermediate data representation format
 * gko::matrix_data with support for variable arguments.
 *
 * @param MATRIX_TYPE  the Ginkgo matrix type (such as `gko::matrix::Csr<>`)
 */
#define READ_MATRIX(MATRIX_TYPE, ...)                                 \
    [](std::shared_ptr<const gko::Executor> exec,                     \
       const gko::matrix_data<etype, itype>& data)                    \
        -> std::unique_ptr<MATRIX_TYPE> {                             \
        auto mat = MATRIX_TYPE::create(std::move(exec), __VA_ARGS__); \
        mat->read(data);                                              \
        return mat;                                                   \
    }


const std::map<std::string, std::function<std::unique_ptr<gko::BatchLinOp>(
                                std::shared_ptr<const gko::Executor>, const int,
                                const std::vector<gko::matrix_data<etype>>&)>>
    batch_matrix_factory2{
        {"batch_csr",
         read_batch_matrix_from_batch_data<gko::matrix::BatchCsr<etype>>},
        {"batch_ell",
         read_batch_matrix_from_batch_data<gko::matrix::BatchEll<etype>>},
        {"batch_dense",
         read_batch_matrix_from_batch_data<gko::matrix::BatchDense<etype>>}};


const std::map<std::string, std::function<std::unique_ptr<gko::BatchLinOp>(
                                std::shared_ptr<const gko::Executor>, const int,
                                const gko::matrix_data<etype>&)>>
    batch_matrix_factory{
        {"batch_ell",
         read_batch_matrix_from_data<gko::matrix::BatchEll<etype>>},
        {"batch_csr",
         read_batch_matrix_from_data<gko::matrix::BatchCsr<etype>>},
        {"batch_dense",
         read_batch_matrix_from_data<gko::matrix::BatchDense<etype>>}};


// clang-format off
const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>,
                                const gko::matrix_data<etype, itype> &)>>
    matrix_factory{
        {"csr",
         [](std::shared_ptr<const gko::Executor> exec,
            const gko::matrix_data<etype, itype> &data) -> std::unique_ptr<csr> {
            auto mat =
                csr::create(exec, create_gpu_strategy<csr::automatical>(exec));
            mat->read(data);
            return mat;
         }},
        {"csri",
         [](std::shared_ptr<const gko::Executor> exec,
            const gko::matrix_data<etype, itype> &data) -> std::unique_ptr<csr> {
             auto mat = csr::create(
                 exec, create_gpu_strategy<csr::load_balance>(exec));
             mat->read(data);
             return mat;
         }},
        {"csrm", READ_MATRIX(csr, std::make_shared<csr::merge_path>())},
        {"csrc", READ_MATRIX(csr, std::make_shared<csr::classical>())},
        {"coo", read_matrix_from_data<gko::matrix::Coo<etype, itype>>},
        {"ell", [](std::shared_ptr<const gko::Executor> exec,
            const gko::matrix_data<etype, itype> &data) {
             check_ell_admissibility(data);
             auto mat = gko::matrix::Ell<etype, itype>::create(exec);
             mat->read(data);
             return mat;
         }},
        {"ell-mixed",
         [](std::shared_ptr<const gko::Executor> exec,
            const gko::matrix_data<etype, itype> &data) {
             check_ell_admissibility(data);
             gko::matrix_data<gko::next_precision<etype>, itype> conv_data;
             conv_data.size = data.size;
             conv_data.nonzeros.resize(data.nonzeros.size());
             auto it = conv_data.nonzeros.begin();
             for (auto &el : data.nonzeros) {
                 it->row = el.row;
                 it->column = el.column;
                 it->value = el.value;
                 ++it;
             }
             auto mat = gko::matrix::Ell<gko::next_precision<etype>, itype>::create(
                 std::move(exec));
             mat->read(conv_data);
             return mat;
         }},
#ifdef HAS_CUDA
#if defined(CUDA_VERSION) && (CUDA_VERSION < 11000)
        {"cusp_csr", read_matrix_from_data<cusp_csr>},
        {"cusp_csrmp", read_matrix_from_data<cusp_csrmp>},
        {"cusp_csrmm", read_matrix_from_data<cusp_csrmm>},
        {"cusp_hybrid", read_matrix_from_data<cusp_hybrid>},
        {"cusp_coo", read_matrix_from_data<cusp_coo>},
        {"cusp_ell", read_matrix_from_data<cusp_ell>},
#else  // CUDA_VERSION >= 11000
       // cusp_csr, cusp_coo use the generic ones from CUDA 11
        {"cusp_csr", read_matrix_from_data<cusp_gcsr>},
        {"cusp_coo", read_matrix_from_data<cusp_gcoo>},
#endif
        {"cusp_csrex", read_matrix_from_data<cusp_csrex>},
#if defined(CUDA_VERSION) &&  \
    (CUDA_VERSION >= 11000 || \
     ((CUDA_VERSION >= 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))
        {"cusp_gcsr", read_matrix_from_data<cusp_gcsr>},
        {"cusp_gcsr2", read_matrix_from_data<cusp_gcsr2>},
        {"cusp_gcoo", read_matrix_from_data<cusp_gcoo>},
#endif  // defined(CUDA_VERSION) && (CUDA_VERSION >= 11000 || ((CUDA_VERSION >=
        // 10020) && !(defined(_WIN32) || defined(__CYGWIN__))))
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
        {"sellp", read_matrix_from_data<gko::matrix::Sellp<etype, itype>>}
};
// clang-format on


}  // namespace formats

#endif  // GKO_BENCHMARK_UTILS_FORMATS_HPP_
