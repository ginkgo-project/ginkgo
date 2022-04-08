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

#ifndef GKO_BENCHMARK_UTILS_FORMATS_HPP_
#define GKO_BENCHMARK_UTILS_FORMATS_HPP_


#include <ginkgo/ginkgo.hpp>


#include <algorithm>
#include <map>
#include <string>


#include <gflags/gflags.h>


#include "benchmark/utils/sparselib_linops.hpp"
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
    ", cusparse_csr, cusparse_csrex, cusparse_coo"
    ", cusparse_csrmp, cusparse_csrmm, cusparse_ell, cusparse_hybrid"
    ", cusparse_gcsr, cusparse_gcsr2, cusparse_gcoo"
#endif  // HAS_CUDA
#ifdef HAS_HIP
    ", hipsparse_csr, hipsparse_csrmm, hipsparse_coo, hipsparse_ell, "
    "hipsparse_hybrid"
#endif  // HAS_HIP
#ifdef HAS_DPCPP
    ", onemkl_csr, onemkl_optimized_csr"
#endif  // HAS_DPCPP
    ".\n";

std::string format_description =
    "coo: Coordinate storage. The GPU kernels use the load-balancing "
    "approach\n"
    "     suggested in Flegar et al.: Overcoming Load Imbalance for\n"
    "     Irregular Sparse Matrices.\n"
    "csr: Compressed Sparse Row storage. Ginkgo implementation with\n"
    "     automatic strategy.\n"
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
    "csrs: Ginkgo's CSR implementation with sparselib strategy.\n"
    "ell: Ellpack format according to Bell and Garland: Efficient Sparse\n"
    "     Matrix-Vector Multiplication on CUDA.\n"
    "ell-mixed: Mixed Precision Ellpack format according to Bell and Garland:\n"
    "           Efficient Sparse Matrix-Vector Multiplication on CUDA.\n"
    "sellp: Sliced Ellpack uses a default block size of 32.\n"
    "hybrid: Hybrid uses ELL and COO to represent the matrix.\n"
    "hybrid0, hybrid25, hybrid33, hybrid40, hybrid60, hybrid80:\n"
    "    Use 0%, 25%, ... quantiles of the row length distribution\n"
    "    to choose number of entries stored in the ELL part.\n"
    "hybridlimit0, hybridlimit25, hybrid33: Similar to hybrid0\n"
    "    but with an additional absolute limit on the number of entries\n"
    "    per row stored in ELL.\n"
    "hybridminstorage: Use the minimal storage to store the matrix."
#ifdef HAS_CUDA
    "\n"
    "cusparse_coo: cuSPARSE COO SpMV, using cusparseXhybmv with \n"
    "              CUSPARSE_HYB_PARTITION_USER for CUDA < 10.2, or\n"
    "              the Generic API otherwise\n"
    "cusparse_csr: cuSPARSE CSR SpMV, using cusparseXcsrmv for CUDA < 10.2,\n"
    "              or the Generic API with default algorithm otherwise\n"
    "cusparse_csrex: cuSPARSE CSR SpMV using cusparseXcsrmvEx\n"
    "cusparse_ell: cuSPARSE ELL SpMV using cusparseXhybmv with\n"
    "              CUSPARSE_HYB_PARTITION_MAX, available for CUDA < 11.0\n"
    "cusparse_csrmp: cuSPARSE CSR SpMV using cusparseXcsrmv_mp,\n"
    "                available for CUDA < 11.0\n"
    "cusparse_csrmm: cuSPARSE CSR SpMV using cusparseXcsrmv_mm,\n"
    "                available for CUDA < 11.0\n"
    "cusparse_hybrid: cuSPARSE Hybrid SpMV using cusparseXhybmv\n"
    "                 with an automatic partition, available for CUDA < 11.0\n"
    "cusparse_gcsr: cuSPARSE CSR SpMV using Generic API with default\n"
    "               algorithm, available for CUDA >= 10.2\n"
    "cusparse_gcsr2: cuSPARSE CSR SpMV using Generic API with\n"
    "                CUSPARSE_CSRMV_ALG2, available for CUDA >= 10.2\n"
    "cusparse_gcoo: cuSPARSE Generic API with default COO SpMV,\n"
    "               available for CUDA >= 10.2\n"
#endif  // HAS_CUDA
#ifdef HAS_HIP
    "\n"
    "hipsparse_csr: hipSPARSE CSR SpMV using hipsparseXcsrmv\n"
    "hipsparse_csrmm: hipSPARSE CSR SpMV using hipsparseXcsrmv_mm\n"
    "hipsparse_hybrid: hipSPARSE CSR SpMV using hipsparseXhybmv\n"
    "                  with an automatic partition\n"
    "hipsparse_coo: hipSPARSE CSR SpMV using hipsparseXhybmv\n"
    "               with HIPSPARSE_HYB_PARTITION_USER\n"
    "hipsparse_ell: hipSPARSE CSR SpMV using hipsparseXhybmv\n"
    "               with HIPSPARSE_HYB_PARTITION_MAX\n"
#endif  // HAS_HIP
#ifdef HAS_DPCPP
    "onemkl_csr: oneMKL Csr SpMV\n"
    "onemkl_optimized_csr: oneMKL optimized Csr SpMV using optimize_gemv after "
    "reading the matrix"
#endif  // HAS_DPCPP
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
 * Creates a Ginkgo sparselib matrix from the intermediate data representation
 * format gko::matrix_data.
 *
 * @param exec  the executor where the matrix will be put
 * @param data  the data represented in the intermediate representation format
 *
 * @tparam MatrixTagType  the tag type for the matrix format, see
 *                        sparselib_linops.hpp
 *
 * @return a `unique_pointer` to the created matrix
 */
template <typename MatrixTagType>
std::unique_ptr<gko::LinOp> read_splib_matrix_from_data(
    std::shared_ptr<const gko::Executor> exec,
    const gko::matrix_data<etype, itype>& data)
{
    auto mat = create_sparselib_linop<MatrixTagType>(std::move(exec));
    gko::as<gko::ReadableFromMatrixData<etype, itype>>(mat.get())->read(data);
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
         read_batch_matrix_from_batch_data<gko::matrix::BatchDense<etype>>},
        {"batch_diagonal",
         read_batch_matrix_from_batch_data<gko::matrix::BatchDiagonal<etype>>}};


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
        {"csrs", READ_MATRIX(csr, std::make_shared<csr::sparselib>())},
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
        {"cusparse_csr", read_splib_matrix_from_data<cusparse_csr>},
        {"cusparse_csrmp", read_splib_matrix_from_data<cusparse_csrmp>},
        {"cusparse_csrmm", read_splib_matrix_from_data<cusparse_csrmm>},
        {"cusparse_hybrid", read_splib_matrix_from_data<cusparse_hybrid>},
        {"cusparse_coo", read_splib_matrix_from_data<cusparse_coo>},
        {"cusparse_ell", read_splib_matrix_from_data<cusparse_ell>},
        {"cusparse_csr", read_splib_matrix_from_data<cusparse_gcsr>},
        {"cusparse_coo", read_splib_matrix_from_data<cusparse_gcoo>},
        {"cusparse_csrex", read_splib_matrix_from_data<cusparse_csrex>},
        {"cusparse_gcsr", read_splib_matrix_from_data<cusparse_gcsr>},
        {"cusparse_gcsr2", read_splib_matrix_from_data<cusparse_gcsr2>},
        {"cusparse_gcoo", read_splib_matrix_from_data<cusparse_gcoo>},
#endif  // HAS_CUDA
#ifdef HAS_HIP
        {"hipsparse_csr", read_splib_matrix_from_data<hipsparse_csr>},
        {"hipsparse_csrmm", read_splib_matrix_from_data<hipsparse_csrmm>},
        {"hipsparse_hybrid", read_splib_matrix_from_data<hipsparse_hybrid>},
        {"hipsparse_coo", read_splib_matrix_from_data<hipsparse_coo>},
        {"hipsparse_ell", read_splib_matrix_from_data<hipsparse_ell>},
#endif  // HAS_HIP
#ifdef HAS_DPCPP
        {"onemkl_csr", read_splib_matrix_from_data<onemkl_csr>},
        {"onemkl_optimized_csr", read_splib_matrix_from_data<onemkl_optimized_csr>},
#endif  // HAS_DPCPP
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
