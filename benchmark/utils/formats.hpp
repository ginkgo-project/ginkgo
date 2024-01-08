// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    "coo, csr, ell, ell_mixed, sellp, hybrid, hybrid0, hybrid25, hybrid33, "
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
    "csrc: Ginkgo's CSR implementation with automatic strategy.\n"
    "csri: Ginkgo's CSR implementation with imbalance strategy.\n"
    "csrm: Ginkgo's CSR implementation with merge_path strategy.\n"
    "csrs: Ginkgo's CSR implementation with sparselib strategy.\n"
    "ell: Ellpack format according to Bell and Garland: Efficient Sparse\n"
    "     Matrix-Vector Multiplication on CUDA.\n"
    "ell_mixed: Mixed Precision Ellpack format according to Bell and Garland:\n"
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
DEFINE_string(formats, "coo", formats::format_command.c_str());

DEFINE_int64(ell_imbalance_limit, 100,
             "Maximal storage overhead above which ELL benchmarks will be "
             "skipped. Negative values mean no limit.");


namespace formats {


// some shortcuts
using hybrid = gko::matrix::Hybrid<etype, itype>;
using csr = gko::matrix::Csr<etype, itype>;
using coo = gko::matrix::Coo<etype, itype>;
using ell = gko::matrix::Ell<etype, itype>;
using ell_mixed = gko::matrix::Ell<gko::next_precision<etype>, itype>;


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
    } else if (auto dpcpp =
                   dynamic_cast<const gko::DpcppExecutor*>(exec.get())) {
        return std::make_shared<Strategy>(dpcpp->shared_from_this());
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


template <typename MatrixType, typename... Args>
auto create_matrix_type(Args&&... args)
{
    return [=](std::shared_ptr<const gko::Executor> exec)
               -> std::unique_ptr<MatrixType> {
        return MatrixType::create(std::move(exec), args...);
    };
}


template <typename MatrixType, typename Strategy>
auto create_matrix_type_with_gpu_strategy()
{
    return [&](std::shared_ptr<const gko::Executor> exec)
               -> std::unique_ptr<MatrixType> {
        return MatrixType::create(exec, create_gpu_strategy<Strategy>(exec));
    };
}


// clang-format off
const std::map<std::string, std::function<std::unique_ptr<gko::LinOp>(
                                std::shared_ptr<const gko::Executor>)>>
    matrix_type_factory{
        {"csr", create_matrix_type_with_gpu_strategy<csr, csr::automatical>()},
        {"csri", create_matrix_type_with_gpu_strategy<csr, csr::load_balance>()},
        {"csrm", create_matrix_type<csr>(std::make_shared<csr::merge_path>())},
        {"csrc", create_matrix_type<csr>(std::make_shared<csr::classical>())},
        {"csrs", create_matrix_type<csr>(std::make_shared<csr::sparselib>())},
        {"coo", create_matrix_type<coo>()},
        {"ell", create_matrix_type<ell>()},
        {"ell_mixed", create_matrix_type<ell_mixed>()},
#ifdef HAS_CUDA
        {"cusparse_csr", create_sparselib_linop<cusparse_csr>},
        {"cusparse_csrmp", create_sparselib_linop<cusparse_csrmp>},
        {"cusparse_csrmm", create_sparselib_linop<cusparse_csrmm>},
        {"cusparse_hybrid", create_sparselib_linop<cusparse_hybrid>},
        {"cusparse_coo", create_sparselib_linop<cusparse_coo>},
        {"cusparse_ell", create_sparselib_linop<cusparse_ell>},
        {"cusparse_csr", create_sparselib_linop<cusparse_gcsr>},
        {"cusparse_coo", create_sparselib_linop<cusparse_gcoo>},
        {"cusparse_csrex", create_sparselib_linop<cusparse_csrex>},
        {"cusparse_gcsr", create_sparselib_linop<cusparse_gcsr>},
        {"cusparse_gcsr2", create_sparselib_linop<cusparse_gcsr2>},
        {"cusparse_gcoo", create_sparselib_linop<cusparse_gcoo>},
#endif  // HAS_CUDA
#ifdef HAS_HIP
        {"hipsparse_csr", create_sparselib_linop<hipsparse_csr>},
        {"hipsparse_csrmm", create_sparselib_linop<hipsparse_csrmm>},
        {"hipsparse_hybrid", create_sparselib_linop<hipsparse_hybrid>},
        {"hipsparse_coo", create_sparselib_linop<hipsparse_coo>},
        {"hipsparse_ell", create_sparselib_linop<hipsparse_ell>},
#endif  // HAS_HIP
#ifdef HAS_DPCPP
        {"onemkl_csr", create_sparselib_linop<onemkl_csr>},
        {"onemkl_optimized_csr", create_sparselib_linop<onemkl_optimized_csr>},
#endif  // HAS_DPCPP
        {"hybrid", create_matrix_type<hybrid>()},
        {"hybrid0",create_matrix_type<hybrid>( std::make_shared<hybrid::imbalance_limit>(0))},
        {"hybrid25",create_matrix_type<hybrid>( std::make_shared<hybrid::imbalance_limit>(0.25))},
        {"hybrid33",
         create_matrix_type<hybrid>(
                     std::make_shared<hybrid::imbalance_limit>(1.0 / 3.0))},
        {"hybrid40",
         create_matrix_type<hybrid>( std::make_shared<hybrid::imbalance_limit>(0.4))},
        {"hybrid60",
         create_matrix_type<hybrid>( std::make_shared<hybrid::imbalance_limit>(0.6))},
        {"hybrid80",
         create_matrix_type<hybrid>( std::make_shared<hybrid::imbalance_limit>(0.8))},
        {"hybridlimit0",
         create_matrix_type<hybrid>(
                     std::make_shared<hybrid::imbalance_bounded_limit>(0))},
        {"hybridlimit25",
         create_matrix_type<hybrid>(
                     std::make_shared<hybrid::imbalance_bounded_limit>(0.25))},
        {"hybridlimit33",
         create_matrix_type<hybrid>( std::make_shared<hybrid::imbalance_bounded_limit>(
                                 1.0 / 3.0))},
        {"hybridminstorage",
         create_matrix_type<hybrid>(
                     std::make_shared<hybrid::minimal_storage_limit>())},
        {"sellp", create_matrix_type<gko::matrix::Sellp<etype, itype>>()}
};
// clang-format on


std::unique_ptr<gko::LinOp> matrix_factory(
    const std::string& format, std::shared_ptr<const gko::Executor> exec,
    const gko::matrix_data<etype, itype>& data)
{
    auto mat = matrix_type_factory.at(format)(exec);
    if (format == "ell" || format == "ell_mixed") {
        check_ell_admissibility(data);
    }
    if (format == "ell_mixed") {
        gko::matrix_data<gko::next_precision<etype>, itype> conv_data;
        conv_data.size = data.size;
        conv_data.nonzeros.resize(data.nonzeros.size());
        auto it = conv_data.nonzeros.begin();
        for (auto& el : data.nonzeros) {
            it->row = el.row;
            it->column = el.column;
            it->value = el.value;
            ++it;
        }
        gko::as<gko::ReadableFromMatrixData<gko::next_precision<etype>, itype>>(
            mat.get())
            ->read(conv_data);
    } else {
        gko::as<gko::ReadableFromMatrixData<etype, itype>>(mat.get())->read(
            data);
    }
    return mat;
}


}  // namespace formats

#endif  // GKO_BENCHMARK_UTILS_FORMATS_HPP_
