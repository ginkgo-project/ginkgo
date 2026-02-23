// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "common/unified/matrix/amp_algorithms.hpp"
#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/amp_helpers.hpp"

namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The AMP matrix format namespace.
 *
 * @ingroup amp
 */
namespace amp {


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::AMP<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_SPMV_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::AMP<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void generate_ell_rownorms_storage(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::precision_array<int, ValueType>& max_nnz_per_row,
    array<remove_complex<ValueType>>& rownorms)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    // Compute minimum representable values for each bin
    const std::array<real_type, q> min_repr =
        get_bins_min_representable<real_type>();

    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const ValueType* const ovals = a->get_const_values();
    const IndexType* const ocolids = a->get_const_col_idxs();
    for (int k = 0; k < q; k++) {
        max_nnz_per_row[k] = 0;
    }
    const auto max_nnz_ptr = &max_nnz_per_row[0];
#pragma omp parallel for reduction(max : max_nnz_ptr [0:q])
    for (int irow = 0; irow < nrows; irow++) {
        // Compute row's 1-norm
        auto rnorm = static_cast<real_type>(0);
        for (int j = 0; j < omax_nnz; j++) {
            if (ocolids[j * ostride + irow] == invalid_index<IndexType>()) {
                break;
            } else {
                rnorm += std::abs(ovals[j * ostride + irow]);
            }
        }
        rownorms.get_data()[irow] = rnorm;

        // Compute lower limits of each precision bin
        const std::array<float, q> min_bin =
            get_bins_precision_lower_bounds<real_type>(rnorm, tolerance);

        // Get max nnz per row for each precision bin matrix
        std::array<int, q> row_nnz = {};
        for (int j = 0; j < omax_nnz; j++) {
            const int ibin = get_adjusted_bin<real_type>(
                min_bin, min_repr, std::abs(ovals[j * ostride + irow]));
            if (ibin >= 0) {
                row_nnz[ibin]++;
            }
        }
        for (int k = 0; k < q; k++) {
            max_nnz_ptr[k] = std::max(max_nnz_ptr[k], row_nnz[k]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL);


}  // namespace amp
}  // namespace omp
}  // namespace kernels
}  // namespace gko
