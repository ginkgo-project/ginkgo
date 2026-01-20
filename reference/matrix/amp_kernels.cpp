// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <iostream>

#include <ginkgo/core/base/amp_types.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/amp_utils.hpp"
#include "core/base/mixed_precision_types.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The AMP matrix format namespace.
 * @ref Amp
 * @ingroup amp
 */
namespace amp {


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
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
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
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


// Get the lower bound of precision bin index prec_idx.
template <typename RealType, int prec_idx>
inline float get_bin_lower_bound(const RealType row_norm, const float tolerance)
{
    using narrow_types = typename gko::amp::narrow_types<RealType>::type;
    using next_type =
        typename std::tuple_element<prec_idx + 1, narrow_types>::type;
    return static_cast<float>(tolerance * row_norm /
                              std::numeric_limits<next_type>::epsilon());
}

// Get the lower bounds for all available precision bins.
template <typename RealType, int q, int k>
inline void get_all_bins_lower_bounds(const RealType row_norm,
                                      const float tolerance,
                                      std::array<float, q>& lbs)
{
    if constexpr (k > q - 1) {
        return;
    }
    if constexpr (k == q - 1) {
        lbs[k] = static_cast<float>(row_norm) * tolerance;
    } else {
        lbs[k] = get_bin_lower_bound<RealType, k>(row_norm, tolerance);
        get_all_bins_lower_bounds<RealType, q, k + 1>(row_norm, tolerance, lbs);
    }
}

// Return the precision bin index that a number should go to.
// Returns -1 if the number should be dropped.
template <typename RealType, int q>
inline int get_precision_bin(const std::array<float, q>& lower_bounds,
                             const RealType abs_number, const int k)
{
    if (k >= q) {
        return -1;
    }
    if (abs_number > static_cast<RealType>(lower_bounds[k])) {
        return k;
    } else {
        return get_precision_bin<RealType, q>(lower_bounds, abs_number, k + 1);
    }
}


template <typename ValueType, typename IndexType>
void generate_ell_rownorms_storage(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::array_prec<int, ValueType>& max_nnz_per_row,
    array<remove_complex<ValueType>>& rownorms)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const ValueType* const ovals = a->get_const_values();
    const IndexType* const ocolids = a->get_const_col_idxs();
    for (int k = 0; k < q; k++) {
        max_nnz_per_row[k] = 0;
    }
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
        std::array<float, q> min_bin;
        get_all_bins_lower_bounds<real_type, q, 0>(rnorm, tolerance, min_bin);

        // Get max nnz per row for each precision bin matrix
        std::array<int, q> row_nnz = {};
        for (int j = 0; j < omax_nnz; j++) {
            const int ibin = get_precision_bin<real_type, q>(
                min_bin, std::abs(ovals[j * ostride + irow]), 0);
            if (ibin >= 0) {
                row_nnz[ibin]++;
            }
        }
        for (int k = 0; k < q; k++) {
            max_nnz_per_row[k] = std::max(max_nnz_per_row[k], row_nnz[k]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL);


template <typename ValueType, typename IndexType>
void generate_ell_scatter_bins(std::shared_ptr<const ReferenceExecutor> exec,
                               const matrix::Ell<ValueType, IndexType>* const a,
                               const float tolerance,
                               gko::amp::array_prec<LinOp*, ValueType>& amat)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const ValueType* const ovals = a->get_const_values();
    const IndexType* const ocolidxs = a->get_const_col_idxs();
    for (int irow = 0; irow < nrows; irow++) {
        // Compute row's 1-norm
        auto rnorm = static_cast<real_type>(0);
        for (int j = 0; j < omax_nnz; j++) {
            if (ocolidxs[j * ostride + irow] == invalid_index<IndexType>()) {
                break;
            } else {
                rnorm += std::abs(ovals[j * ostride + irow]);
            }
        }

        // Compute lower limits of each precision bin
        std::array<float, q> min_bin;
        get_all_bins_lower_bounds<real_type, q, 0>(rnorm, tolerance, min_bin);

        using EllTuple = gko::instantiation_tuple_t<
            gko::generator_partial<gko::matrix::Ell, IndexType>,
            typename gko::amp::narrow_types<ValueType>::type>;
        using ScalarPtrTuple = gko::instantiation_tuple_t<
            gko::generator<gko::ptr_type>,
            typename gko::amp::narrow_types<ValueType>::type>;
        ScalarPtrTuple xvalues;
        gko::amp::array_prec<IndexType*, ValueType> xcol_idxs;
        gko::amp::array_prec<size_type, ValueType> bin_strides;
        std::array<int, q> ixj = {};

        // initialize bins
        gko::constexpr_for<0, q, 1>([&](auto k) {
            using EllType = typename std::tuple_element<k, EllTuple>::type;
            auto ematk = dynamic_cast<EllType*>(amat[k]);
            xcol_idxs[k] = ematk->get_col_idxs();
            bin_strides[k] = ematk->get_stride();
            std::get<k>(xvalues) = ematk->get_values();
            const auto nnz_row = ematk->get_num_stored_elements_per_row();
            const auto stride = ematk->get_stride();
            for (int j = 0; j < nnz_row; j++) {
                xcol_idxs[k][j * stride + irow] =
                    gko::invalid_index<IndexType>();
                std::get<k>(xvalues)[j * stride + irow] = 0;
            }
        });

        for (int j = 0; j < omax_nnz; j++) {
            const ptrdiff_t oloc = j * ostride + irow;
            const int ibin = get_precision_bin<real_type, q>(
                min_bin, std::abs(ovals[oloc]), 0);
            if (ibin >= 0) {
                const auto nzloc =
                    ixj[ibin] * static_cast<ptrdiff_t>(bin_strides[ibin]) +
                    irow;
                xcol_idxs[ibin][nzloc] = ocolidxs[oloc];
                assign_value_to_array_tuple<q, 0>(xvalues, ovals[oloc], ibin,
                                                  nzloc);
                ixj[ibin]++;
            }
        }
    }  // End loop over rows
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_ELL_SCATTER_BINS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::AMP<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::AMP<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_EXTRACT_DIAGONAL_KERNEL);


}  // namespace amp
}  // namespace reference
}  // namespace kernels
}  // namespace gko
