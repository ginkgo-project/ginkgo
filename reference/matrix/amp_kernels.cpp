// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <cassert>
#include <iostream>

#include <ginkgo/core/base/amp_types.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"
#include "core/matrix/amp_algorithms.hpp"


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
          const matrix::AMP<MatrixValueType, IndexType>* const a,
          const matrix::Dense<InputValueType>* const b,
          matrix::Dense<OutputValueType>* const c)
{
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto ell0 = dynamic_cast<const matrix::Ell<MatrixValueType, IndexType>*>(
        a->get_bin_matrix(0));
    if (!ell0) {
        GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
    }
    auto y = c->get_values();
    auto x = b->get_const_values();
    const auto nrows0 = static_cast<int>(a->get_size()[0]);
    const auto stride0 = ell0->get_stride();
    auto avals = ell0->get_const_values();
    auto acols = ell0->get_const_col_idxs();
    const auto max_nnz0 = ell0->get_num_stored_elements_per_row();
    using highest_type =
        gko::highest_precision<MatrixValueType, InputValueType>;
    for (int i = 0; i < nrows0; i++) {
        y[i] = 0;
        for (int j = 0; j < max_nnz0; j++) {
            if (acols[i + j * stride0] >= 0) {
                y[i] += static_cast<OutputValueType>(
                    static_cast<highest_type>(avals[i + j * stride0]) *
                    static_cast<highest_type>(x[acols[i + j * stride0]]));
            }
        }
    }
    gko::constexpr_for<1, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        auto ellk = dynamic_cast<const matrix::Ell<value_type, IndexType>*>(
            a->get_bin_matrix(k));
        if (!ellk) {
            GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
        }
        using high_type = gko::highest_precision<value_type, InputValueType>;
        const auto nrows = static_cast<int>(a->get_size()[0]);
        assert(nrows == nrows0);
        const auto stride = ellk->get_stride();
        auto avals = ellk->get_const_values();
        auto acols = ellk->get_const_col_idxs();
        const auto max_nnz = ellk->get_num_stored_elements_per_row();
        if (max_nnz > 0) {
            for (int i = 0; i < nrows; i++) {
                for (int j = 0; j < max_nnz; j++) {
                    if (acols[i + j * stride] >= 0) {
                        y[i] += static_cast<OutputValueType>(
                            static_cast<high_type>(avals[i + j * stride]) *
                            static_cast<high_type>(x[acols[i + j * stride]]));
                    }
                }
            }
        }
    });
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


template <typename ValueType, typename IndexType>
void generate_ell_rownorms_storage(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::array_prec<int, ValueType>& max_nnz_per_row,
    array<remove_complex<ValueType>>& rownorms)
{
    using real_type = remove_complex<ValueType>;
    constexpr int q = gko::matrix::AMP<ValueType, IndexType>::num_precisions;
    // Compute minimum representable values for each bin
    const std::array<float, q> min_repr =
        gko::amp::get_bins_min_representable<real_type>();

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
        const std::array<float, q> min_bin =
            gko::amp::get_bins_precision_lower_bounds<real_type>(rnorm,
                                                                 tolerance);

        // Get max nnz per row for each precision bin matrix
        std::array<int, q> row_nnz = {};
        for (int j = 0; j < omax_nnz; j++) {
            const int ibin = gko::amp::get_adjusted_bin<real_type>(
                min_bin, min_repr, std::abs(ovals[j * ostride + irow]));
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
    // Compute minimum representable values for each bin
    const std::array<float, q> min_repr =
        gko::amp::get_bins_min_representable<real_type>();

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
        const std::array<float, q> min_bin =
            gko::amp::get_bins_precision_lower_bounds<real_type>(rnorm,
                                                                 tolerance);

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
            const int ibin = gko::amp::get_adjusted_bin<real_type>(
                min_bin, min_repr, std::abs(ovals[oloc]));
            if (ibin >= 0) {
                const auto nzloc =
                    ixj[ibin] * static_cast<ptrdiff_t>(bin_strides[ibin]) +
                    irow;
                xcol_idxs[ibin][nzloc] = ocolidxs[oloc];
                assign_value_to_array_tuple<0>(xvalues, ovals[oloc], ibin,
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
