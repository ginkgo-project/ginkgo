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
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto y = c->get_values();
    auto x = b->get_const_values();
    const auto x_stride = b->get_stride();
    const auto y_stride = c->get_stride();
    const auto nrhs = c->get_size()[1];

    // Get precision buckets' arrays
    using ScalarPtrTuple = gko::instantiation_tuple_t<
        gko::generator<gko::ptr_to_const_type>,
        typename narrow_types<MatrixValueType>::type>;
    ScalarPtrTuple xvalues;
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::precision_array<const IndexType*,
                                                             MatrixValueType>
        xcol_idxs;
    precision_array<size_type, MatrixValueType> bin_strides;
    precision_array<size_type, MatrixValueType> max_nnzs;
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        using EllType = matrix::Ell<value_type, IndexType>;
        auto ematk = dynamic_cast<const EllType*>(a->get_bin_matrix(k));
        if (!ematk) {
            GKO_NOT_SUPPORTED(ematk);
        }
        xcol_idxs[k] = ematk->get_const_col_idxs();
        bin_strides[k] = ematk->get_stride();
        max_nnzs[k] = ematk->get_num_stored_elements_per_row();
        std::get<k>(xvalues) = as_device_type(ematk->get_const_values());
    });

    const auto nrows = static_cast<int>(a->get_size()[0]);
#pragma omp parallel for
    for (int i = 0; i < nrows; i++) {
        for (int irhs = 0; irhs < nrhs; irhs++) {
            y[i * y_stride + irhs] = 0;
            gko::constexpr_for<0, q, 1>([&](auto k) {
                using value_type = typename std::tuple_element<
                    k, typename gko::amp::narrow_types<MatrixValueType>::type>::
                    type;
                // We need mult type because complex numbers of different
                // precisions don't get automatically promoted.
                using mult_type =
                    gko::highest_precision<value_type, InputValueType>;
                using highest_type =
                    gko::highest_precision<mult_type, OutputValueType>;
                const auto stride = bin_strides[k];
                auto avals = std::get<k>(xvalues);
                auto acols = xcol_idxs[k];
                const auto max_nnz = max_nnzs[k];
                if (max_nnz > 0) {
                    highest_type sum = 0;
                    for (int j = 0; j < max_nnz; j++) {
                        if (acols[i + j * stride] >= 0) {
                            sum += static_cast<highest_type>(
                                static_cast<mult_type>(avals[i + j * stride]) *
                                static_cast<mult_type>(
                                    x[acols[i + j * stride] * x_stride +
                                      irhs]));
                        }
                    }
                    y[i * y_stride + irhs] += static_cast<OutputValueType>(sum);
                }
            });
        }
    }
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
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;
    static_assert(q > 0, "Need at least 1 bin!");
    auto y = c->get_values();
    auto x = b->get_const_values();
    const auto x_stride = b->get_stride();
    const auto y_stride = c->get_stride();
    const auto nrhs = c->get_size()[1];
    const auto alph = alpha->get_const_values();
    const auto bet = beta->get_const_values();

    // Get precision buckets' arrays
    using ScalarPtrTuple = gko::instantiation_tuple_t<
        gko::generator<gko::ptr_to_const_type>,
        typename narrow_types<MatrixValueType>::type>;
    ScalarPtrTuple xvalues;
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::precision_array<const IndexType*,
                                                             MatrixValueType>
        xcol_idxs;
    precision_array<size_type, MatrixValueType> bin_strides;
    precision_array<size_type, MatrixValueType> max_nnzs;
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<MatrixValueType>::type>::type;
        using EllType = matrix::Ell<value_type, IndexType>;
        auto ematk = dynamic_cast<const EllType*>(a->get_bin_matrix(k));
        if (!ematk) {
            GKO_NOT_SUPPORTED(ematk);
        }
        xcol_idxs[k] = ematk->get_const_col_idxs();
        bin_strides[k] = ematk->get_stride();
        max_nnzs[k] = ematk->get_num_stored_elements_per_row();
        std::get<k>(xvalues) = as_device_type(ematk->get_const_values());
    });

    const auto nrows = static_cast<int>(a->get_size()[0]);
#pragma omp parallel for
    for (int i = 0; i < nrows; i++) {
        for (int irhs = 0; irhs < nrhs; irhs++) {
            y[i * y_stride + irhs] = bet[0] * y[i * y_stride + irhs];
            gko::constexpr_for<0, q, 1>([&](auto k) {
                using value_type = typename std::tuple_element<
                    k, typename gko::amp::narrow_types<MatrixValueType>::type>::
                    type;
                // We need mult type because complex numbers of different
                // precisions don't get automatically promoted.
                using mult_type =
                    gko::highest_precision<value_type, InputValueType>;
                using highest_type =
                    gko::highest_precision<mult_type, OutputValueType>;
                const auto stride = bin_strides[k];
                auto avals = std::get<k>(xvalues);
                auto acols = xcol_idxs[k];
                const auto max_nnz = max_nnzs[k];
                if (max_nnz > 0) {
                    highest_type sum = 0;
                    for (int j = 0; j < max_nnz; j++) {
                        const auto col = acols[i + j * stride];
                        if (col >= 0) {
                            sum += static_cast<highest_type>(
                                static_cast<mult_type>(avals[i + j * stride]) *
                                static_cast<mult_type>(
                                    x[col * x_stride + irhs]));
                        }
                    }
                    y[i * y_stride + irhs] += static_cast<OutputValueType>(
                        static_cast<highest_type>(alph[0]) * sum);
                }
            });
        }
    }
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
