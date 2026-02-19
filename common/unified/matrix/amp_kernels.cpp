// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/amp_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/matrix/amp_algorithms.hpp"
//#include "core/base/mixed_precision_types.hpp"
#include "core/matrix/amp_helpers.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The AMP matrix format namespace.
 *
 * @ingroup amp
 */
namespace amp {


template <typename ValueType, typename IndexType>
void generate_ell_rownorms_storage(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::precision_array<int, ValueType>& max_nnz,
    array<gko::remove_complex<ValueType>>& rownorms)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_CWISE_ELL_STEP1_KERNEL);


template <typename ValueType, typename IndexType>
void generate_ell_scatter_bins(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Ell<ValueType, IndexType>* a, const float tolerance,
    gko::amp::precision_array<LinOp*, ValueType>& amat)
{
    using DValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<ValueType>;
    using d_real_type = gko::remove_complex<DValueType>;
    constexpr int q = narrow_types<DValueType>::num_types;
    // Compute minimum representable values for each bin
    const std::array<d_real_type, q> min_repr =
        get_bins_min_representable<d_real_type>();

    const auto nrows = a->get_size()[0];
    const auto ostride = a->get_stride();
    const auto omax_nnz = a->get_num_stored_elements_per_row();
    const DValueType* const ovals = as_device_type(a->get_const_values());
    const IndexType* const ocolidxs = a->get_const_col_idxs();

    using EllTuple = gko::instantiation_tuple_t<
        gko::generator_partial<gko::matrix::Ell, IndexType>,
        typename gko::amp::narrow_types<ValueType>::type>;
    using ScalarPtrTuple =
        gko::instantiation_tuple_t<gko::generator<gko::ptr_type>,
                                   typename narrow_types<DValueType>::type>;
    ScalarPtrTuple xvalues;
    gko::kernels::GKO_DEVICE_NAMESPACE::amp::precision_array<IndexType*,
                                                             DValueType>
        xcol_idxs;
    precision_array<size_type, DValueType> bin_strides;

    auto setup_kernel = [] GKO_KERNEL(auto irow, auto k, auto nnz_row,
                                      auto stride, auto xcol_idxs,
                                      auto xvalues) {
        for (int j = 0; j < nnz_row; j++) {
            xcol_idxs[k][j * stride + irow] = gko::invalid_index<IndexType>();
            std::get<k>(xvalues)[j * stride + irow] = 0;
        }
    };

    // initialize bins
    gko::constexpr_for<0, q, 1>([&](auto k) {
        using EllType = typename std::tuple_element<k, EllTuple>::type;
        auto ematk = dynamic_cast<EllType*>(amat[k]);
        xcol_idxs[k] = ematk->get_col_idxs();
        bin_strides[k] = ematk->get_stride();
        std::get<k>(xvalues) = as_device_type(ematk->get_values());
        const auto nnz_row = ematk->get_num_stored_elements_per_row();
        const auto stride = ematk->get_stride();
        run_kernel(exec, setup_kernel, nrows, k, nnz_row, stride, xcol_idxs,
                   xvalues);
    });

    run_kernel(
        exec,
        [tolerance, min_repr, bin_strides] GKO_KERNEL(
            auto irow, auto ocolidxs, auto ovals, auto ostride, auto omax_nnz,
            auto xcol_idxs, auto xvalues) {
            // Compute row's 1-norm
            auto rnorm = static_cast<d_real_type>(0);
            for (int j = 0; j < omax_nnz; j++) {
                if (ocolidxs[j * ostride + irow] ==
                    invalid_index<IndexType>()) {
                    break;
                } else {
                    rnorm += abs(ovals[j * ostride + irow]);
                }
            }

            // Compute lower limits of each precision bin
            const std::array<float, q> min_bin =
                get_bins_precision_lower_bounds<d_real_type>(rnorm, tolerance);

            std::array<int, q> ixj = {};

            for (int j = 0; j < omax_nnz; j++) {
                const ptrdiff_t oloc = j * ostride + irow;
                const int ibin = get_adjusted_bin<d_real_type>(
                    min_bin, min_repr, abs(ovals[oloc]));
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
        },  // End loop over rows
        nrows, ocolidxs, ovals, ostride, omax_nnz, xcol_idxs, xvalues);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_GENERATE_ELL_SCATTER_BINS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::AMP<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    using DValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<ValueType>;
    using d_real_type = gko::remove_complex<DValueType>;
    constexpr int q = narrow_types<DValueType>::num_types;
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto result) {
            result(row, col) = zero<DValueType>();
        },
        result->get_size(), result);

    auto fill_kernel = [] GKO_KERNEL(auto i, auto j, const auto stride,
                                     auto vals, auto cols, auto result) {
        const auto col = cols[i + j * stride];
        if (col >= 0) {
            result(i, col) += static_cast<DValueType>(vals[i + j * stride]);
        }
    };

    // Process bin 0 (full precision)
    auto ell0 = dynamic_cast<const matrix::Ell<ValueType, IndexType>*>(
        source->get_bin_matrix(0));
    if (!ell0) {
        GKO_NOT_SUPPORTED(source->get_bin_matrix(0));
    }
    const auto nrows = source->get_size()[0];
    const auto stride0 = ell0->get_stride();
    const auto max_nnz0 = ell0->get_num_stored_elements_per_row();
    auto vals0 = ell0->get_const_values();
    auto cols0 = ell0->get_const_col_idxs();
    run_kernel(exec, fill_kernel, gko::dim<2>{nrows, max_nnz0}, stride0, vals0,
               cols0, result);

    // remaining bins
    gko::constexpr_for<1, q, 1>([&](auto k) {
        // use the host value type only to get the concrete Ell matrix.
        using bin_value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<ValueType>::type>::type;
        auto ellk = dynamic_cast<const matrix::Ell<bin_value_type, IndexType>*>(
            source->get_bin_matrix(k));
        if (!ellk) {
            GKO_NOT_SUPPORTED(source->get_bin_matrix(k));
        }
        const auto stride = ellk->get_stride();
        const auto max_nnz = ellk->get_num_stored_elements_per_row();
        auto vals = ellk->get_const_values();
        auto cols = ellk->get_const_col_idxs();
        if (max_nnz > 0) {
            run_kernel(exec, fill_kernel, gko::dim<2>{nrows, max_nnz}, stride,
                       vals, cols, result);
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::AMP<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    using DValueType =
        gko::kernels::GKO_DEVICE_NAMESPACE::device_type<ValueType>;
    using d_real_type = gko::remove_complex<DValueType>;
    constexpr int q = narrow_types<DValueType>::num_types;
    const auto diag_size = diag->get_size()[0];
    auto diag_values = diag->get_values();

    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto diag) { diag[i] = zero<DValueType>(); },
        diag_size, diag_values);

    // Process bin 0 (full precision)
    auto ell0 = dynamic_cast<const matrix::Ell<ValueType, IndexType>*>(
        orig->get_bin_matrix(0));
    if (!ell0) {
        GKO_NOT_SUPPORTED(orig->get_bin_matrix(0));
    }
    const auto nrows = orig->get_size()[0];
    const auto stride0 = ell0->get_stride();
    const auto max_nnz0 = ell0->get_num_stored_elements_per_row();
    auto vals0 = ell0->get_const_values();
    auto cols0 = ell0->get_const_col_idxs();
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, const auto stride, auto vals, auto cols,
                      auto diag) {
            const auto col = cols[i + j * stride];
            if (col == static_cast<IndexType>(i)) {
                diag[i] = static_cast<DValueType>(vals[i + j * stride]);
            }
        },
        gko::dim<2>{diag_size, max_nnz0}, stride0, vals0, cols0, diag_values);

    // remaining bins
    auto fill_kernel = [] GKO_KERNEL(auto i, auto j, const auto stride,
                                     auto vals, auto cols, auto diag) {
        const auto col = cols[i + j * stride];
        if (col == static_cast<IndexType>(i)) {
            diag[i] += static_cast<DValueType>(vals[i + j * stride]);
        }
    };
    gko::constexpr_for<1, q, 1>([&](auto k) {
        // use the host value type only to get the concrete Ell matrix.
        using bin_value_type = typename std::tuple_element<
            k, typename gko::amp::narrow_types<ValueType>::type>::type;
        auto ellk = dynamic_cast<const matrix::Ell<bin_value_type, IndexType>*>(
            orig->get_bin_matrix(k));
        if (!ellk) {
            GKO_NOT_SUPPORTED(orig->get_bin_matrix(k));
        }
        const auto stride = ellk->get_stride();
        const auto max_nnz = ellk->get_num_stored_elements_per_row();
        auto vals = ellk->get_const_values();
        auto cols = ellk->get_const_col_idxs();
        if (max_nnz > 0) {
            run_kernel(exec, fill_kernel, gko::dim<2>{diag_size, max_nnz},
                       stride, vals, cols, diag_values);
        }
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_AMP_EXTRACT_DIAGONAL_KERNEL);


}  // namespace amp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
