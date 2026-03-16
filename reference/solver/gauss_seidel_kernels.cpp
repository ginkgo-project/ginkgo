// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/mixed_precision_types.hpp"
#include "core/base/utils.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Gauss Seidel solver namespace.
 *
 * @ingroup gssdl
 */
namespace gssdl {


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_ell(std::shared_ptr<const ReferenceExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::Ell<MatrixValueType, IndexType>* const a,
                        const matrix::Dense<InputValueType>* const b,
                        matrix::Dense<OutputValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    if (first_iter) {
        for (size_type j = 0; j < stop_status->get_size(); ++j) {
            stop_status->get_data()[j].reset();
        }
    }

    if (color_ptrs.size() < 2) {
        return;
    }

    const auto num_colors = color_ptrs.size() - 1;
    const auto num_cols_rhs = b->get_size()[1];
    const auto nnz_per_row = a->get_num_stored_elements_per_row();
    const auto stride = a->get_stride();
    const auto* const col_idxs = a->get_const_col_idxs();
    const auto* const values = a->get_const_values();
    auto* const x_vals = x->get_values();
    const auto* const b_vals = b->get_const_values();
    const auto x_stride = x->get_stride();
    const auto b_stride = b->get_stride();
    constexpr auto invalid = invalid_index<IndexType>();

    using highest_type = gko::highest_precision<InputValueType, MatrixValueType,
                                                OutputValueType>;

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];

        for (IndexType row = row_begin; row < row_end; ++row) {
            for (size_type irhs = 0; irhs < num_cols_rhs; ++irhs) {
                auto sum =
                    static_cast<highest_type>(b_vals[row * b_stride + irhs]);
                auto diag = zero<MatrixValueType>();

                for (size_type k = 0; k < nnz_per_row; ++k) {
                    const auto col = col_idxs[k * stride + row];
                    if (col == invalid) {
                        continue;
                    }
                    const auto val = values[k * stride + row];
                    if (col == row) {
                        diag = val;
                    } else {
                        sum -= static_cast<highest_type>(val) *
                               static_cast<highest_type>(
                                   x_vals[col * x_stride + irhs]);
                    }
                }

                if (diag != zero<MatrixValueType>()) {
                    x_vals[row * x_stride + irhs] =
                        static_cast<OutputValueType>(
                            sum / static_cast<highest_type>(diag));
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_amp(std::shared_ptr<const ReferenceExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::AMP<MatrixValueType, IndexType>* const a,
                        const matrix::Dense<InputValueType>* const b,
                        matrix::Dense<OutputValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    if (first_iter) {
        for (size_type j = 0; j < stop_status->get_size(); ++j) {
            stop_status->get_data()[j].reset();
        }
    }

    if (color_ptrs.size() < 2) {
        return;
    }

    const auto num_colors = color_ptrs.size() - 1;
    const auto num_cols_rhs = b->get_size()[1];
    auto* const x_vals = x->get_values();
    const auto* const b_vals = b->get_const_values();
    const auto x_stride = x->get_stride();
    const auto b_stride = b->get_stride();
    constexpr auto invalid = invalid_index<IndexType>();
    constexpr int q = matrix::AMP<MatrixValueType, IndexType>::num_precisions;

    using highest_type = gko::highest_precision<InputValueType, MatrixValueType,
                                                OutputValueType>;

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];

        for (IndexType row = row_begin; row < row_end; ++row) {
            auto diag = zero<MatrixValueType>();
            for (size_type irhs = 0; irhs < num_cols_rhs; ++irhs) {
                auto sum =
                    static_cast<highest_type>(b_vals[row * b_stride + irhs]);
                gko::constexpr_for<0, q, 1>([&](auto k) {
                    using value_type = typename std::tuple_element<
                        k, typename gko::amp::narrow_types<
                               MatrixValueType>::type>::type;
                    auto ellk =
                        dynamic_cast<const matrix::Ell<value_type, IndexType>*>(
                            a->get_bin_matrix(k));
                    if (!ellk) {
                        GKO_NOT_SUPPORTED(a->get_bin_matrix(0));
                    }
                    const auto nnz_per_row =
                        ellk->get_num_stored_elements_per_row();
                    const auto stride = ellk->get_stride();
                    const auto* const col_idxs = ellk->get_const_col_idxs();
                    const auto* const values = ellk->get_const_values();
                    for (size_type k = 0; k < nnz_per_row; ++k) {
                        const auto col = col_idxs[k * stride + row];
                        if (col == invalid) {
                            continue;
                        }
                        const auto val = values[k * stride + row];
                        if (col == row) {
                            diag = static_cast<MatrixValueType>(val);
                        } else {
                            sum -= static_cast<highest_type>(val) *
                                   static_cast<highest_type>(
                                       x_vals[col * x_stride + irhs]);
                        }
                    }
                });
                if (diag != zero<MatrixValueType>()) {
                    x_vals[row * x_stride + irhs] =
                        static_cast<OutputValueType>(
                            sum / static_cast<highest_type>(diag));
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_AMP_KERNEL);


}  // namespace gssdl
}  // namespace reference
}  // namespace kernels
}  // namespace gko
