// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include "ginkgo/core/base/exception_helpers.hpp"
#include "ginkgo/core/base/types.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Gauss Seidel solver namespace.
 *
 * @ingroup gssdl
 */
namespace gssdl {


template <typename ValueType, typename IndexType>
void multicolor_fgs_ell(std::shared_ptr<const OmpExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::Ell<ValueType, IndexType>* const a,
                        const matrix::Dense<ValueType>* const b,
                        matrix::Dense<ValueType>* const x,
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

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];

#pragma omp parallel for
        for (IndexType row = row_begin; row < row_end; ++row) {
            for (size_type irhs = 0; irhs < num_cols_rhs; ++irhs) {
                ValueType sum = b_vals[row * b_stride + irhs];
                ValueType diag = zero<ValueType>();

                for (size_type k = 0; k < nnz_per_row; ++k) {
                    const auto col = col_idxs[k * stride + row];
                    if (col == invalid) {
                        continue;
                    }
                    const auto val = values[k * stride + row];
                    if (col == row) {
                        diag = val;
                    } else {
                        sum -= val * x_vals[col * x_stride + irhs];
                    }
                }

                if (diag != zero<ValueType>()) {
                    x_vals[row * x_stride + irhs] = sum / diag;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL);


}  // namespace gssdl
}  // namespace omp
}  // namespace kernels
}  // namespace gko
