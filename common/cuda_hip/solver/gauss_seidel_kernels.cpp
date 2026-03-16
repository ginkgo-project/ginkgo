// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/types.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Gauss Seidel solver namespace.
 *
 * @ingroup gssdl
 */
namespace gssdl {

namespace gkerd = gko::kernels::GKO_DEVICE_NAMESPACE;

constexpr int default_block_size = 1024;

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void mc_fgs_ell(
    const IndexType max_nnz_rows, const size_type stride,
    const IndexType* const __restrict__ col_idxs,
    const ValueType* const __restrict__ values, const IndexType begin_row,
    const IndexType end_row, const size_type b_stride,
    const ValueType* const __restrict__ b, const size_type x_stride,
    ValueType* const __restrict__ x,
    stopping_status* const __restrict__ stopstatus, const bool first_iter)
{
    // TODO: Optimize cached loads/stores
    const auto row = begin_row + blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= end_row) {
        return;
    }
    const auto irhs = blockIdx.y;
    if (first_iter && blockIdx.x == 0 && threadIdx.x == 0) {
        stopstatus[irhs].reset();
    }
    ValueType sum = b[row * b_stride + irhs];
    ValueType diag = zero<ValueType>();

    constexpr auto invalid = invalid_index<IndexType>();

    for (IndexType k = 0; k < max_nnz_rows; ++k) {
        const auto col = col_idxs[k * stride + row];
        if (col == invalid) {
            continue;
        }
        const auto val = values[k * stride + row];
        if (col == row) {
            diag = val;
        } else {
            sum -= val * x[col * x_stride + irhs];
        }
    }

    if (diag != zero<ValueType>()) {
        x[row * x_stride + irhs] = sum / diag;
    }
}


template <typename ValueType, typename IndexType>
void multicolor_fgs_ell(std::shared_ptr<const DefaultExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::Ell<ValueType, IndexType>* const a,
                        const matrix::Dense<ValueType>* const b,
                        matrix::Dense<ValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    if (color_ptrs.size() < 2) {
        return;
    }

    using d_val_type = typename gkerd::device_type<ValueType>;
    const auto num_colors = static_cast<int>(color_ptrs.size() - 1);
    const auto num_rhs = b->get_size()[1];
    const auto nnz_per_row = a->get_num_stored_elements_per_row();
    const auto stride = a->get_stride();
    const auto* const col_idxs = a->get_const_col_idxs();
    const auto* const values = as_device_type(a->get_const_values());
    d_val_type* const x_vals = as_device_type(x->get_values());
    const auto* const b_vals = as_device_type(b->get_const_values());
    const auto x_stride = x->get_stride();
    const auto b_stride = b->get_stride();

    for (int color = 0; color < num_colors; ++color) {
        const auto row_begin = color_ptrs[color];
        const auto row_end = color_ptrs[color + 1];
        const auto nrows = row_end - row_begin;
        const dim3 nblocks{
            static_cast<uint32>(ceildiv(nrows, default_block_size)),
            static_cast<uint32>(num_rhs), 1u};
        mc_fgs_ell<d_val_type, IndexType>
            <<<nblocks, default_block_size, 0, exec->get_stream()>>>(
                nnz_per_row, stride, col_idxs, values, row_begin, row_end,
                b_stride, b_vals, x_stride, x_vals, stop_status->get_data(),
                first_iter);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL);


}  // namespace gssdl
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
