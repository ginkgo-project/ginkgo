// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sellp_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/unified/base/arithmetic_accessor.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The SELL-P matrix format namespace.
 *
 */
namespace sellp {


constexpr int default_block_size = 512;


template <typename IndexType, typename ValueAccessor, typename BAccessor,
          typename CAccessor>
__global__ __launch_bounds__(default_block_size) void spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type slice_size,
    const size_type* __restrict__ slice_sets, ValueAccessor a,
    const IndexType* __restrict__ cols, BAccessor b, CAccessor c)
{
    const auto row = thread::get_thread_id_flat();
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = blockIdx.y;
    auto val = zero<arithmetic_type_t<CAccessor>>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b(col, column_id);
            }
        }
        c(row, column_id) = val;
    }
}


template <typename IndexType, typename ValueAccessor, typename BAccessor,
          typename CAccessor>
__global__ __launch_bounds__(default_block_size) void advanced_spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type slice_size,
    const size_type* __restrict__ slice_sets, ValueAccessor alpha,
    ValueAccessor a, const IndexType* __restrict__ cols, BAccessor b,
    ValueAccessor beta, CAccessor c)
{
    const auto row = thread::get_thread_id_flat();
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = blockIdx.y;
    auto val = zero<arithmetic_type_t<CAccessor>>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b(col, column_id);
            }
        }
        c(row, column_id) = is_zero(beta[0])
                                ? alpha[0] * val
                                : beta[0] * c(row, column_id) + alpha[0] * val;
    }
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::sellp<const ValueType, const IndexType> a,
          matrix::view::dense<const ValueType> b,
          matrix::view::dense<ValueType> c)
{
    const auto block_size = default_block_size;
    const dim3 grid(ceildiv(a.size[0], block_size), b.size[1]);

    const auto a_vals =
        arithmetic<ValueType>(a.values, a.slice_size * a.total_cols);
    const auto b_vals = arithmetic<ValueType>(b);
    const auto c_vals = arithmetic<ValueType>(c);
    if (grid.x > 0 && grid.y > 0) {
        spmv_kernel<<<grid, block_size, 0, exec->get_stream()>>>(
            a.size[0], b.size[1], a.slice_size, a.slice_sets,
            acc::as_device_range(a_vals), a.col_idxs,
            acc::as_device_range(b_vals), acc::as_device_range(c_vals));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::sellp<const ValueType, const IndexType> a,
                   matrix::view::dense<const ValueType> b,
                   matrix::view::dense<const ValueType> beta,
                   matrix::view::dense<ValueType> c)
{
    const auto block_size = default_block_size;
    const dim3 grid(ceildiv(a.size[0], block_size), b.size[1]);

    const auto alpha_val = arithmetic<ValueType, 1>(alpha);
    const auto a_vals =
        arithmetic<ValueType>(a.values, a.slice_size * a.total_cols);
    const auto b_vals = arithmetic<ValueType>(b);
    const auto beta_val = arithmetic<ValueType, 1>(beta);
    const auto c_vals = arithmetic<ValueType>(c);
    if (grid.x > 0 && grid.y > 0) {
        advanced_spmv_kernel<<<grid, block_size, 0, exec->get_stream()>>>(
            a.size[0], b.size[1], a.slice_size, a.slice_sets,
            acc::as_device_range(alpha_val), acc::as_device_range(a_vals),
            a.col_idxs, acc::as_device_range(b_vals),
            acc::as_device_range(beta_val), acc::as_device_range(c_vals));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


}  // namespace sellp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
