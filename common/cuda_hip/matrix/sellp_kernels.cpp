// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sellp_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/sparselib_bindings.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The SELL-P matrix format namespace.
 *
 * @ingroup sellp
 */
namespace sellp {


constexpr int default_block_size = 512;


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
    size_type c_stride, size_type slice_size,
    const size_type* __restrict__ slice_sets, const ValueType* __restrict__ a,
    const IndexType* __restrict__ cols, const ValueType* __restrict__ b,
    ValueType* __restrict__ c)
{
    const auto row = thread::get_thread_id_flat();
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = blockIdx.y;
    auto val = zero<ValueType>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b[col * b_stride + column_id];
            }
        }
        c[row * c_stride + column_id] = val;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void advanced_spmv_kernel(
    size_type num_rows, size_type num_right_hand_sides, size_type b_stride,
    size_type c_stride, size_type slice_size,
    const size_type* __restrict__ slice_sets,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ a,
    const IndexType* __restrict__ cols, const ValueType* __restrict__ b,
    const ValueType* __restrict__ beta, ValueType* __restrict__ c)
{
    const auto row = thread::get_thread_id_flat();
    const auto slice_id = row / slice_size;
    const auto row_in_slice = row % slice_size;
    const auto column_id = blockIdx.y;
    auto val = zero<ValueType>();
    if (row < num_rows && column_id < num_right_hand_sides) {
        for (auto i = slice_sets[slice_id]; i < slice_sets[slice_id + 1]; i++) {
            const auto ind = row_in_slice + i * slice_size;
            const auto col = cols[ind];
            if (col != invalid_index<IndexType>()) {
                val += a[ind] * b[col * b_stride + column_id];
            }
        }
        c[row * c_stride + column_id] =
            is_zero(beta[0])
                ? alpha[0] * val
                : beta[0] * c[row * c_stride + column_id] + alpha[0] * val;
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

    if (grid.x > 0 && grid.y > 0) {
        spmv_kernel<<<grid, block_size, 0, exec->get_stream()>>>(
            a.size[0], b.size[1], b.stride, c.stride, a.slice_size,
            a.slice_sets, as_device_type(a.values), a.col_idxs,
            as_device_type(b.values), as_device_type(c.values));
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

    if (grid.x > 0 && grid.y > 0) {
        advanced_spmv_kernel<<<grid, block_size, 0, exec->get_stream()>>>(
            a.size[0], b.size[1], b.stride, c.stride, a.slice_size,
            a.slice_sets, as_device_type(alpha.values),
            as_device_type(a.values), a.col_idxs, as_device_type(b.values),
            as_device_type(beta.values), as_device_type(c.values));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


}  // namespace sellp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
