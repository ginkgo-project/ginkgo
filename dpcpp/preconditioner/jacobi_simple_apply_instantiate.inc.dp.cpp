// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/base/extended_float.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/warp_blas.dp.hpp"
#include "dpcpp/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {


namespace kernel {


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void apply(
    const ValueType* __restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    const ValueType* __restrict__ b, int32 b_stride, ValueType* __restrict__ x,
    int32 x_stride, sycl::nd_item<3> item_ct1)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>(item_ct1);
    const auto subwarp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    multiply_vec<max_block_size>(
        subwarp, block_size, v,
        blocks + storage_scheme.get_global_block_offset(block_id) +
            subwarp.thread_rank(),
        storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
        x_stride,
        [](ValueType& result, const ValueType& out) { result = out; });
}

template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void apply(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const ValueType* blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const IndexType* block_ptrs, size_type num_blocks, const ValueType* b,
    int32 b_stride, ValueType* x, int32 x_stride)
{
    queue->parallel_for(
        sycl_nd_range(grid, block),
        [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(subwarp_size)]] {
                apply<max_block_size, subwarp_size, warps_per_block>(
                    blocks, storage_scheme, block_ptrs, num_blocks, b, b_stride,
                    x, x_stride, item_ct1);
            });
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void adaptive_apply(
    const ValueType* __restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction* __restrict__ block_precisions,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    const ValueType* __restrict__ b, int32 b_stride, ValueType* __restrict__ x,
    int32 x_stride, sycl::nd_item<3> item_ct1)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>(item_ct1);
    const auto subwarp = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
        ValueType, block_precisions[block_id],
        multiply_vec<max_block_size>(
            subwarp, block_size, v,
            reinterpret_cast<const resolved_precision*>(
                blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id) +
                subwarp.thread_rank(),
            storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
            x_stride,
            [](ValueType& result, const ValueType& out) { result = out; }));
}

template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
void adaptive_apply(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const ValueType* blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction* block_precisions, const IndexType* block_ptrs,
    size_type num_blocks, const ValueType* b, int32 b_stride, ValueType* x,
    int32 x_stride)
{
    queue->parallel_for(
        sycl_nd_range(grid, block),
        [=](sycl::nd_item<3> item_ct1)
            [[sycl::reqd_sub_group_size(subwarp_size)]] {
                adaptive_apply<max_block_size, subwarp_size, warps_per_block>(
                    blocks, storage_scheme, block_precisions, block_ptrs,
                    num_blocks, b, b_stride, x, x_stride, item_ct1);
            });
}


}  // namespace kernel


// clang-format off
#cmakedefine GKO_JACOBI_BLOCK_SIZE @GKO_JACOBI_BLOCK_SIZE@
// clang-format on
// make things easier for IDEs
#ifndef GKO_JACOBI_BLOCK_SIZE
#define GKO_JACOBI_BLOCK_SIZE 1
#endif


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void apply(syn::value_list<int, max_block_size>,
           std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
           const precision_reduction* block_precisions,
           const IndexType* block_pointers, const ValueType* blocks,
           const preconditioner::block_interleaved_storage_scheme<IndexType>&
               storage_scheme,
           const ValueType* b, size_type b_stride, ValueType* x,
           size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const dim3 grid_size(ceildiv(num_blocks, warps_per_block * blocks_per_warp),
                         1, 1);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (block_precisions) {
        kernel::adaptive_apply<max_block_size, subwarp_size, warps_per_block>(
            grid_size, block_size, 0, exec->get_queue(), blocks, storage_scheme,
            block_precisions, block_pointers, num_blocks, b, b_stride, x,
            x_stride);
    } else {
        kernel::apply<max_block_size, subwarp_size, warps_per_block>(
            grid_size, block_size, 0, exec->get_queue(), blocks, storage_scheme,
            block_pointers, num_blocks, b, b_stride, x, x_stride);
    }
}


#define DECLARE_JACOBI_SIMPLE_APPLY_INSTANTIATION(ValueType, IndexType)       \
    void apply<config::min_warps_per_block, GKO_JACOBI_BLOCK_SIZE, ValueType, \
               IndexType>(                                                    \
        syn::value_list<int, GKO_JACOBI_BLOCK_SIZE>,                          \
        std::shared_ptr<const DefaultExecutor>, size_type,                    \
        const precision_reduction*, const IndexType*, const ValueType*,       \
        const preconditioner::block_interleaved_storage_scheme<IndexType>&,   \
        const ValueType*, size_type, ValueType*, size_type)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    DECLARE_JACOBI_SIMPLE_APPLY_INSTANTIATION);


}  // namespace jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
