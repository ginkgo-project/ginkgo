// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/warp_blas.hpp"
#include "core/base/extended_float.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/preconditioner/jacobi_kernels.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
// generated header
#include "common/cuda_hip/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace jacobi {
namespace kernel {


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void __launch_bounds__(warps_per_block* config::warp_size)
    advanced_apply(const ValueType* __restrict__ blocks,
                   preconditioner::block_interleaved_storage_scheme<IndexType>
                       storage_scheme,
                   const IndexType* __restrict__ block_ptrs,
                   size_type num_blocks, const ValueType* __restrict__ alpha,
                   const ValueType* __restrict__ b, int32 b_stride,
                   ValueType* __restrict__ x, int32 x_stride)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = alpha[0] *
            b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    multiply_vec<max_block_size>(
        subwarp, block_size, v,
        blocks + storage_scheme.get_global_block_offset(block_id) +
            subwarp.thread_rank(),
        storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
        x_stride,
        [](ValueType& result, const ValueType& out) { result += out; });
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType>
__global__ void
__launch_bounds__(warps_per_block* config::warp_size) advanced_adaptive_apply(
    const ValueType* __restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction* __restrict__ block_precisions,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    const ValueType* __restrict__ alpha, const ValueType* __restrict__ b,
    int32 b_stride, ValueType* __restrict__ x, int32 x_stride)
{
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    auto alpha_val = alpha == nullptr ? one<ValueType>() : alpha[0];
    ValueType v = zero<ValueType>();
    if (subwarp.thread_rank() < block_size) {
        v = alpha[0] *
            b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
        ValueType, block_precisions[block_id],
        multiply_vec<max_block_size>(
            subwarp, block_size, v,
            reinterpret_cast<const device_type<resolved_precision>*>(
                blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id) +
                subwarp.thread_rank(),
            storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
            x_stride,
            [](ValueType& result, const ValueType& out) { result += out; }));
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
void advanced_apply(
    syn::value_list<int, max_block_size>,
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    const precision_reduction* block_precisions,
    const IndexType* block_pointers, const ValueType* blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    const ValueType* alpha, const ValueType* b, size_type b_stride,
    ValueType* x, size_type x_stride)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const auto grid_size =
        ceildiv(num_blocks, warps_per_block * blocks_per_warp);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);

    if (grid_size > 0) {
        if (block_precisions) {
            kernel::advanced_adaptive_apply<max_block_size, subwarp_size,
                                            warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_precisions,
                    block_pointers, num_blocks, as_device_type(alpha),
                    as_device_type(b), b_stride, as_device_type(x), x_stride);
        } else {
            kernel::advanced_apply<max_block_size, subwarp_size,
                                   warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_pointers,
                    num_blocks, as_device_type(alpha), as_device_type(b),
                    b_stride, as_device_type(x), x_stride);
        }
    }
}


#define DECLARE_JACOBI_ADVANCED_APPLY_INSTANTIATION(ValueType, IndexType)   \
    void advanced_apply<config::min_warps_per_block, GKO_JACOBI_BLOCK_SIZE, \
                        ValueType, IndexType>(                              \
        syn::value_list<int, GKO_JACOBI_BLOCK_SIZE>,                        \
        std::shared_ptr<const DefaultExecutor> exec, size_type,             \
        const precision_reduction*, const IndexType* block_pointers,        \
        const ValueType*,                                                   \
        const preconditioner::block_interleaved_storage_scheme<IndexType>&, \
        const ValueType*, const ValueType*, size_type, ValueType*, size_type)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    DECLARE_JACOBI_ADVANCED_APPLY_INSTANTIATION);


}  // namespace jacobi
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
