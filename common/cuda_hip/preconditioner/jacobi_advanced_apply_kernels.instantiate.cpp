// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>

#include "accessor/cuda_hip_helper.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/warp_blas.hpp"
#include "common/unified/base/arithmetic_accessor.hpp"
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
          typename ValueType, typename IndexType, typename BAccessor,
          typename XAccessor>
__global__ void __launch_bounds__(warps_per_block* config::warp_size)
    advanced_apply(const ValueType* __restrict__ blocks,
                   preconditioner::block_interleaved_storage_scheme<IndexType>
                       storage_scheme,
                   const IndexType* __restrict__ block_ptrs,
                   size_type num_blocks, BAccessor alpha, BAccessor b,
                   int32 b_stride, XAccessor x, int32 x_stride)
{
    using arithmetic_type = arithmetic_type_t<XAccessor>;
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    arithmetic_type v = zero<arithmetic_type>();
    if (subwarp.thread_rank() < block_size) {
        v = alpha[0] *
            b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    // the result of this thread, which it then adds to x
    arithmetic_type result{};
    multiply_vec<max_block_size>(
        subwarp, block_size, v,
        blocks + storage_scheme.get_global_block_offset(block_id) +
            subwarp.thread_rank(),
        storage_scheme.get_stride(), &result, 0,
        [](arithmetic_type& res, const arithmetic_type& out) { res = out; });
    if (subwarp.thread_rank() < block_size) {
        x[(block_ptrs[block_id] + subwarp.thread_rank()) * x_stride] += result;
    }
}


template <int max_block_size, int subwarp_size, int warps_per_block,
          typename ValueType, typename IndexType, typename BAccessor,
          typename XAccessor>
__global__ void
__launch_bounds__(warps_per_block* config::warp_size) advanced_adaptive_apply(
    const ValueType* __restrict__ blocks,
    preconditioner::block_interleaved_storage_scheme<IndexType> storage_scheme,
    const precision_reduction* __restrict__ block_precisions,
    const IndexType* __restrict__ block_ptrs, size_type num_blocks,
    BAccessor alpha, BAccessor b, int32 b_stride, XAccessor x, int32 x_stride)
{
    using arithmetic_type = arithmetic_type_t<XAccessor>;
    const auto block_id =
        thread::get_subwarp_id<subwarp_size, warps_per_block>();
    const auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    if (block_id >= num_blocks) {
        return;
    }
    const auto block_size = block_ptrs[block_id + 1] - block_ptrs[block_id];
    arithmetic_type v = zero<arithmetic_type>();
    if (subwarp.thread_rank() < block_size) {
        v = alpha[0] *
            b[(block_ptrs[block_id] + subwarp.thread_rank()) * b_stride];
    }
    // the result of this thread, which it then adds to x
    arithmetic_type result{};
    GKO_PRECONDITIONER_JACOBI_RESOLVE_PRECISION(
        ValueType, block_precisions[block_id],
        multiply_vec<max_block_size>(
            subwarp, block_size, v,
            reinterpret_cast<const device_type<resolved_precision>*>(
                blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id) +
                subwarp.thread_rank(),
            storage_scheme.get_stride(), &result, 0,
            [](arithmetic_type& res, const arithmetic_type& out) {
                res = out;
            }));
    if (subwarp.thread_rank() < block_size) {
        x[(block_ptrs[block_id] + subwarp.thread_rank()) * x_stride] += result;
    }
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
    const ValueType* alpha, matrix::view::dense<const ValueType> b,
    matrix::view::dense<ValueType> x)
{
    constexpr int subwarp_size = get_larger_power(max_block_size);
    constexpr int blocks_per_warp = config::warp_size / subwarp_size;
    const auto grid_size =
        ceildiv(num_blocks, warps_per_block * blocks_per_warp);
    const dim3 block_size(subwarp_size, blocks_per_warp, warps_per_block);
    const auto alpha_val = arithmetic<ValueType>(alpha, 1);
    const auto b_vals = arithmetic<ValueType, 1>(b);
    const auto x_vals = arithmetic<ValueType, 1>(x);

    if (grid_size > 0) {
        if (block_precisions) {
            kernel::advanced_adaptive_apply<max_block_size, subwarp_size,
                                            warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_precisions,
                    block_pointers, num_blocks, acc::as_device_range(alpha_val),
                    acc::as_device_range(b_vals), b.stride,
                    acc::as_device_range(x_vals), x.stride);
        } else {
            kernel::advanced_apply<max_block_size, subwarp_size,
                                   warps_per_block>
                <<<grid_size, block_size, 0, exec->get_stream()>>>(
                    as_device_type(blocks), storage_scheme, block_pointers,
                    num_blocks, acc::as_device_range(alpha_val),
                    acc::as_device_range(b_vals), b.stride,
                    acc::as_device_range(x_vals), x.stride);
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
        const ValueType*, matrix::view::dense<const ValueType>,             \
        matrix::view::dense<ValueType>)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    DECLARE_JACOBI_ADVANCED_APPLY_INSTANTIATION);


}  // namespace jacobi
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
