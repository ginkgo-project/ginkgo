// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
            reinterpret_cast<const resolved_precision*>(
                blocks + storage_scheme.get_group_offset(block_id)) +
                storage_scheme.get_block_offset(block_id) +
                subwarp.thread_rank(),
            storage_scheme.get_stride(), x + block_ptrs[block_id] * x_stride,
            x_stride,
            [](ValueType& result, const ValueType& out) { result += out; }));
}


}  // namespace kernel
