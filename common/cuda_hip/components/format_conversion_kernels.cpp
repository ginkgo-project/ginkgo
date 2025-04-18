// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"

#include <ginkgo/core/base/types.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/load_balancing.hpp"
#include "common/cuda_hip/components/searching.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


struct ptr_to_idx_config {
    constexpr static int threadblock_size = 512;
    constexpr static int elements_per_warp = 256;
};


namespace kernel {


template <typename RowPtrType, typename IndexType>
__global__
__launch_bounds__(ptr_to_idx_config::threadblock_size) void convert_ptrs_to_idx(
    const RowPtrType* __restrict__ ptrs, IndexType num_blocks,
    RowPtrType num_elements, IndexType* __restrict__ idxs)
{
    using cfg = ptr_to_idx_config;
    const auto warp_begin =
        thread::get_subwarp_id_flat<config::warp_size, RowPtrType>() *
        ptr_to_idx_config::elements_per_warp;
    const auto warp_end =
        min(warp_begin + cfg::elements_per_warp, num_elements);
    if (warp_begin >= num_elements) {
        return;
    }
    // warp searches for the first ptr >= tid_base
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = warp.thread_rank();
    const auto first_block = group_wide_search<IndexType>(
        0, num_blocks, warp, [&](auto i) { return ptrs[i + 1] > warp_begin; });
    const auto end_block =
        group_wide_search<IndexType>(0, num_blocks, warp, [&](auto i) {
            // this needs to be inclusive, since warp_end points past the end
            return ptrs[i + 1] >= warp_end;
        });
    assert(first_block < num_blocks);
    assert(end_block < num_blocks);
    const auto num_local_blocks = end_block - first_block + 1;
    if (num_local_blocks == 1) {
        // if the warp begins and ends in the same block, fill directly
        for (auto i = warp_begin + lane; i < warp_end; i += config::warp_size) {
            idxs[i] = first_block;
        }
    } else {
        // otherwise distribute work
        load_balance_subwarp<config::warp_size>(
            num_local_blocks,
            [&](IndexType local_block) {
                const auto block = local_block + first_block;
                const auto block_begin =
                    block == first_block ? warp_begin : ptrs[block];
                const auto block_end =
                    block == end_block ? warp_end : ptrs[block + 1];
                return block_end - block_begin;
            },
            [&](IndexType local_block, RowPtrType local_i, auto) {
                const auto block = local_block + first_block;
                const auto i = local_i + warp_begin;
                idxs[i] = block;
            });
    }
}


}  // namespace kernel


template <typename IndexType, typename RowPtrType>
void convert_ptrs_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                          const RowPtrType* ptrs, size_type num_blocks,
                          IndexType* idxs)
{
    const auto num_elements = exec->copy_val_to_host(ptrs + num_blocks);
    if (num_elements > 0) {
        const auto num_threadblocks =
            ceildiv(num_elements, ptr_to_idx_config::threadblock_size);
        kernel::convert_ptrs_to_idx<<<num_threadblocks,
                                      ptr_to_idx_config::threadblock_size, 0,
                                      exec->get_stream()>>>(
            ptrs, static_cast<IndexType>(num_blocks), num_elements, idxs);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS64);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
