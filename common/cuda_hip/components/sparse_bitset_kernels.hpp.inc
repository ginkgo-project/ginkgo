/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "ginkgo/core/base/exception_helpers.hpp"
template <typename GlobalIndexType>
void sort(std::shared_ptr<const DefaultExecutor> exec, GlobalIndexType* indices,
          size_type size)
{
    auto index_it = thrust::device_pointer_cast(indices);
    thrust::sort(thrust::device, index_it, index_it + size);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_SPARSE_BITSET_SORT_KERNEL);


namespace kernel {


template <typename GlobalIndexType>
__global__ __launch_bounds__(default_block_size) void build_bitmap(
    const GlobalIndexType* indices, size_type size, uint32* bitmap)
{
    // how many chunks of buffer between warps do we need to avoid conflicts?
    constexpr auto atomic_buffer = config::warp_size / sparse_bitset_word_size;
    const auto num_chunks = size / config::warp_size;
    const auto num_warps = gridDim.x * blockDim.x / config::warp_size;
    const auto chunks_per_warp = ceildiv(num_chunks, num_warps);
    const auto warp_id = thread::get_subwarp_id_flat<config::warp_size>();
    const auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    const auto lane = warp.thread_rank();
    const auto chunks_begin = warp_id * chunks_per_warp;
    const auto chunks_end = (warp_id + 1) * chunks_per_warp;
    for (auto chunk = chunks_begin; chunk < chunks_end; chunk++) {
        const auto entry = chunk * config::warp_size + lane;
        const auto index = entry < size
                               ? indices[entry]
                               : device_numeric_limits<GlobalIndexType>::max;
        const auto bitmap_idx = index / sparse_bitset_word_size;
        const auto local_idx = index % sparse_bitset_word_size;
        auto mask = uint32{entry < size ? 1u : 0u} << local_idx;
        const auto is_first =
            // TODO fix with #994
            segment_scan(warp, local_idx, &mask);
        if (is_first && entry < size) {
            atomicOr(bitmap + bitmap_idx, mask);
        }
    }
}


}  // namespace kernel


template <typename GlobalIndexType>
void build_bitmap(std::shared_ptr<const DefaultExecutor> exec,
                  const GlobalIndexType* indices, size_type size,
                  uint32* bitmap, size_type num_blocks)
{
    auto bitmap_it = thrust::device_pointer_cast(bitmap);
    thrust::fill_n(thrust::device, bitmap_it, num_blocks, uint32{});
    if (size > 0) {
        const auto num_thread_blocks = std::min<size_type>(
            exec->get_num_multiprocessor() * 2,
            ceildiv(size, default_block_size / config::warp_size));
        kernel::build_bitmap<<<num_thread_blocks, default_block_size>>>(
            indices, size, bitmap);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_KERNEL);


template <typename LocalIndexType>
void build_bitmap_ranks(std::shared_ptr<const DefaultExecutor> exec,
                        const uint32* bitmap, size_type num_blocks,
                        LocalIndexType* ranks)
{
    auto bitmap_popcnt_it = thrust::make_transform_iterator(
        thrust::device_pointer_cast(bitmap),
        [] __device__(uint32 mask) { return popcnt(mask); });
    auto ranks_it = thrust::device_pointer_cast(ranks);
    thrust::exclusive_scan(thrust::device, bitmap_popcnt_it,
                           bitmap_popcnt_it + num_blocks, ranks_it,
                           LocalIndexType{});
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SPARSE_BITSET_BUILD_BITMAP_RANKS_KERNEL);


template <typename LocalIndexType, typename GlobalIndexType>
void build_multilevel(std::shared_ptr<const DefaultExecutor> exec,
                      const GlobalIndexType* values, size_type size,
                      array<uint32>& bitmaps, array<LocalIndexType>& ranks,
                      int depth, GlobalIndexType* offsets)
{
    /*if (size == 0) {
        // TODO
        return;
    }
    const auto idx_ptr = thrust::device_pointer_cast(values);
    const auto idx_zip_ptr = thrust::make_zip_iterator(idx_ptr, idx_ptr + 1);
    offsets = thrust::transform_reduce(
        thrust::device, idx_zip_ptr, idx_zip_ptr + (size - 1), [](auto pair) {},
        [](auto a, auto b) {});
    for (auto& offset : offsets) {
        offset += 1;
    }
    // This runs on the CPU, I use it because C++14 doesn't have this algorithm
    thrust::exclusive_scan(offsets.begin(), offsets.end(), GlobalIndexType{});
    for (int i = 0; i < depth; i++) {
        thrust::reduce_by_key(...)
        thrust::exclusive_scan(...)
        thrust::unique_copy(...)
    }*/
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SPARSE_BITSET_BUILD_MULTILEVEL_KERNEL);
