// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query_kernels.hpp"

#include <limits>

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/intrinsics.hpp"
#include "core/components/bit_packed_storage.hpp"
#include "core/components/range_minimum_query.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace range_minimum_query {


template <typename IndexType>
void compute_lookup_small(std::shared_ptr<const DefaultExecutor> exec,
                          const IndexType* values, IndexType size,
                          bit_packed_span<int, IndexType, uint32>& block_argmin,
                          IndexType* block_min, uint16* block_tree_index)
{
#ifdef GKO_COMPILING_DPCPP
    // The Intel SYCL compiler doesn't support constexpr initialization of
    // non-trivial objects on the device.
    GKO_NOT_IMPLEMENTED;
#else
    using device_type = device_range_minimum_query<IndexType>;
    constexpr auto block_size = device_type::block_size;
    using tree_index_type = std::decay_t<decltype(*block_tree_index)>;
    using device_lut_type = typename device_type::block_lut_type;
    using lut_type = typename device_type::block_lut_view_type;
    static_assert(
        lut_type::num_trees <= std::numeric_limits<tree_index_type>::max(),
        "block type storage too small");
    // block_argmin stores multiple values per memory word, so we need to make
    // sure that no two different threads write to the same memory location.
    // The easiest way to do that is to have every thread handle all elements
    // that map to the same memory location.
    // The argmin inside a block is in the range [0, block_size - 1], so
    // it needs ceil_log2_constexpr(block_size) bits. For efficiency
    // reasons, we round that up to the next power of two.
    // This expression is essentially bits_per_word /
    // round_up_pow2_constexpr(ceil_log2_constexpr(block_size)), i.e. how
    // many values are stored per word.
    constexpr auto collation_width =
        1 << (std::decay_t<decltype(block_argmin)>::bits_per_word_log2 -
              ceil_log2_constexpr(ceil_log2_constexpr(block_size)));
    const device_lut_type lut{exec};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto collated_block_idx, auto values, auto block_argmin,
                      auto block_min, auto block_tree_index, auto lut,
                      auto size) {
            // we need to put this here because some compilers interpret capture
            // rules around constexpr incorrectly
            constexpr auto block_size = device_type::block_size;
            constexpr auto infinity = std::numeric_limits<IndexType>::max();
            const auto num_blocks = ceildiv(size, block_size);
            for (auto block_idx = collated_block_idx * collation_width;
                 block_idx <
                 std::min<int64>((collated_block_idx + 1) * collation_width,
                                 num_blocks);
                 block_idx++) {
                const auto i = block_idx * block_size;
                IndexType local_values[block_size];
                int argmin = 0;
#pragma unroll
                for (int local_i = 0; local_i < block_size; local_i++) {
                    // use "infinity" as sentinel for minimum computations
                    local_values[local_i] =
                        local_i + i < size ? values[local_i + i] : infinity;
                    if (local_values[local_i] < local_values[argmin]) {
                        argmin = local_i;
                    }
                }
                const auto tree_number = lut->compute_tree_index(local_values);
                const auto min = local_values[argmin];
                block_argmin.set(block_idx, argmin);
                block_min[block_idx] = min;
                block_tree_index[block_idx] =
                    static_cast<tree_index_type>(tree_number);
            }
        },
        ceildiv(ceildiv(size, block_size), collation_width), values,
        block_argmin, block_min, block_tree_index, lut.get(), size);
#endif
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_SMALL_KERNEL);


template <typename IndexType>
void compute_lookup_large(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* block_min,
    IndexType num_blocks,
    range_minimum_query_superblocks<IndexType>& superblocks)
{
#ifdef GKO_COMPILING_DPCPP
    GKO_NOT_IMPLEMENTED;
#else
    if (num_blocks < 2) {
        return;
    }
    using superblock_type = range_minimum_query_superblocks<IndexType>;
    using storage_type = typename superblock_type::storage_type;
    // we need to collate all writes that target the same memory word in a
    // single thread
    constexpr auto level0_collation_width = sizeof(storage_type) * CHAR_BIT;
    // initialize the first level of blocks
    run_kernel(
        exec,
        [] GKO_KERNEL(auto collated_i, auto block_min, auto superblocks,
                      auto num_blocks) {
            constexpr auto infinity = std::numeric_limits<IndexType>::max();
            for (auto i = collated_i * level0_collation_width;
                 i < std::min<int64>((collated_i + 1) * level0_collation_width,
                                     num_blocks);
                 i++) {
                const auto min1 = block_min[i];
                const auto min2 =
                    i + 1 < num_blocks ? block_min[i + 1] : infinity;
                // we need to use <= here to make sure ties always break to the
                // left
                superblocks.set_block_argmin(0, i, min1 <= min2 ? 0 : 1);
            }
        },
        ceildiv(num_blocks, level0_collation_width), block_min, superblocks,
        num_blocks);
    // we computed argmins for blocks of size 2, now recursively combine them.
    const auto num_levels = superblocks.num_levels();
    for (int block_level = 1; block_level < num_levels; block_level++) {
        const auto block_size =
            superblock_type::block_size_for_level(block_level);
        // we need block_level + 1 bits to represent values of size block_size
        // and round up to the next power of two
        const auto collation_width =
            level0_collation_width / round_up_pow2(block_level + 1);
        run_kernel(
            exec,
            [] GKO_KERNEL(auto collated_i, auto block_level, auto block_min,
                          auto superblocks, auto num_blocks,
                          auto collation_width) {
                const auto block_size =
                    superblock_type::block_size_for_level(block_level);
                for (auto i = collated_i * collation_width;
                     i < std::min<int64>((collated_i + 1) * collation_width,
                                         num_blocks);
                     i++) {
                    const auto i2 = i + block_size / 2;
                    const auto argmin1 =
                        i + superblocks.block_argmin(block_level - 1, i);
                    const auto argmin2 =
                        i2 < num_blocks
                            ? i2 + superblocks.block_argmin(block_level - 1, i2)
                            : argmin1;
                    const auto min1 = block_min[argmin1];
                    const auto min2 = block_min[argmin2];
                    // we need to use <= here to make sure
                    // ties always break to the left
                    superblocks.set_block_argmin(
                        block_level, i,
                        min1 <= min2 ? argmin1 - i : argmin2 - i);
                }
            },
            ceildiv(num_blocks, collation_width), block_level, block_min,
            superblocks, num_blocks, collation_width);
    }
#endif
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_LARGE_KERNEL);


}  // namespace range_minimum_query
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
