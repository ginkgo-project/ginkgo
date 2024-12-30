// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query_kernels.hpp"

#include <limits>

#include "core/components/range_minimum_query.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace range_minimum_query {


template <typename IndexType>
void compute_lookup_small(std::shared_ptr<const DefaultExecutor> exec,
                          const IndexType* values, IndexType size,
                          block_argmin_storage_type& block_argmin,
                          uint16* block_type)
{
    using tree_index_type = std::decay_t<decltype(*block_type)>;
    using lut_type =
        gko::block_range_minimum_query_lookup_table<small_block_size>;
    lut_type table;
    static_assert(
        lut_type::num_trees <= std::numeric_limits<tree_index_type>::max(),
        "block type storage too small");
    for (IndexType i = 0; i < size; i += small_block_size) {
        IndexType local_values[small_block_size];
        for (int local_i = 0; local_i < small_block_size; local_i++) {
            // use "infinity" as sentinel for minimum computations
            local_values[local_i] = local_i + i < size
                                        ? values[local_i + i]
                                        : std::numeric_limits<IndexType>::max();
        }
        const auto tree_number = table.compute_tree_index(local_values);
        const auto min_it = std::min_element(values, values + small_block_size);
        const auto min_idx = static_cast<uint32>(std::distance(values, min_it));
        const auto block_idx = i / small_block_size;
        block_argmin.set(block_idx, min_idx);
        block_type[block_idx] = static_cast<tree_index_type>(tree_number);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_SMALL_KERNEL);


template <typename IndexType>
void compute_lookup_large(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* values,
    const block_argmin_storage_type& block_argmin, IndexType size,
    range_minimum_query_superblocks<IndexType>& superblocks)
{
    constexpr auto infinity = std::numeric_limits<IndexType>::max();
    const auto num_small_blocks = ceildiv(size, small_block_size);
    // initialize the first level of blocks
    for (IndexType i = 0; i < num_small_blocks; i += 2) {
        const auto min1 = values[i * small_block_size + block_argmin.get(i)];
        const auto min2 =
            i + 1 < num_small_blocks
                ? values[i * small_block_size + block_argmin.get(i)]
                : infinity;
        superblocks.set(0, i / 2, min1 < min2 ? 0 : 1);
    }
    for (IndexType block_size = 4; block_size < size; block_size *= 2) {
        const auto min1 = superblocks.get()
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_RANGE_MINIMUM_QUERY_COMPUTE_LOOKUP_LARGE_KERNEL);


}  // namespace range_minimum_query
}  // namespace reference
}  // namespace kernels
}  // namespace gko
