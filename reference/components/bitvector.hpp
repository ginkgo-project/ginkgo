// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_COMPONENTS_BITVECTOR_HPP_
#define GKO_REFERENCE_COMPONENTS_BITVECTOR_HPP_

#include "core/base/index_range.hpp"
#include "core/components/bitvector.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace bitvector {


template <typename IndexType, typename DevicePredicate>
gko::bitvector<IndexType> from_predicate(
    std::shared_ptr<const DefaultExecutor> exec, IndexType size,
    DevicePredicate device_predicate)
{
    using storage_type = typename device_bitvector<IndexType>::storage_type;
    constexpr auto block_size = device_bitvector<IndexType>::block_size;
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    array<storage_type> bits{exec, num_blocks};
    array<IndexType> ranks{exec, num_blocks};
    std::fill_n(bits.get_data(), num_blocks, 0);
    std::fill_n(ranks.get_data(), num_blocks, 0);
    for (auto i : irange{size}) {
        if (device_predicate(i)) {
            bits.get_data()[i / block_size] |= storage_type{1}
                                               << (i % block_size);
            ranks.get_data()[i / block_size]++;
        }
    }
    components::prefix_sum_nonnegative(exec, ranks.get_data(), num_blocks);

    return gko::bitvector<IndexType>{std::move(bits), std::move(ranks), size};
}


template <typename IndexIterator>
gko::bitvector<typename std::iterator_traits<IndexIterator>::value_type>
from_sorted_indices(
    std::shared_ptr<const DefaultExecutor> exec, IndexIterator it,
    typename std::iterator_traits<IndexIterator>::difference_type count,
    typename std::iterator_traits<IndexIterator>::value_type size)
{
    using index_type = typename std::iterator_traits<IndexIterator>::value_type;
    using storage_type = typename device_bitvector<index_type>::storage_type;
    constexpr auto block_size = device_bitvector<index_type>::block_size;
    const auto num_blocks = ceildiv(size, block_size);
    array<storage_type> bits{exec, static_cast<size_type>(num_blocks)};
    array<index_type> ranks{exec, static_cast<size_type>(num_blocks)};
    std::fill_n(bits.get_data(), num_blocks, 0);
    assert(std::is_sorted(it, it + count));
    for (auto i : irange{count}) {
        const auto value = it[i];
        const auto [block, mask] =
            device_bitvector<index_type>::get_block_and_mask(value);
        assert((bits.get_data()[block] & mask) == 0);
        bits.get_data()[block] |= mask;
    }
    index_type rank{};
    for (auto i : irange{num_blocks}) {
        ranks.get_data()[i] = rank;
        rank += gko::detail::popcount(bits.get_const_data()[i]);
    }

    return gko::bitvector<index_type>{std::move(bits), std::move(ranks), size};
}


}  // namespace bitvector
}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_COMPONENTS_BITVECTOR_HPP_
