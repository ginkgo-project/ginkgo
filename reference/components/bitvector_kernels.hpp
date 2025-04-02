// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_COMPONENTS_BITVECTOR_KERNELS_GENERIC_HPP_
#define GKO_REFERENCE_COMPONENTS_BITVECTOR_KERNELS_GENERIC_HPP_

#include "core/components/bitvector.hpp"

#include "core/base/index_range.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace bitvector {


template <typename IndexType, typename DevicePredicate>
gko::bitvector<IndexType> bitvector_from_predicate(
    std::shared_ptr<const DefaultExecutor> exec, IndexType size,
    DevicePredicate device_predicate)
{
    using storage_type = typename device_bitvector<IndexType>::storage_type;
    constexpr auto block_size = device_bitvector<IndexType>::block_size;
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    array<uint32> bits{exec, num_blocks};
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


}  // namespace bitvector
}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_COMPONENTS_BITVECTOR_KERNELS_GENERIC_HPP_
