// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_COMPONENTS_BITVECTOR_KERNELS_GENERIC_HPP_
#define GKO_COMMON_UNIFIED_COMPONENTS_BITVECTOR_KERNELS_GENERIC_HPP_

#include "common/unified/base/kernel_launch.hpp"
#include "core/components/bitvector.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
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
    run_kernel(
        exec,
        [] GKO_KERNEL(auto block_i, auto size, auto device_predicate, auto bits,
                      auto ranks) {
            const auto base_i = block_i * block_size;
            storage_type mask{};
            if (base_i + block_size <= size) {
                for (int local_i = 0; local_i < block_size; local_i++) {
                    const storage_type bit =
                        device_predicate(base_i + local_i) ? 1 : 0;
                    mask |= bit << local_i;
                }
            } else {
                int local_i = 0;
                for (int local_i = 0; base_i + local_i < size; local_i++) {
                    const storage_type bit =
                        device_predicate(base_i + local_i) ? 1 : 0;
                    mask |= bit << local_i;
                }
            }
            bits[block_i] = mask;
            ranks[block_i] = gko::detail::popcount(mask);
        },
        num_blocks, size, device_predicate, bits, ranks);
    components::prefix_sum_nonnegative(exec, ranks.get_data(), num_blocks);

    return gko::bitvector<IndexType>{std::move(bits), std::move(ranks), size};
}


}  // namespace bitvector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_UNIFIED_COMPONENTS_BITVECTOR_KERNELS_GENERIC_HPP_
