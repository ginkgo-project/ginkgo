// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_BITVECTOR_DP_HPP_
#define GKO_DPCPP_COMPONENTS_BITVECTOR_DP_HPP_


#include <sycl/sycl.hpp>

#include <ginkgo/core/base/intrinsics.hpp>

#include "core/components/bitvector.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace bitvector {


template <typename IndexType, typename DevicePredicate>
gko::bitvector<IndexType> from_predicate(
    std::shared_ptr<const DefaultExecutor> exec, IndexType size,
    DevicePredicate device_predicate)
{
    using storage_type = typename device_bitvector<IndexType>::storage_type;
    constexpr auto block_size = device_bitvector<IndexType>::block_size;
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    array<uint32> bit_array{exec, num_blocks};
    array<IndexType> rank_array{exec, num_blocks};
    const auto bits = bit_array.get_data();
    const auto ranks = rank_array.get_data();
    const auto queue = exec->get_queue();
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_blocks, [=](sycl::id<1> block_i) {
            const auto base_i = static_cast<IndexType>(block_i) * block_size;
            storage_type mask{};
            const auto local_op = [&](int local_i) {
                const storage_type bit =
                    device_predicate(base_i + local_i) ? 1 : 0;
                mask |= bit << local_i;
            };
            if (base_i + block_size <= size) {
#pragma unroll
                for (int local_i = 0; local_i < block_size; local_i++) {
                    local_op(local_i);
                }
            } else {
                for (int local_i = 0; base_i + local_i < size; local_i++) {
                    local_op(local_i);
                }
            }
            bits[block_i] = mask;
            ranks[block_i] = gko::detail::popcount(mask);
        });
    });
    components::prefix_sum_nonnegative(exec, ranks, num_blocks);

    return gko::bitvector<IndexType>{std::move(bit_array),
                                     std::move(rank_array), size};
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
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    array<storage_type> bit_array{exec, num_blocks};
    array<index_type> rank_array{exec, num_blocks};
    components::fill_array(exec, bit_array.get_data(), num_blocks,
                           storage_type{});
    const auto bits = bit_array.get_data();
    const auto ranks = rank_array.get_data();
    const auto queue = exec->get_queue();
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(count, [=](sycl::id<1> i) {
            auto value = it[i];
            const auto [block, mask] =
                device_bitvector<index_type>::get_block_and_mask(value);
            sycl::atomic_ref<storage_type, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic(bits[block]);
            atomic.fetch_or(mask);
        });
    });
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_blocks, [=](sycl::id<1> i) {
            ranks[i] = gko::detail::popcount(bits[i]);
        });
    });
    components::prefix_sum_nonnegative(exec, ranks, num_blocks);

    return gko::bitvector<index_type>{std::move(bit_array),
                                      std::move(rank_array), size};
}


}  // namespace bitvector
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_BITVECTOR_DP_HPP_
