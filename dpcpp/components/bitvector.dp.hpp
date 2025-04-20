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
            const auto block = value / block_size;
            const auto local = value % block_size;
            sycl::atomic_ref<storage_type, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic(bits[block]);
            atomic.fetch_or(storage_type{1} << local);
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
