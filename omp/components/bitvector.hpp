// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <omp.h>

#include <ginkgo/core/base/intrinsics.hpp>

#include "core/base/index_range.hpp"
#include "core/components/bitvector.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace bitvector {


template <typename IndexType, typename DevicePredicate>
gko::bitvector<IndexType> from_predicate(
    std::shared_ptr<const DefaultExecutor> exec, IndexType size,
    DevicePredicate device_predicate)
{
    using storage_type = typename device_bitvector<IndexType>::storage_type;
    constexpr auto block_size = device_bitvector<IndexType>::block_size;
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    array<storage_type> bit_array{exec, num_blocks};
    array<IndexType> rank_array{exec, num_blocks};
    const auto bits = bit_array.get_data();
    const auto ranks = rank_array.get_data();
#pragma omp parallel for
    for (IndexType block_i = 0; block_i < num_blocks; block_i++) {
        const auto base_i = block_i * block_size;
        storage_type mask{};
        const auto local_op = [&](int local_i) {
            const storage_type bit = device_predicate(base_i + local_i) ? 1 : 0;
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
    }
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
    using bv = device_bitvector<index_type>;
    using storage_type = typename bv::storage_type;
    constexpr auto block_size = bv::block_size;
    const auto num_blocks = static_cast<size_type>(ceildiv(size, block_size));
    array<storage_type> bit_array{exec, num_blocks};
    array<index_type> rank_array{exec, num_blocks};
    const auto bits = bit_array.get_data();
    const auto ranks = rank_array.get_data();
    components::fill_array(exec, bits, num_blocks, storage_type{});
    const auto num_threads = omp_get_max_threads();
    const auto work_per_thread = ceildiv(count, num_threads);
    assert(std::is_sorted(it, it + count));
#pragma omp parallel num_threads(num_threads)
    {
        const auto tid = omp_get_thread_num();
        const auto begin = std::min<index_type>(tid * work_per_thread, count);
        const auto end = std::min<index_type>(begin + work_per_thread, count);
        if (begin < end) {
            const auto first_block = it[begin] / block_size;
            const auto last_block = it[end - 1] / block_size;
            storage_type mask{0};
            auto block = first_block;
            for (auto i : irange{begin, end}) {
                const auto value = it[i];
                const auto new_block = value / block_size;
                const auto local = value % block_size;
                if (new_block != block) {
                    assert(new_block > block);
                    if (block == first_block) {
#pragma omp atomic
                        bits[block] |= mask;
                    } else {
                        bits[block] = mask;
                    }
                    mask = 0;
                    block = new_block;
                }
                mask |= storage_type{1} << local;
            }
#pragma omp atomic
            bits[last_block] |= mask;
        }
    }
#pragma omp parallel for
    for (size_type i = 0; i < num_blocks; i++) {
        ranks[i] = gko::detail::popcount(bits[i]);
    }
    components::prefix_sum_nonnegative(exec, ranks, num_blocks);
    return gko::bitvector<index_type>{std::move(bit_array),
                                      std::move(rank_array), size};
}


}  // namespace bitvector
}  // namespace omp
}  // namespace kernels
}  // namespace gko
