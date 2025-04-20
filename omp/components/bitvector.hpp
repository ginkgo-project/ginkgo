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
        const auto begin = std::min(tid * work_per_thread, count);
        const auto end = std::min(begin + work_per_thread, count);
        if (begin < end) {
            const auto first_block = it[begin] / block_size;
            const auto last_block = it[end - 1] / block_size;
            storage_type word{};
            auto block = first_block;
            for (auto i : irange{begin, end}) {
                const auto value = it[i];
                const auto new_block = value / block_size;
                const auto local = value % block_size;
                if (new_block != block) {
                    assert(new_block > block);
                    if (block == first_block) {
#pragma omp atomic
                        bits[block] |= word;
                    } else {
                        bits[block] = word;
                    }
                    word = 0;
                    block = new_block;
                }
                word |= storage_type{1} << local;
            }
#pragma omp atomic
            bits[last_block] |= word;
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
