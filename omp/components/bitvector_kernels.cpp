// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector_kernels.hpp"

#include <ginkgo/core/base/intrinsics.hpp>

#include "core/base/index_range.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace bitvector {


template <typename IndexType>
void compute_bits_and_ranks(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* indices,
    IndexType num_indices, IndexType size,
    typename device_bitvector<IndexType>::storage_type* bits, IndexType* ranks)
{
    using bv = device_bitvector<IndexType>;
    using storage_type = typename bv::storage_type;
    const auto num_blocks = ceildiv(size, bv::block_size);
#pragma omp parallel for
    for (IndexType i = 0; i < num_blocks; i++) {
        bits[i] = 0;
    }
#pragma omp parallel for
    for (IndexType i = 0; i < num_indices; i++) {
        const auto index = indices[i];
        const auto block_idx = index / bv::block_size;
        const auto mask = storage_type{1} << index % bv::block_size;
#pragma omp atomic
        bits[block_idx] |= mask;
    }
#pragma omp parallel for
    for (IndexType i = 0; i < num_blocks; i++) {
        ranks[i] = gko::detail::popcount(bits[i]);
    }
    components::prefix_sum_nonnegative(exec, ranks, num_blocks);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BITVECTOR_COMPUTE_BITS_AND_RANKS_KERNEL);


}  // namespace bitvector
}  // namespace omp
}  // namespace kernels
}  // namespace gko
