// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bitvector_kernels.hpp"

#include "core/base/index_range.hpp"
#include "core/base/intrinsics.hpp"


namespace gko {
namespace kernels {
namespace reference {
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
    std::fill_n(bits, num_blocks, 0u);
    for (auto i : irange{num_indices}) {
        const auto index = indices[i];
        assert(index >= 0);
        assert(index < size);
        bits[index / bv::block_size] |= storage_type{1}
                                        << index % bv::block_size;
    }
    IndexType rank{};
    for (auto i : irange{num_blocks}) {
        ranks[i] = rank;
        rank += gko::detail::popcount(bits[i]);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BITVECTOR_COMPUTE_BITS_AND_RANKS_KERNEL);


}  // namespace bitvector
}  // namespace reference
}  // namespace kernels
}  // namespace gko
