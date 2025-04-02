// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_BITVECTOR_HPP_
#define GKO_CORE_COMPONENTS_BITVECTOR_HPP_

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/intrinsics.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>

namespace gko {


template <typename IndexType>
class device_bitvector {
public:
    using index_type = IndexType;
    using storage_type = uint32;
    constexpr static int block_size = CHAR_BIT * sizeof(storage_type);

    constexpr device_bitvector(const storage_type* bits,
                               const index_type* ranks, index_type size)
        : bits_{bits}, ranks_{ranks}, size_{size}
    {}

    constexpr index_type size() const { return size_; }

    constexpr index_type num_blocks() const
    {
        return (size() + block_size - 1) / block_size;
    }

    constexpr bool get(index_type i) const
    {
        assert(i >= 0);
        assert(i < size());
        const auto block = i / block_size;
        const auto local = i % block_size;
        return bool((bits_[block] >> local) & 1);
    }

    constexpr index_type rank(index_type i) const
    {
        assert(i >= 0);
        assert(i < size());
        const auto block = i / block_size;
        const auto local = i % block_size;
        const auto prefix_mask = (storage_type{1} << local) - 1;
        return ranks_[block] + detail::popcount(prefix_mask & bits_[block]);
    }

private:
    const index_type* ranks_;
    const storage_type* bits_;
    index_type size_;
};


/**
 * Bitvector with rank support. It supports bit queries (whether a bit is set)
 * and rank queries (how many bits are set before a specific index).
 *
 * @tparam IndexType  the type of indices used in the input and rank array.
 */
template <typename IndexType>
class bitvector {
public:
    using index_type = IndexType;
    using storage_type = uint32;
    constexpr static int block_size = CHAR_BIT * sizeof(storage_type);

    device_bitvector<index_type> device_view() const;

    static bitvector from_sorted_indices(const array<index_type>& indices,
                                         index_type size);

    std::shared_ptr<const Executor> get_executor() const;

    const storage_type* get_bits() const;

    const index_type* get_ranks() const;

    index_type get_size() const;

    index_type get_num_blocks() const;

    bitvector(array<storage_type> bits, array<index_type> ranks,
              index_type size);

private:
    bitvector(std::shared_ptr<const Executor> exec, index_type size);

    index_type size_;
    array<storage_type> bits_;
    array<index_type> ranks_;
};


}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_BITVECTOR_HPP_
