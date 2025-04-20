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

    /**
     * Constructs a device_bitvector from its underlying data.
     *
     * @param bits  the bitmask array
     * @param rank  the rank array, it must be the prefix sum over
     *              `popcount(bits[i])`.
     * @param size  the number of bits we consider part of this bitvector.
     */
    constexpr device_bitvector(const storage_type* bits,
                               const index_type* ranks, index_type size)
        : size_{size}, bits_{bits}, ranks_{ranks}
    {}

    /** Returns the number of bits stored in this bitvector. */
    constexpr index_type size() const { return size_; }

    /** Returns the number of words (of type storage_type) in this bitvector. */
    constexpr index_type num_blocks() const
    {
        return (this->size() + block_size - 1) / block_size;
    }

    /**
     * Returns whether the bit at the given index is set.
     *
     * @param i  the index in range [0, size())
     * @return true if the bit is set, false otherwise.
     */
    constexpr bool get(index_type i) const
    {
        assert(i >= 0);
        assert(i < size());
        const auto block = i / block_size;
        const auto local = i % block_size;
        return bool((bits_[block] >> local) & 1);
    }

    /**
     * Returns the rank of the given index.
     *
     * @param i  the index in range [0, size())
     * @return the rank of the given index, i.e. the number of 1 bits set
     *         before the corresponding bit (exclusive).
     */
    constexpr index_type rank(index_type i) const
    {
        assert(i >= 0);
        assert(i < size());
        const auto block = i / block_size;
        const auto local = i % block_size;
        const auto prefix_mask = (storage_type{1} << local) - 1;
        return ranks_[block] + detail::popcount(prefix_mask & bits_[block]);
    }

    /**
     * Returns the inclusive rank of the given index.
     *
     * @param i  the index in range [0, size())
     * @return the rank of the given index, i.e. the number of 1 bits set
     *         up to and including the corresponding bit (inclusive).
     */
    constexpr index_type rank_inclusive(index_type i) const
    {
        assert(i >= 0);
        assert(i < size());
        const auto block = i / block_size;
        const auto local = i % block_size;
        const auto mask = storage_type{1} << local;
        const auto prefix_mask = mask - 1 | mask;
        return ranks_[block] + detail::popcount(prefix_mask & bits_[block]);
    }

private:
    index_type size_;
    const storage_type* bits_;
    const index_type* ranks_;
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
    using view_type = device_bitvector<index_type>;
    using storage_type = typename view_type::storage_type;
    constexpr static int block_size = CHAR_BIT * sizeof(storage_type);

    static index_type get_num_blocks(index_type size)
    {
        return (size + block_size - 1) / block_size;
    }

    view_type device_view() const
    {
        return view_type{this->get_bits(), this->get_ranks(), this->get_size()};
    }

    static bitvector from_sorted_indices(const array<index_type>& indices,
                                         index_type size);

    std::shared_ptr<const Executor> get_executor() const
    {
        return bits_.get_executor();
    }

    const storage_type* get_bits() const { return bits_.get_const_data(); }

    const index_type* get_ranks() const { return ranks_.get_const_data(); }

    index_type get_size() const { return size_; }

    index_type get_num_blocks() const { return get_num_blocks(get_size()); }

    bitvector(array<storage_type> bits, array<index_type> ranks,
              index_type size)
        : size_{size}, bits_{std::move(bits)}, ranks_{std::move(ranks)}
    {
        GKO_ASSERT(bits_.get_executor() == ranks_.get_executor());
        GKO_ASSERT(this->get_num_blocks() == bits_.get_size());
        GKO_ASSERT(this->get_num_blocks() == ranks_.get_size());
    }

private:
    index_type size_;
    array<storage_type> bits_;
    array<index_type> ranks_;
};


}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_BITVECTOR_HPP_
