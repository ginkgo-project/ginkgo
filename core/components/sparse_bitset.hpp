/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_COMPONENTS_SPARSE_BITSET_HPP_
#define GKO_CORE_COMPONENTS_SPARSE_BITSET_HPP_


#include <bitset>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {


static constexpr int sparse_bitset_word_size = 32;


namespace detail {


template <int size, typename ValueType>
struct device_array {
    ValueType data[size];

    GKO_ATTRIBUTES GKO_INLINE ValueType& operator[](int idx)
    {
        return data[idx];
    }

    GKO_ATTRIBUTES GKO_INLINE const ValueType& operator[](int idx) const
    {
        return data[idx];
    }

    GKO_ATTRIBUTES GKO_INLINE device_array<size - 1, ValueType> pop_front()
        const
    {
        device_array<size - 1, ValueType> result;
#pragma unroll
        for (int i = 1; i < size; i++) {
            result[i - 1] = (*this)[i];
        }
        return result;
    }

    friend bool operator==(device_array a, device_array b)
    {
        return std::equal(&a.data[0], a.data + size, &b.data[0]);
    }
};


template <typename ValueType>
struct device_array<0, ValueType> {
    ValueType* data{};

    GKO_ATTRIBUTES GKO_INLINE ValueType& operator[](int idx) { return *data; }

    GKO_ATTRIBUTES GKO_INLINE const ValueType& operator[](int idx) const
    {
        return *data;
    }

    GKO_ATTRIBUTES GKO_INLINE device_array<0, ValueType> pop_front() const
    {
        return {};
    }

    friend bool operator==(device_array, device_array) { return true; }
};


template <int depth, typename LocalIndexType, typename GlobalIndexType>
struct sparse_bitset_helper {
    GKO_ATTRIBUTES GKO_INLINE static bool contains(
        GlobalIndexType i, detail::device_array<depth, GlobalIndexType> offsets,
        const uint32* bitsets)
    {
        const auto block = i / sparse_bitset_word_size;
        const auto local_i = static_cast<uint32>(i % sparse_bitset_word_size);
        const auto local_mask = uint32{1} << local_i;
        return bitsets[block + offsets[0]] & local_mask
                   ? sparse_bitset_helper<
                         depth - 1, LocalIndexType,
                         GlobalIndexType>::contains(i / sparse_bitset_word_size,
                                                    offsets.pop_front(),
                                                    bitsets)
                   : false;
    }

    GKO_ATTRIBUTES GKO_INLINE static bool rank(
        GlobalIndexType i, detail::device_array<depth, GlobalIndexType>,
        const uint32* bitmap, const LocalIndexType* ranks)
    {
        return 0;
    }
};

// base case just necessary for things to compile
template <typename LocalIndexType, typename GlobalIndexType>
struct sparse_bitset_helper<0, LocalIndexType, GlobalIndexType> {
    GKO_ATTRIBUTES GKO_INLINE static bool contains(
        GlobalIndexType i, detail::device_array<0, GlobalIndexType>,
        const uint32* bitmap)
    {
        const auto block = i / sparse_bitset_word_size;
        const auto local = i % sparse_bitset_word_size;
        return (bitmap[block] & (uint32{1} << local)) != 0;
    }

    GKO_ATTRIBUTES GKO_INLINE static LocalIndexType rank(
        GlobalIndexType i, detail::device_array<0, GlobalIndexType>,
        const uint32* bitmap, const LocalIndexType* ranks)
    {
        const auto block = i / sparse_bitset_word_size;
        const auto local = i % sparse_bitset_word_size;
        const auto prefix_mask = (uint32{1} << local) - 1u;
        return ranks[block] +
               std::bitset<sparse_bitset_word_size>(bitmap[block] & prefix_mask)
                   .count();
    }
};


}  // namespace detail


template <int depth, typename LocalIndexType, typename GlobalIndexType>
struct device_sparse_bitset {
    GKO_ATTRIBUTES GKO_INLINE bool contains(GlobalIndexType i) const
    {
        return detail::sparse_bitset_helper<depth, LocalIndexType,
                                            GlobalIndexType>::contains(i,
                                                                       offsets,
                                                                       bitmaps);
    }

    GKO_ATTRIBUTES GKO_INLINE LocalIndexType rank(GlobalIndexType i) const
    {
        return detail::sparse_bitset_helper<depth, LocalIndexType,
                                            GlobalIndexType>::rank(i, offsets,
                                                                   bitmaps,
                                                                   ranks);
    }

    detail::device_array<depth, GlobalIndexType> offsets;
    const uint32* bitmaps;
    const LocalIndexType* ranks;
};


/**
 * A sparse set of indices in a global range [0, end). Each index is
 * assigned a unique rank
 *
 * @tparam depth  the number of recursion levels to use for storing indices.
 *                Rule of thumb: Use depth big enough such that the size of the
 *                universe divided by 32^depth easily fits into device memory.
 *                Currently instantiated for depth = 0, 1, 2.
 * @tparam GlobalIndexType  the type used to represent global indices.
 * @tparam LocalIndexType  the type used to represent ranks (= local indices)
 *                         inside the set.
 */
template <int depth, typename LocalIndexType, typename GlobalIndexType>
class sparse_bitset {
public:
    /**
     * Creates a sparse bitset from the given sorted array of unique indices.
     *
     * @param data  the array of sorted indices. It must not contain duplicates.
     * @param universe_size  the size of global indices.
     */
    static sparse_bitset from_indices_sorted(const array<GlobalIndexType>& data,
                                             GlobalIndexType universe_size);

    /**
     * Creates a sparse bitset from the given, not necessarily sorted array of
     * unique indices.
     *
     * @param data  the array of indices. It must not contain duplicates.
     * @param universe_size  the size of global indices.
     */
    static sparse_bitset from_indices_unsorted(array<GlobalIndexType> data,
                                               GlobalIndexType universe_size);

    device_sparse_bitset<depth, LocalIndexType, GlobalIndexType> to_device()
        const;

private:
    sparse_bitset(std::shared_ptr<const Executor> exec);

    std::array<GlobalIndexType, depth> offsets;
    array<uint32> bitmaps;
    array<LocalIndexType> ranks;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_SPARSE_BITSET_HPP_
