// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_BIT_PACKED_STORAGE_HPP_
#define GKO_CORE_COMPONENTS_BIT_PACKED_STORAGE_HPP_

#include <algorithm>
#include <limits>
#include <utility>

#include <ginkgo/core/base/intrinsics.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/index_range.hpp"

namespace gko {


/**
 * Computes the rounded-up binary logarithm of a number in constexpr fashion.
 * In performance-critical runtime contexts, use ceil_log2 instead.
 */
template <typename T>
constexpr int ceil_log2_constexpr(T value)
{
    if (value == 1) {
        return 0;
    }
    return 1 + ceil_log2_constexpr((value + 1) / 2);
}


/**
 * Computes the rounded-down binary logarithm of a number in constexpr fashion.
 * In performance-critical runtime contexts, use floor_log2 instead.
 */
template <typename T>
constexpr int floor_log2_constexpr(T value)
{
    if (value == 1) {
        return 0;
    }
    return 1 + floor_log2_constexpr(value / 2);
}


/**
 * Computes the next larger (or equal) power of two for a number in constexpr
 * fashion.
 */
template <typename T>
constexpr int round_up_pow2_constexpr(T value)
{
    return T{1} << ceil_log2_constexpr(value);
}


/**
 * Computes the next smaller (or equal) power of two for a number in constexpr
 * fashion.
 */
template <typename T>
constexpr int round_down_pow2_constexpr(T value)
{
    return T{1} << floor_log2_constexpr(value);
}


/**
 * Computes the rounded-up binary logarithm of a number.
 */
template <typename T>
constexpr int ceil_log2(T value)
{
    assert(value >= 1);
    return value > 1 ? detail::find_highest_bit(
                           static_cast<std::make_unsigned_t<T>>(value - 1)) +
                           1
                     : 0;
}


/**
 * Computes the rounded-down binary logarithm of a number.
 */
template <typename T>
constexpr int floor_log2(T value)
{
    assert(value >= 1);
    return detail::find_highest_bit(
        static_cast<std::make_unsigned_t<T>>(value));
}


/**
 * Computes the next larger (or equal) power of two for a number.
 */
template <typename T>
constexpr int round_up_pow2(T value)
{
    return T{1} << ceil_log2(value);
}


/**
 * Computes the next smaller (or equal) power of two for a number.
 */
template <typename T>
constexpr int round_down_pow2(T value)
{
    return T{1} << floor_log2(value);
}


/**
 * A compact representation of a span of unsigned integers with num_bits bits.
 * Each integer gets stored using round_up_pow2(num_bits) bits inside a WordType
 * word.
 */
template <typename IndexType, typename WordType>
class bit_packed_span {
    static_assert(std::is_unsigned_v<WordType>);

public:
    constexpr static int bits_per_word = sizeof(WordType) * CHAR_BIT;
    constexpr static int bits_per_word_log2 =
        ceil_log2_constexpr(bits_per_word);

    // the constexpr here is only signalling for CUDA,
    // since we don't have if consteval to switch between two implementations
    constexpr static int bits_per_value_log2(int num_bits)
    {
        return ceil_log2(num_bits);
    }

    constexpr static int values_per_word_log2(int num_bits)
    {
        return bits_per_word_log2 - bits_per_value_log2(num_bits);
    }

    constexpr static IndexType storage_size(IndexType size, int num_bits)
    {
        const auto shift = values_per_word_log2(num_bits);
        const auto div = WordType{1} << shift;
        return (size + div - 1) >> shift;
    }

    constexpr void set_from_zero(IndexType i, WordType value)
    {
        assert(value >= 0);
        assert(value <= mask_);
        const auto [block, shift] = get_block_and_shift(i);
        data_[block] |= value << shift;
    }

    constexpr void clear(IndexType i)
    {
        const auto [block, shift] = get_block_and_shift(i);
        data_[block] &= ~(mask_ << shift);
    }

    constexpr void set(IndexType i, WordType value)
    {
        clear(i);
        set_from_zero(i, value);
    }

    constexpr WordType get(IndexType i) const
    {
        const auto [block, shift] = get_block_and_shift(i);
        return (data_[block] >> shift) & mask_;
    }

    constexpr std::pair<int, int> get_block_and_shift(IndexType i) const
    {
        assert(i >= 0);
        assert(i < size_);
        return std::make_pair(i >> values_per_word_log2_,
                              (i & local_index_mask_) * bits_per_value_);
    }

    explicit constexpr bit_packed_span(WordType* data, int num_bits,
                                       IndexType size)
        : data_{data},
          size_{size},
          mask_{(WordType{1} << num_bits) - 1},
          bits_per_value_{round_up_pow2(num_bits)},
          values_per_word_log2_{values_per_word_log2(num_bits)},
          local_index_mask_{(1 << values_per_word_log2_) - 1}
    {
        assert(bits_per_value_ <= bits_per_word);
    }

private:
    WordType* data_;
    IndexType size_;
    WordType mask_;
    int bits_per_value_;
    int values_per_word_log2_;
    int local_index_mask_;
};


/**
 * An array of size unsigned integers stored in a compact fashion with num_bits
 * bits each.
 */
template <int num_bits, int size>
class bit_packed_array {
public:
    using word_type = uint32;
    constexpr static int bits_per_word = sizeof(word_type) * CHAR_BIT;
    constexpr static int bits_per_value = round_up_pow2_constexpr(num_bits);
    constexpr static int values_per_word = bits_per_word / bits_per_value;
    constexpr static int num_words =
        (size + values_per_word - 1) / values_per_word;
    // we need to shift by less than 32 to avoid UB
    constexpr static word_type mask =
        (word_type{2} << (bits_per_value - 1)) - word_type{1};
    static_assert(num_bits <= bits_per_word);
    static_assert(num_bits > 0);
    static_assert(size >= 0);

    constexpr bit_packed_array() : data_{} {}

    constexpr void set_from_zero(int i, int value)
    {
        assert(value >= 0);
        assert(value <= mask);
        data_[i / values_per_word] |=
            value << ((i % values_per_word) * bits_per_value);
    }

    constexpr void clear(int i)
    {
        data_[i / values_per_word] &=
            ~(mask << ((i % values_per_word) * bits_per_value));
    }

    constexpr void set(int i, int value)
    {
        clear(i);
        set_from_zero(i, value);
    }

    constexpr int get(int i) const
    {
        return (data_[i / values_per_word] >>
                (i % values_per_word) * bits_per_value) &
               mask;
    }

private:
    word_type data_[num_words];
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_BIT_PACKED_STORAGE_HPP_
