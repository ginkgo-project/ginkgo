// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_BIT_PACKED_STORAGE_HPP_
#define GKO_CORE_COMPONENTS_BIT_PACKED_STORAGE_HPP_

#include <algorithm>
#include <limits>
#include <utility>

#include <ginkgo/core/base/types.hpp>

#include "core/base/intrinsics.hpp"

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
 * Computes the next larger (or equal) power of two for a number in constexpr
 * fashion.
 */
template <typename T>
constexpr int round_up_pow2_constexpr(T value)
{
    return T{1} << ceil_log2_constexpr(value);
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
 * A compact representation of an array of unsigned integers that can be
 * represented using num_bits bits, e.g. values in [0, 2^num_bits). Each integer
 * gets stored using round_up_pow2(num_bits) bits inside a StorageType word.
 * This storage layout allows for fast division-less access to values, which a
 * non-power-of-two number of bits would make hard otherwise.
 * The cass is a non-owning view.
 *
 * @tparam ValueType  the type used to represent values in the span
 * @tparam IndexType  the type used to represent indices in the span
 * @tparam StorageType  the type used to internally represent the values. It
 *                      needs to be large enough to represent all values to be
 *                      stored.
 */
template <typename ValueType, typename IndexType, typename StorageType>
class bit_packed_span {
    static_assert(std::is_unsigned_v<StorageType>);

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using storage_type = StorageType;
    /** How many bits are available in StorageType */
    constexpr static int bits_per_word = sizeof(storage_type) * CHAR_BIT;
    /** Binary logarithm of bits_per_word */
    constexpr static int bits_per_word_log2 =
        ceil_log2_constexpr(bits_per_word);

    /**
     * Returns the binary logarithm of the number of bits that should be used
     * internally to store num_bits bits. This is used to allow faster accesses
     * without the need for arbitrary integer division, since a division by
     * bits_per_value can be represented using a shift by bits_per_value_log2.
     */
    constexpr static int bits_per_value_log2(int num_bits)
    {
        return ceil_log2(num_bits);
    }

    /**
     * Returns the binary logarithm of the number of values can be stored inside
     * a single storage_type word. This gets used to avoid integer divisions in
     * favor of faster bit shifts.
     */
    constexpr static int values_per_word_log2(int num_bits)
    {
        return bits_per_word_log2 - bits_per_value_log2(num_bits);
    }

    /**
     * Computes how many storage_type words will be necessary to store size
     * values requiring num_bits bits.
     *
     * @param size  The number of values to store
     * @param num_bits  The number of bits necessary to store values inside this
     *                  span. This means that all values need to be in the range
     *                  [0, 2^num_bits).
     */
    constexpr static index_type storage_size(index_type size, int num_bits)
    {
        const auto shift = values_per_word_log2(num_bits);
        const auto div = storage_type{1} << shift;
        return (size + div - 1) >> shift;
    }

    /**
     * Sets a value inside the span, assuming that value was zero before.
     * This can be used if the underlying storage was zero-initialized.
     *
     * @param i  The index to write to
     * @param value  The value to write. It needs to be in [0, 2^num_bits).
     */
    constexpr void set_from_zero(index_type i, value_type value)
    {
        assert(value >= 0);
        assert(value <= static_cast<value_type>(mask_));
        const auto [block, shift] = get_block_and_shift(i);
        data_[block] |= static_cast<storage_type>(value) << shift;
    }

    /**
     * Clears a value inside the span, setting it to zero.
     *
     * @param i  The index to clear
     */
    constexpr void clear(index_type i)
    {
        const auto [block, shift] = get_block_and_shift(i);
        data_[block] &= ~(mask_ << shift);
    }

    /**
     * Sets a value inside the span.
     *
     * @param i  The index to write to
     * @param value  The value to write. It needs to be in [0, 2^num_bits).
     */
    constexpr void set(index_type i, value_type value)
    {
        clear(i);
        set_from_zero(i, value);
    }

    /**
     * Returns a value from the span.
     *
     * @param i  The index to read from
     * @return  the read value
     */
    constexpr value_type get(index_type i) const
    {
        const auto [block, shift] = get_block_and_shift(i);
        return static_cast<value_type>((data_[block] >> shift) & mask_);
    }

    /**
     * Construct a span from the underlying data.
     *
     * @param data  the data array of size `storage_size(size, num_bits)`.
     * @param num_bits  the number of bits necessary to store a single value.
     * @param size  the number of elements in the span.
     *
     */
    explicit constexpr bit_packed_span(storage_type* data, int num_bits,
                                       index_type size)
        : data_{data},
          size_{size},
          mask_{(storage_type{1} << num_bits) - 1},
          bits_per_value_{round_up_pow2(num_bits)},
          values_per_word_log2_{values_per_word_log2(num_bits)},
          local_index_mask_{(1 << values_per_word_log2_) - 1}
    {
        // ignore the sign bit in this comparison
        assert(num_bits < bits_per_word);
    }

private:
    constexpr std::pair<int, int> get_block_and_shift(index_type i) const
    {
        assert(i >= 0);
        assert(i < size_);
        return std::make_pair(i >> values_per_word_log2_,
                              (i & local_index_mask_) * bits_per_value_);
    }

    storage_type* data_;
    index_type size_;
    storage_type mask_;
    int bits_per_value_;
    int values_per_word_log2_;
    int local_index_mask_;
};


/**
 * An array of size unsigned integers stored in a compact fashion with num_bits
 * bits each. This is a statically-sized, owning equivalent of
 * bit_packed_span<int, uint32>.

 * @tparam num_bits  The number of bits necessary to store a single value in the
 *                   array. Values need to be in the range [0, 2^num_bits).
 * @tparam size  The number of values to store in the array.
 * @tparam StorageType  the underlying storage type to use for each individual
 word
 */
template <int num_bits, int size, typename StorageType = uint32>
class bit_packed_array {
public:
    using storage_type = StorageType;
    constexpr static int bits_per_word = sizeof(storage_type) * CHAR_BIT;
    constexpr static int bits_per_value = round_up_pow2_constexpr(num_bits);
    constexpr static int values_per_word = bits_per_word / bits_per_value;
    constexpr static int num_words =
        (size + values_per_word - 1) / values_per_word;
    // we need to shift by less than 32 to avoid UB
    constexpr static storage_type mask =
        (storage_type{2} << (bits_per_value - 1)) - storage_type{1};
    // ignore the sign bit in this comparison
    static_assert(num_bits < bits_per_word);
    static_assert(num_bits > 0);
    static_assert(size >= 0);

    /** Zero-initializes all values in the array. */
    constexpr bit_packed_array() : data_{} {}

    /** @copydoc bit_packed_span::set_from_zero */
    constexpr void set_from_zero(int i, int value)
    {
        assert(value >= 0);
        assert(value <= mask);
        data_[i / values_per_word] |=
            value << ((i % values_per_word) * bits_per_value);
    }

    /** @copydoc bit_packed_span::clear */
    constexpr void clear(int i)
    {
        data_[i / values_per_word] &=
            ~(mask << ((i % values_per_word) * bits_per_value));
    }

    /** @copydoc bit_packed_span::set */
    constexpr void set(int i, int value)
    {
        clear(i);
        set_from_zero(i, value);
    }

    /** @copydoc bit_packed_span::get */
    constexpr int get(int i) const
    {
        return (data_[i / values_per_word] >>
                (i % values_per_word) * bits_per_value) &
               mask;
    }

private:
    storage_type data_[num_words];
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_BIT_PACKED_STORAGE_HPP_
