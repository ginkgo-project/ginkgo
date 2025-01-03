// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/bit_packed_storage.hpp"

#include <random>

#include <gtest/gtest.h>

#include "core/test/utils.hpp"
#include "gtest/gtest.h"


template <typename WordType>
class BitPackedSpan : public ::testing::Test {
public:
    using word_type = WordType;
    using bit_packed_span = gko::bit_packed_span<WordType>;

    std::default_random_engine rng{2457};
};

using WordTypes = ::testing::Types<gko::uint32, gko::uint64>;

TYPED_TEST_SUITE(BitPackedSpan, WordTypes, TypenameNameGenerator);


TYPED_TEST(BitPackedSpan, Works)
{
    using bit_packed_span = typename TestFixture::bit_packed_span;
    using word_type = typename TestFixture::word_type;
    for (const auto size : {0, 10, 100, 1000, 1023, 1023}) {
        SCOPED_TRACE(size);
        for (const auto num_bits : {2, 3, 5, 7, 8, 9, 31}) {
            SCOPED_TRACE(num_bits);
            const word_type max_value = 1ull << num_bits;
            std::vector<word_type> packed_data(
                bit_packed_span::storage_size(size, num_bits));
            std::vector<word_type> packed_data2(
                bit_packed_span::storage_size(size, num_bits));
            std::vector<word_type> data(size);
            std::vector<word_type> retrieved_data(size);
            std::vector<word_type> retrieved_data2(size);
            std::uniform_int_distribution<word_type> dist{
                0, static_cast<word_type>(max_value - 1)};
            for (auto& val : data) {
                val = dist(this->rng);
            }
            // scrable packed_data2 to check proper initialization
            std::uniform_int_distribution<word_type> dist2{word_type{},
                                                           ~word_type{}};
            for (auto& val : packed_data2) {
                val = dist2(this->rng);
            }

            bit_packed_span span{packed_data.data(), num_bits,
                                 static_cast<gko::size_type>(size)};
            bit_packed_span span2{packed_data2.data(), num_bits,
                                  static_cast<gko::size_type>(size)};
            for (const auto i : gko::irange{size}) {
                span.set_from_zero(i, data[i]);
                span2.set(i, data[i]);
            }

            for (const auto i : gko::irange{size}) {
                retrieved_data[i] = span.get(i);
                retrieved_data2[i] = span2.get(i);
            }
            ASSERT_EQ(data, retrieved_data);
            ASSERT_EQ(data, retrieved_data2);
        }
    }
}


template <int num_bits_, int size_>
struct size_info {
    constexpr static int num_bits = num_bits_;
    constexpr static int size = size_;
};


template <typename SizeInfoType>
class BitPackedArray : public ::testing::Test {
public:
    constexpr static int num_bits = SizeInfoType::num_bits;
    constexpr static int size = SizeInfoType::size;
    using bit_packed_array = gko::bit_packed_array<num_bits, size>;

    std::default_random_engine rng{6735};
};

using Sizes =
    ::testing::Types<size_info<1, 15>,    // single word, partially filled
                     size_info<1, 1024>,  // multiple words
                     size_info<3, 8>,   // single word, non power-of-two number
                                        // of bits, fully filled
                     size_info<3, 9>,   // multiple words, partially filled
                     size_info<5, 52>,  // larger size
                     size_info<32, 3>   // single word for each value
                     >;

TYPED_TEST_SUITE(BitPackedArray, Sizes, TypenameNameGenerator);


TYPED_TEST(BitPackedArray, Works)
{
    constexpr auto num_bits = TestFixture::num_bits;
    constexpr auto size = TestFixture::size;
    using bit_packed_array = typename TestFixture::bit_packed_array;
    using word_type = typename bit_packed_array::word_type;
    const auto max_value = word_type{2} << (num_bits - 1);
    std::array<word_type, size> data{};
    std::array<word_type, size> retrieved_data{};
    std::array<word_type, size> retrieved_data2{};
    std::uniform_int_distribution<word_type> dist{
        0, static_cast<word_type>(max_value - 1)};
    for (auto& val : data) {
        val = dist(this->rng);
    }

    bit_packed_array array;
    bit_packed_array array2;
    for (const auto i : gko::irange{size}) {
        array.set_from_zero(i, data[i]);
        array2.set(i, dist(this->rng));
        array2.set(i, data[i]);
    }

    for (const auto i : gko::irange{size}) {
        retrieved_data[i] = array.get(i);
        retrieved_data2[i] = array.get(i);
    }
    ASSERT_EQ(data, retrieved_data);
    ASSERT_EQ(data, retrieved_data2);
}
