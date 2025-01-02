// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query.hpp"

#include <random>

#include <gtest/gtest.h>

#include "core/test/utils.hpp"
#include "gtest/gtest.h"


TEST(RangeMinimumQuery, RepresentativesAreExhaustive)
{
    constexpr auto size = 8;
    using tree = gko::detail::cartesian_tree<size>;
    int values[size]{};
    std::iota(values, values + size, 0);
    constexpr auto reps = tree::representatives;
    do {
        const auto tree_number = tree::compute_tree_index(values);
        const auto rep_tree_number =
            tree::compute_tree_index(reps[tree_number]);

        ASSERT_EQ(tree_number, rep_tree_number);
    } while (std::next_permutation(values, values + size));
}


TEST(RangeMinimumQuery, RepresentativesLargeAreExhaustive)
{
    constexpr auto size = 9;
    using tree = gko::detail::cartesian_tree<size>;
    int values[size]{};
    std::iota(values, values + size, 0);
    constexpr auto reps = tree::representatives;
    do {
        const auto tree_number = tree::compute_tree_index(values);
        const auto rep_tree_number =
            tree::compute_tree_index(reps[tree_number]);

        ASSERT_EQ(tree_number, rep_tree_number);
    } while (std::next_permutation(values, values + size));
}


TEST(RangeMinimumQuery, LookupRepresentatives)
{
    constexpr auto size = 8;
    using tree = gko::detail::cartesian_tree<size>;
    constexpr gko::block_range_minimum_query_lookup_table<size> table;
    auto reps = tree::compute_tree_representatives();
    for (const auto& rep : reps) {
        const auto tree = tree::compute_tree_index(rep);
        for (const auto first : gko::irange{size}) {
            for (const auto last : gko::irange{size}) {
                const auto begin = rep + first;
                const auto end = rep + last + 1;
                const auto min_pos =
                    first > last ? 0 : std::min_element(begin, end) - rep;

                ASSERT_EQ(table.lookup(tree, first, last), min_pos);
            }
        }
    }
}


TEST(RangeMinimumQuery, LookupExhaustive)
{
    constexpr auto size = 8;
    gko::block_range_minimum_query_lookup_table<size> table;
    int values[size]{};
    std::iota(values, values + size, 0);
    do {
        const auto tree_number = table.compute_tree_index(values);
        for (const auto first : gko::irange{size}) {
            for (const auto last : gko::irange{first, size}) {
                const auto lookup_val = table.lookup(tree_number, first, last);
                const auto actual_val =
                    std::min_element(values + first, values + last + 1) -
                    values;

                ASSERT_EQ(lookup_val, actual_val);
            }
        }
    } while (std::next_permutation(values, values + size));
}


TEST(RangeMinimumQuery, OffsetsAreCorrect)
{
    constexpr auto data = gko::range_minimum_query_superblocks<
        int>::compute_block_offset_lookup();
    constexpr auto data_long = gko::range_minimum_query_superblocks<
        long>::compute_block_offset_lookup();
    ASSERT_EQ(data[0], 0);
    ASSERT_EQ(data_long[0], 0);
    // blocks of size 2^1 need 1 bit each
    ASSERT_EQ(data[1], 1);
    ASSERT_EQ(data_long[1], 1);
    // blocks of size 2^2 need 2 bits each
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data_long[2], 3);
    // blocks of size 2^3 need 4 bits each
    ASSERT_EQ(data[3], 7);
    ASSERT_EQ(data_long[3], 7);
    // blocks of size 2^4 need 4 bits each
    ASSERT_EQ(data[4], 11);
    ASSERT_EQ(data_long[4], 11);
    // blocks of size 2^5 need 8 bits each
    ASSERT_EQ(data[5], 19);
    ASSERT_EQ(data_long[5], 19);
    // blocks of size 2^6 need 8 bits each
    ASSERT_EQ(data[6], 27);
    ASSERT_EQ(data_long[6], 27);
    // blocks of size 2^7 need 8 bits each
    ASSERT_EQ(data[7], 35);
    ASSERT_EQ(data_long[7], 35);
    // blocks of size 2^8 need 8 bits each
    ASSERT_EQ(data[8], 43);
    ASSERT_EQ(data_long[8], 43);
    // blocks of size 2^9 - 2^16 need 16 bits each
    ASSERT_EQ(data[9], 59);
    ASSERT_EQ(data_long[9], 59);
    ASSERT_EQ(data[16], 171);
    ASSERT_EQ(data_long[16], 171);
    // blocks of size 2^17-2^32 need 32 bits each
    ASSERT_EQ(data[31], 651);
    ASSERT_EQ(data_long[31], 651);
    ASSERT_EQ(data_long[32], 683);
    // blocks of size 2^33-2^64 need 64 bits each
    ASSERT_EQ(data_long[63], 2667);
}


template <typename WordType>
class BitPackedSpan : public ::testing::Test {
public:
    using word_type = WordType;
    using bit_packed_span = gko::detail::bit_packed_span<WordType>;

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

            bit_packed_span span{packed_data.data(), num_bits, size};
            bit_packed_span span2{packed_data2.data(), num_bits, size};
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
