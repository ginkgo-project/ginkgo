// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query.hpp"

#include <gtest/gtest.h>

#include "core/test/utils.hpp"


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
