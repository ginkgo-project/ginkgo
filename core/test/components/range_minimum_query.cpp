// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query.hpp"

#include <random>

#include <gtest/gtest.h>

#include "core/test/utils.hpp"


TEST(RangeMinimumQuery, RepresentativesAreExhaustive)
{
    constexpr auto size = 8;
    using tree = gko::detail::cartesian_tree<size>;
    int values[size]{};
    std::iota(values, values + size, 0);
    const auto reps = tree::compute_tree_representatives();
    do {
        const auto tree_number =
            tree::compute_tree_index(values, tree::ballot_number);
        const auto rep_tree_number =
            tree::compute_tree_index(reps[tree_number], tree::ballot_number);

        ASSERT_EQ(tree_number, rep_tree_number);
    } while (std::next_permutation(values, values + size));
}


TEST(RangeMinimumQuery, LookupRepresentatives)
{
    constexpr auto size = 8;
    using tree = gko::detail::cartesian_tree<size>;
    gko::device_block_range_minimum_query_lookup_table<size> table;
    const auto reps = tree::compute_tree_representatives();
    for (const auto& rep : reps) {
        const auto tree = tree::compute_tree_index(rep, tree::ballot_number);
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
    gko::device_block_range_minimum_query_lookup_table<size> table;
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
    constexpr auto data = gko::device_range_minimum_query_superblocks<
        gko::int32>::compute_block_offset_lookup();
    constexpr auto data_long = gko::device_range_minimum_query_superblocks<
        gko::int64>::compute_block_offset_lookup();
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


TEST(RangeMinimumQuery, NumLevelsIsCorrect)
{
    const auto test = [](auto value) {
        using index_type = decltype(value);
        using superblocks =
            gko::device_range_minimum_query_superblocks<index_type>;
        ASSERT_EQ(superblocks::num_levels(0), 0);
        ASSERT_EQ(superblocks::num_levels(1), 0);
        ASSERT_EQ(superblocks::num_levels(2), 1);
        ASSERT_EQ(superblocks::num_levels(3), 1);
        ASSERT_EQ(superblocks::num_levels(4), 1);
        ASSERT_EQ(superblocks::num_levels(5), 2);
        ASSERT_EQ(superblocks::num_levels(8), 2);
        ASSERT_EQ(superblocks::num_levels(9), 3);
        ASSERT_EQ(superblocks::num_levels(16), 3);
        ASSERT_EQ(superblocks::num_levels(17), 4);
        ASSERT_EQ(superblocks::num_levels(32), 4);
        ASSERT_EQ(superblocks::num_levels(33), 5);
        ASSERT_EQ(
            superblocks::num_levels(std::numeric_limits<index_type>::max()),
            sizeof(index_type) * CHAR_BIT - 2);
    };
    test(gko::int32{});
    test(gko::int64{});
}


TEST(RangeMinimumQuery, LevelForDistanceIsCorrect)
{
    const auto test = [](auto value) {
        using index_type = decltype(value);
        using superblocks =
            gko::device_range_minimum_query_superblocks<index_type>;
        ASSERT_EQ(superblocks::level_for_distance(0), 0);
        ASSERT_EQ(superblocks::level_for_distance(1), 0);
        ASSERT_EQ(superblocks::level_for_distance(2), 0);
        ASSERT_EQ(superblocks::level_for_distance(3), 0);
        ASSERT_EQ(superblocks::level_for_distance(4), 1);
        ASSERT_EQ(superblocks::level_for_distance(7), 1);
        ASSERT_EQ(superblocks::level_for_distance(8), 2);
        ASSERT_EQ(superblocks::level_for_distance(15), 2);
        ASSERT_EQ(superblocks::level_for_distance(16), 3);
        ASSERT_EQ(superblocks::level_for_distance(
                      std::numeric_limits<index_type>::max()),
                  sizeof(index_type) * CHAR_BIT - 3);
    };
    test(gko::int32{});
    test(gko::int64{});
}
