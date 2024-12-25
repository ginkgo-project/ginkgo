// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query.hpp"

#include <gtest/gtest.h>

#include "core/test/utils.hpp"


namespace {


TEST(RangeMinimumQuery, CatalanNumber)
{
    using gko::detail::cartesian_tree;
    ASSERT_EQ(cartesian_tree::catalan_number(1), 1);
    ASSERT_EQ(cartesian_tree::catalan_number(2), 2);
    ASSERT_EQ(cartesian_tree::catalan_number(3), 5);
    ASSERT_EQ(cartesian_tree::catalan_number(4), 14);
    ASSERT_EQ(cartesian_tree::catalan_number(5), 42);
    ASSERT_EQ(cartesian_tree::catalan_number(6), 132);
    ASSERT_EQ(cartesian_tree::catalan_number(7), 429);
    ASSERT_EQ(cartesian_tree::catalan_number(8), 1430);
}


TEST(RangeMinimumQuery, BallotNumber)
{
    using gko::detail::cartesian_tree;
    ASSERT_EQ(cartesian_tree::ballot_number(0, 0), 1);
    ASSERT_EQ(cartesian_tree::ballot_number(0, 3), 1);
    ASSERT_EQ(cartesian_tree::ballot_number(0, 2), 1);
    ASSERT_EQ(cartesian_tree::ballot_number(1, 3), 3);
}


TEST(RangeMinimumQuery, ComputeTreeNumber)
{
    using gko::detail::cartesian_tree;
    ASSERT_EQ((cartesian_tree::compute_tree_index<3, int>({1, 2, 3})), 0);
    ASSERT_EQ((cartesian_tree::compute_tree_index<3, int>({1, 3, 2})), 1);
    ASSERT_EQ((cartesian_tree::compute_tree_index<3, int>({2, 3, 1})), 2);
    ASSERT_EQ((cartesian_tree::compute_tree_index<3, int>({2, 1, 3})), 3);
    ASSERT_EQ((cartesian_tree::compute_tree_index<3, int>({3, 2, 1})), 4);
}


TEST(RangeMinimumQuery, Representatives)
{
    using gko::detail::cartesian_tree;
    constexpr auto size = 8;
    std::array<int, size> values{{}};
    std::iota(values.begin(), values.end(), 0);
    constexpr auto reps = cartesian_tree::compute_tree_representatives<size>();
    do {
        const auto tree_number =
            cartesian_tree::compute_tree_index<size>(values);
        const auto rep_tree_number =
            cartesian_tree::compute_tree_index<size>(reps[tree_number]);
        ASSERT_EQ(tree_number, rep_tree_number);
    } while (std::next_permutation(values.begin(), values.end()));
}


TEST(RangeMinimumQuery, Lookup)
{
    using namespace gko::detail;
    constexpr auto size = 8;
    block_range_minimum_query_lookup_table<size> table;
    auto reps = cartesian_tree::compute_tree_representatives<size>();
    for (auto v : table.lookup_table) {
        for (auto vv : v) {
            std::cout << std::hex << vv << ' ';
        }
        std::cout << '\n';
    }
    for (const auto& rep : reps) {
        const auto tree = cartesian_tree::compute_tree_index(rep);
        for (const auto first : gko::irange{size}) {
            for (const auto last : gko::irange{size}) {
                const auto begin = rep.begin() + first;
                const auto end = rep.begin() + last + 1;
                const auto min_pos =
                    begin > end ? 0
                                : std::distance(rep.begin(),
                                                std::min_element(begin, end));
                ASSERT_EQ(table.lookup(tree, first, last), min_pos)
                    << tree << ' ' << first << ' ' << last;
            }
        }
    }
}


}  // namespace
