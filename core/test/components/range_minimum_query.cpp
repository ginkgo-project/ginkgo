// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query.hpp"

#include <gtest/gtest.h>

#include "core/test/utils.hpp"


namespace {


TEST(RangeMinimumQuery, CatalanNumber)
{
    ASSERT_EQ(gko::detail::catalan_number(1), 1);
    ASSERT_EQ(gko::detail::catalan_number(2), 2);
    ASSERT_EQ(gko::detail::catalan_number(3), 5);
    ASSERT_EQ(gko::detail::catalan_number(4), 14);
    ASSERT_EQ(gko::detail::catalan_number(5), 42);
    ASSERT_EQ(gko::detail::catalan_number(6), 132);
    ASSERT_EQ(gko::detail::catalan_number(7), 429);
    ASSERT_EQ(gko::detail::catalan_number(8), 1430);
}


TEST(RangeMinimumQuery, BallotNumber)
{
    ASSERT_EQ(gko::detail::ballot_number(0, 0), 1);
    ASSERT_EQ(gko::detail::ballot_number(0, 3), 1);
    ASSERT_EQ(gko::detail::ballot_number(0, 2), 1);
    ASSERT_EQ(gko::detail::ballot_number(1, 3), 3);
}


TEST(RangeMinimumQuery, ComputeTreeNumber)
{
    ASSERT_EQ((gko::detail::compute_tree_number<3, int>({1, 2, 3})), 0);
    ASSERT_EQ((gko::detail::compute_tree_number<3, int>({1, 3, 2})), 1);
    ASSERT_EQ((gko::detail::compute_tree_number<3, int>({2, 3, 1})), 2);
    ASSERT_EQ((gko::detail::compute_tree_number<3, int>({2, 1, 3})), 3);
    ASSERT_EQ((gko::detail::compute_tree_number<3, int>({3, 2, 1})), 4);
}


TEST(RangeMinimumQuery, Representatives)
{
    constexpr auto size = 8;
    std::array<int, size> values{{}};
    std::iota(values.begin(), values.end(), 0);
    constexpr auto reps = gko::detail::compute_tree_representatives<size>();
    do {
        const auto tree_number = gko::detail::compute_tree_number<size>(values);
        const auto rep_tree_number =
            gko::detail::compute_tree_number<size>(reps[tree_number]);
        ASSERT_EQ(tree_number, rep_tree_number);
    } while (std::next_permutation(values.begin(), values.end()));
}


}  // namespace
