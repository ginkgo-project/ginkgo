// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/range_minimum_query.hpp"

#include <gtest/gtest.h>

#include "core/test/utils.hpp"


namespace {


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
    constexpr gko::detail::block_range_minimum_query_lookup_table<size> table;
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
    gko::detail::block_range_minimum_query_lookup_table<size> table;
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


}  // namespace
