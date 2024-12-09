// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/index_range.hpp"

#include <gtest/gtest.h>


TEST(IRange, KnowsItsProperties)
{
    gko::irange<int> range(0, 10);

    ASSERT_EQ(range.begin_index(), 0);
    ASSERT_EQ(range.end_index(), 10);
    ASSERT_EQ(*range.begin(), 0);
    // For other iterators, this would be illegal, but it is allowed for
    // irange::iterator
    ASSERT_EQ(*range.end(), 10);
    ASSERT_TRUE(range == gko::irange<int>(0, 10));
    ASSERT_TRUE(range != gko::irange<int>(1, 10));
    ASSERT_TRUE(range != gko::irange<int>(0, 9));
    ASSERT_FALSE(range != gko::irange<int>(0, 10));
    ASSERT_FALSE(range == gko::irange<int>(1, 10));
    ASSERT_FALSE(range == gko::irange<int>(0, 9));
    // test single-argument constructor
    ASSERT_TRUE(range == gko::irange<int>(10));
}


TEST(IRange, SingleParameterConstructor)
{
    gko::irange<int> range(10);

    ASSERT_EQ(range.begin_index(), 0);
    ASSERT_EQ(range.end_index(), 10);
}


TEST(IRange, RangeFor)
{
    std::vector<int> v;

    for (auto i : gko::irange<int>(1, 4)) {
        v.push_back(i);
    }

    ASSERT_EQ(v, std::vector<int>({1, 2, 3}));
}


TEST(IRange, WorksInAlgorithm)
{
    gko::irange<int> range(1, 15);

    auto it = std::lower_bound(range.begin(), range.end(), 10);

    ASSERT_EQ(*it, 10);
    ASSERT_EQ(it - range.begin(), 9);
}


TEST(IRangeIterator, IteratorProperties)
{
    gko::irange<int> range(0, 10);

    auto it = range.begin();
    it += 4;
    ASSERT_EQ(*it, 4);
    it -= 4;
    ASSERT_EQ(*it, 0);
    ASSERT_EQ(*it++, 0);
    ASSERT_EQ(*++it, 2);
    ASSERT_EQ(*it--, 2);
    ASSERT_EQ(*--it, 0);
    ASSERT_EQ(*(it + 1), 1);
    ASSERT_EQ(*(it - -1), 1);
    ASSERT_EQ(it[2], 2);
    ASSERT_EQ((it + 4) - it, 4);
    ASSERT_TRUE(it == it);
    ASSERT_TRUE(it != (it + 1));
    ASSERT_TRUE(it < (it + 1));
    ASSERT_TRUE(it <= (it + 1));
    ASSERT_TRUE(it <= it);
    ASSERT_TRUE((it + 1) > it);
    ASSERT_TRUE((it + 1) >= it);
    ASSERT_TRUE(it >= it);
    ASSERT_FALSE(it != it);
    ASSERT_FALSE(it == (it + 1));
    ASSERT_FALSE(it > it);
    ASSERT_FALSE(it > (it + 1));
    ASSERT_FALSE(it < it);
    ASSERT_FALSE((it + 1) < it);
    ASSERT_FALSE(it >= (it + 1));
    ASSERT_FALSE((it + 1) <= it);
}


#ifndef NDEBUG


bool check_assertion_exit_code(int exit_code)
{
#ifdef _MSC_VER
    // MSVC picks up the exit code incorrectly,
    // so we can only check that it exits
    return true;
#else
    return exit_code != 0;
#endif
}


TEST(DeathTest, Assertions)
{
    // irange
    // end >= begin
    EXPECT_EXIT((void)gko::irange<int>(1, 0), check_assertion_exit_code, "");
}


#endif
