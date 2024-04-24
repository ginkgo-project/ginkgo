// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include "core/base/integer_range.hpp"
#include "core/test/utils/death_test_helpers.hpp"


TEST(IRange, KnowsItsProperties)
{
    gko::irange<int> range(0, 10);

    ASSERT_FALSE(range.empty());
    ASSERT_EQ(range.size(), 10);
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
    ASSERT_TRUE(gko::irange<int>{0}.empty());
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


TEST(IRangeStrided, KnowsItsProperties)
{
    gko::irange_strided<int> range{0, 4, 3};

    ASSERT_EQ(range.begin_index(), 0);
    ASSERT_EQ(range.end_index(), 6);
    ASSERT_EQ(range.stride(), 3);
    ASSERT_EQ(*range.begin(), 0);
    ASSERT_EQ(*range.end(), 6);
}


TEST(IRangeStrided, EndComputationIsCorrect)
{
    ASSERT_EQ(gko::irange_strided<int>(0, 3, 3).end_index(), 3);
    ASSERT_EQ(gko::irange_strided<int>(0, 4, 3).end_index(), 6);
    ASSERT_EQ(gko::irange_strided<int>(0, 5, 3).end_index(), 6);
    ASSERT_EQ(gko::irange_strided<int>(0, 6, 3).end_index(), 6);
}


TEST(IRangeStridedIterator, IteratorProperties)
{
    gko::irange_strided<int> range{0, 4, 3};

    auto it = range.begin();
    auto end = range.end();
    ASSERT_EQ(*it, 0);
    ASSERT_EQ(*++it, 3);
    ASSERT_TRUE(it != end);
    ASSERT_TRUE(end != it);
    ASSERT_FALSE(it == end);
    ASSERT_FALSE(end == it);
    ++it;
    ASSERT_TRUE(it == end);
    ASSERT_TRUE(end == it);
    ASSERT_FALSE(it != end);
    ASSERT_FALSE(end != it);
}


TEST(IRangeStridedIterator, RangeFor)
{
    std::vector<int> v;

    for (auto i : gko::irange_strided<int>(1, 10, 2)) {
        v.push_back(i);
    }

    ASSERT_EQ(v, std::vector<int>({1, 3, 5, 7, 9}));
}


#ifndef NDEBUG


TEST(DeathTest, Assertions)
{
    // integer_iterator
    // stride > 0
    EXPECT_ASSERT_FAILURE(gko::integer_iterator<int>(0, 0));
    // a.stride_ == b.stride_
    EXPECT_ASSERT_FAILURE(gko::integer_iterator<int>(0, 1) -
                          gko::integer_iterator<int>(0, 2));
    // (*a - *b) % a.stride_ == 0
    EXPECT_ASSERT_FAILURE(gko::integer_iterator<int>(0, 2) -
                          gko::integer_iterator<int>(1, 2));
    // irange
    // end >= begin
    EXPECT_ASSERT_FAILURE(gko::irange<int>(1, 0));
    // irange_strided
    // end >= begin
    EXPECT_ASSERT_FAILURE(gko::irange_strided<int>(1, 0, 1));
    // stride > 0
    EXPECT_ASSERT_FAILURE(gko::irange_strided<int>(0, 1, 0));
}


#endif
