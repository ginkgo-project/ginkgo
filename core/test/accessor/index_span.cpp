// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include "accessor/index_span.hpp"


namespace {


TEST(IndexSpan, CreatesSpan)
{
    gko::acc::index_span s{3, 5};

    ASSERT_EQ(s.begin, 3);
    ASSERT_EQ(s.end, 5);
}


TEST(IndexSpan, CreatesPoint)
{
    gko::acc::index_span s{3};

    ASSERT_EQ(s.begin, 3);
    ASSERT_EQ(s.end, 4);
}


TEST(IndexSpan, LessThanEvaluatesToTrue)
{
    ASSERT_TRUE(gko::acc::index_span(2, 3) < gko::acc::index_span(4, 7));
}


TEST(IndexSpan, LessThanEvaluatesToFalse)
{
    ASSERT_FALSE(gko::acc::index_span(2, 4) < gko::acc::index_span(4, 7));
}


TEST(IndexSpan, LessOrEqualEvaluatesToTrue)
{
    ASSERT_TRUE(gko::acc::index_span(2, 4) <= gko::acc::index_span(4, 7));
}


TEST(IndexSpan, LessOrEqualEvaluatesToFalse)
{
    ASSERT_FALSE(gko::acc::index_span(2, 5) <= gko::acc::index_span(4, 7));
}


TEST(IndexSpan, GreaterThanEvaluatesToTrue)
{
    ASSERT_TRUE(gko::acc::index_span(4, 7) > gko::acc::index_span(2, 3));
}


TEST(IndexSpan, GreaterThanEvaluatesToFalse)
{
    ASSERT_FALSE(gko::acc::index_span(4, 7) > gko::acc::index_span(2, 4));
}


TEST(IndexSpan, GreaterOrEqualEvaluatesToTrue)
{
    ASSERT_TRUE(gko::acc::index_span(4, 7) >= gko::acc::index_span(2, 4));
}


TEST(IndexSpan, GreaterOrEqualEvaluatesToFalse)
{
    ASSERT_FALSE(gko::acc::index_span(4, 7) >= gko::acc::index_span(2, 5));
}


TEST(IndexSpan, EqualityEvaluatesToTrue)
{
    ASSERT_TRUE(gko::acc::index_span(2, 4) == gko::acc::index_span(2, 4));
}


TEST(IndexSpan, EqualityEvaluatesToFalse)
{
    ASSERT_FALSE(gko::acc::index_span(3, 4) == gko::acc::index_span(2, 5));
}


TEST(IndexSpan, NotEqualEvaluatesToTrue)
{
    ASSERT_TRUE(gko::acc::index_span(3, 4) != gko::acc::index_span(2, 5));
}


TEST(IndexSpan, NotEqualEvaluatesToFalse)
{
    ASSERT_FALSE(gko::acc::index_span(2, 4) != gko::acc::index_span(2, 4));
}


}  // namespace
