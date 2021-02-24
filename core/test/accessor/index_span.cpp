/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
