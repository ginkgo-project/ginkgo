/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, Karlsruhe Institute of Technology
Copyright (c) 2017-2019, Universitat Jaume I
Copyright (c) 2017-2019, University of Tennessee
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

#include <ginkgo/core/base/range.hpp>


#include <gtest/gtest.h>


#include <array>


namespace {


TEST(Span, CreatesSpan)
{
    gko::span s{3, 5};

    ASSERT_EQ(s.begin, 3);
    ASSERT_EQ(s.end, 5);
}


TEST(Span, CreatesPoint)
{
    gko::span s{3};

    ASSERT_EQ(s.begin, 3);
    ASSERT_EQ(s.end, 4);
}


TEST(Span, LessThanEvaluatesToTrue)
{
    ASSERT_TRUE(gko::span(2, 3) < gko::span(4, 7));
}


TEST(Span, LessThanEvaluatesToFalse)
{
    ASSERT_FALSE(gko::span(2, 4) < gko::span(4, 7));
}


TEST(Span, LessOrEqualEvaluatesToTrue)
{
    ASSERT_TRUE(gko::span(2, 4) <= gko::span(4, 7));
}


TEST(Span, LessOrEqualEvaluatesToFalse)
{
    ASSERT_FALSE(gko::span(2, 5) <= gko::span(4, 7));
}


TEST(Span, GreaterThanEvaluatesToTrue)
{
    ASSERT_TRUE(gko::span(4, 7) > gko::span(2, 3));
}


TEST(Span, GreaterThanEvaluatesToFalse)
{
    ASSERT_FALSE(gko::span(4, 7) > gko::span(2, 4));
}


TEST(Span, GreaterOrEqualEvaluatesToTrue)
{
    ASSERT_TRUE(gko::span(4, 7) >= gko::span(2, 4));
}


TEST(Span, GreaterOrEqualEvaluatesToFalse)
{
    ASSERT_FALSE(gko::span(4, 7) >= gko::span(2, 5));
}


TEST(Span, EqualityEvaluatesToTrue)
{
    ASSERT_TRUE(gko::span(2, 4) == gko::span(2, 4));
}


TEST(Span, EqualityEvaluatesToFalse)
{
    ASSERT_FALSE(gko::span(3, 4) == gko::span(2, 5));
}


TEST(Span, NotEqualEvaluatesToTrue)
{
    ASSERT_TRUE(gko::span(3, 4) != gko::span(2, 5));
}


TEST(Span, NotEqualEvaluatesToFalse)
{
    ASSERT_FALSE(gko::span(2, 4) != gko::span(2, 4));
}


// 0-memory constant accessor, which "stores" x*i + y*j + k at location
// (i, j, k)
struct dummy_accessor {
    static constexpr gko::size_type dimensionality = 3;

    dummy_accessor(gko::size_type size, int x, int y)
        : sizes{size, size, size}, x{x}, y{y}
    {}

    dummy_accessor(gko::size_type size_x, gko::size_type size_y,
                   gko::size_type size_z, int x, int y)
        : sizes{size_x, size_y, size_z}, x{x}, y{y}
    {}

    int operator()(int a, int b, int c) const { return x * a + y * b + c; }

    void copy_from(const dummy_accessor &other) const
    {
        x = other.x;
        y = other.y;
    }

    gko::size_type length(gko::size_type dim) const { return sizes[dim]; }

    std::array<gko::size_type, 3> sizes;
    mutable int x;
    mutable int y;
};


using dummy_range = gko::range<dummy_accessor>;


TEST(Range, CreatesRange)
{
    dummy_range r{5u, 2, 3};

    EXPECT_EQ(r->x, 2);
    ASSERT_EQ(r->y, 3);
}


TEST(Range, ForwardsCallsToAccessor)
{
    dummy_range r{5u, 2, 3};

    EXPECT_EQ(r(1, 2, 3), 2 * 1 + 3 * 2 + 3);
    ASSERT_EQ(r(4, 2, 5), 2 * 4 + 3 * 2 + 5);
}


TEST(Range, ForwardsCopyToAccessor)
{
    dummy_range r{5u, 2, 3};
    r = dummy_range{5u, 2, 5};

    EXPECT_EQ(r->x, 2);
    ASSERT_EQ(r->y, 5);
}


TEST(Range, ForwardsLength)
{
    dummy_range r{5u, 2, 3};

    EXPECT_EQ(r->length(0), 5);
    EXPECT_EQ(r->length(1), 5);
    ASSERT_EQ(r->length(2), 5);
}


TEST(Range, ComputesUnaryPlus)
{
    dummy_range r{5u, 2, 3};

    auto res = +r;

    EXPECT_EQ(res(1, 2, 3), +(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), +(8 + 6 + 5));
}


TEST(Range, ComputesUnaryMinus)
{
    dummy_range r{5u, 2, 3};

    auto res = -r;

    EXPECT_EQ(res(1, 2, 3), -(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), -(8 + 6 + 5));
}


TEST(Range, ComputesLogicalNot)
{
    dummy_range r{5u, 2, 3};

    auto res = !r;

    EXPECT_EQ(res(1, 2, 3), !(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), !(8 + 6 + 5));
}


TEST(Range, ComputesBitwiseNot)
{
    dummy_range r{5u, 2, 3};

    auto res = ~r;

    EXPECT_EQ(res(1, 2, 3), ~(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), ~(8 + 6 + 5));
}


TEST(Range, AppliesZero)
{
    dummy_range r{5u, 2, 3};

    auto res = zero(r);

    EXPECT_EQ(res(1, 2, 3), gko::zero(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::zero(8 + 6 + 5));
}


TEST(Range, AppliesOne)
{
    dummy_range r{5u, 2, 3};

    auto res = one(r);

    EXPECT_EQ(res(1, 2, 3), gko::one(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::one(8 + 6 + 5));
}


TEST(Range, AppliesAbs)
{
    dummy_range r{5u, 2, 3};

    auto res = abs(r);

    EXPECT_EQ(res(-1, -2, -3), gko::abs(-2 + -6 + -3));
    ASSERT_EQ(res(4, 2, 5), gko::abs(8 + 6 + 5));
}


TEST(Range, AppliesReal)
{
    dummy_range r{5u, 2, 3};

    auto res = real(r);

    EXPECT_EQ(res(1, 2, 3), gko::real(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::real(8 + 6 + 5));
}


TEST(Range, AppliesImag)
{
    dummy_range r{5u, 2, 3};

    auto res = imag(r);

    EXPECT_EQ(res(1, 2, 3), gko::imag(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::imag(8 + 6 + 5));
}


TEST(Range, AppliesConj)
{
    dummy_range r{5u, 2, 3};

    auto res = conj(r);

    EXPECT_EQ(res(1, 2, 3), gko::conj(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::conj(8 + 6 + 5));
}


TEST(Range, AppliesSquaredNorm)
{
    dummy_range r{5u, 2, 3};

    auto res = squared_norm(r);

    EXPECT_EQ(res(1, 2, 3), gko::squared_norm(2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::squared_norm(8 + 6 + 5));
}


TEST(Range, TransposesRange)
{
    dummy_range r{5u, 1, 2};

    auto res = transpose(r);

    EXPECT_EQ(res(1, 2, 3), r(2, 1, 3));
    EXPECT_EQ(res(4, 2, 5), r(2, 4, 5));
}


TEST(Range, AddsRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 + r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) + (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) + (8 + 6 + 5));
}


TEST(Range, SubtractsRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 - r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) - (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) - (8 + 6 + 5));
}


TEST(Range, MultipliesRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 * r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) * (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) * (8 + 6 + 5));
}


TEST(Range, DividesRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 / r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) / (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) / (8 + 6 + 5));
}


TEST(Range, ModsRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 % r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) % (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) % (8 + 6 + 5));
}


TEST(Range, ComparesRangesWithLessThan)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 < r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) < (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) < (8 + 6 + 5));
}


TEST(Range, ComparesRangesWithGreaterThan)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 > r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) > (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) > (8 + 6 + 5));
}


TEST(Range, ComparesRangesWithLessOrEqual)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 <= r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) <= (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) <= (8 + 6 + 5));
}


TEST(Range, ComparesRangesWithGreaterOrEqual)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 >= r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) >= (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) >= (8 + 6 + 5));
}


TEST(Range, ComparesRangesWithEqual)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 == r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) == (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) == (8 + 6 + 5));
}


TEST(Range, ComparesRangesWithNotEqual)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 != r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) != (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) != (8 + 6 + 5));
}


TEST(Range, AppliesOrToRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 0, 0};

    auto res = r1 || r2;

    EXPECT_EQ(res(0, 0, 0), (0 + 0 + 0) || (0 + 0 + 0));
    ASSERT_EQ(res(4, 2, 0), (4 + 4 + 0) || (0 + 0 + 0));
}


TEST(Range, AppliesAndToRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 0, 0};

    auto res = r1 && r2;

    EXPECT_EQ(res(0, 0, 0), (0 + 0 + 0) && (0 + 0 + 0));
    ASSERT_EQ(res(4, 2, 0), (4 + 4 + 0) && (0 + 0 + 0));
}


TEST(Range, AppliesBitwiseOrToRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 | r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) | (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) | (8 + 6 + 5));
}


TEST(Range, AppliesBitwiseAndToRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 & r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) & (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) & (8 + 6 + 5));
}


TEST(Range, AppliesBitwiseXorOnRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 ^ r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) ^ (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) ^ (8 + 6 + 5));
}


TEST(Range, ShiftsRangesLeft)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 << r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) << (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) << (8 + 6 + 5));
}


TEST(Range, ShiftsRangesRight)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = r1 >> r2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) >> (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) >> (8 + 6 + 5));
}


TEST(Range, ComputesMaximumOfRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = max(r1, r2);

    EXPECT_EQ(res(1, 2, 3), gko::max(1 + 4 + 3, 2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::max(4 + 4 + 5, 8 + 6 + 5));
}


TEST(Range, ComputesMinimumOfRanges)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = min(r1, r2);

    EXPECT_EQ(res(1, 2, 3), gko::min(1 + 4 + 3, 2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), gko::min(4 + 4 + 5, 8 + 6 + 5));
}


TEST(Range, MultipliesMatrices)
{
    dummy_range r1{2u, 1, 2};
    dummy_range r2{2u, 2, 3};

    auto res = mmul(r1, r2);

    EXPECT_EQ(res(0, 1, 0),
              r1(0, 0, 0) * r2(0, 1, 0) + r1(0, 1, 0) * r2(1, 1, 0));
    ASSERT_EQ(res(1, 1, 1),
              r1(1, 0, 1) * r2(0, 1, 1) + r1(1, 1, 1) * r2(1, 1, 1));
}


TEST(Range, MultipliesMatricesOfDifferentSizes)
{
    dummy_range r1{2u, 1u, 1u, 1, 2};
    dummy_range r2{1u, 3u, 1u, 2, 3};

    auto res = mmul(r1, r2);

    EXPECT_EQ(res(0, 1, 0), r1(0, 0, 0) * r2(0, 1, 0));
    ASSERT_EQ(res(1, 2, 1), r1(1, 0, 1) * r2(0, 2, 1));
}


TEST(Range, AddsScalarAndRange)
{
    dummy_range r{5u, 2, 3};

    auto res = 2 + r;

    EXPECT_EQ(res(1, 2, 3), 2 + (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), 2 + (8 + 6 + 5));
}


TEST(Range, SubtractsScalarAndRange)
{
    dummy_range r{5u, 2, 3};

    auto res = 2 - r;

    EXPECT_EQ(res(1, 2, 3), 2 - (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), 2 - (8 + 6 + 5));
}


TEST(Range, MultipliesScalarAndRange)
{
    dummy_range r{5u, 2, 3};

    auto res = 2 * r;

    EXPECT_EQ(res(1, 2, 3), 2 * (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), 2 * (8 + 6 + 5));
}


TEST(Range, DividesScalarAndRange)
{
    dummy_range r{5u, 2, 3};

    auto res = 50 / r;

    EXPECT_EQ(res(1, 2, 3), 50 / (2 + 6 + 3));
    ASSERT_EQ(res(4, 2, 5), 50 / (8 + 6 + 5));
}


TEST(Range, AddsRangeAndSclar)
{
    dummy_range r{5u, 1, 2};

    auto res = r + 2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) + 2);
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) + 2);
}


TEST(Range, SubtractsRangeAndSclar)
{
    dummy_range r{5u, 1, 2};

    auto res = r - 2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) - 2);
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) - 2);
}


TEST(Range, MultipliesRangeAndSclar)
{
    dummy_range r{5u, 1, 2};

    auto res = r * 2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) * 2);
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) * 2);
}


TEST(Range, DividesRangeAndSclar)
{
    dummy_range r{5u, 1, 2};

    auto res = r / 2;

    EXPECT_EQ(res(1, 2, 3), (1 + 4 + 3) / 2);
    ASSERT_EQ(res(4, 2, 5), (4 + 4 + 5) / 2);
}


TEST(Range, CanAssembleComplexOperation)
{
    dummy_range r1{5u, 1, 2};
    dummy_range r2{5u, 2, 3};

    auto res = abs(r1 - r2) / max(abs(r1), abs(r2));

    EXPECT_EQ(res(1, 2, 3), gko::abs(8 - 11) / gko::max(8, 11));
    ASSERT_EQ(res(4, 2, 5), gko::abs(13 - 19) / gko::max(13, 19));
}


}  // namespace
