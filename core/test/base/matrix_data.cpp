/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/base/matrix_data.hpp>


#include <gtest/gtest.h>


#include <random>


namespace {


TEST(MatrixData, InitializesANullMatrix)
{
    gko::matrix_data<double, int> m;

    ASSERT_EQ(m.size, gko::dim<2>(0, 0));
    ASSERT_EQ(m.nonzeros.size(), 0);
}


TEST(MatrixData, InitializesWithZeros)
{
    gko::matrix_data<double, int> m(gko::dim<2>{3, 5});

    ASSERT_EQ(m.size, gko::dim<2>(3, 5));
    ASSERT_EQ(m.nonzeros.size(), 0);
}


TEST(MatrixData, InitializesWithValue)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    gko::matrix_data<double, int> m(gko::dim<2>{2, 3}, 8.3);

    ASSERT_EQ(m.size, gko::dim<2>(2, 3));
    ASSERT_EQ(m.nonzeros.size(), 6);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 8.3));
    EXPECT_EQ(m.nonzeros[1], nnz(0, 1, 8.3));
    EXPECT_EQ(m.nonzeros[2], nnz(0, 2, 8.3));
    EXPECT_EQ(m.nonzeros[3], nnz(1, 0, 8.3));
    EXPECT_EQ(m.nonzeros[4], nnz(1, 1, 8.3));
    EXPECT_EQ(m.nonzeros[5], nnz(1, 2, 8.3));
}


TEST(MatrixData, InitializesWithRandomValues)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    gko::matrix_data<double, int> m(
        gko::dim<2>{2, 3}, std::uniform_real_distribution<double>(-1, 1),
        std::ranlux48(19));

    ASSERT_EQ(m.size, gko::dim<2>(2, 3));
    ASSERT_LE(m.nonzeros.size(), 6);
    for (const auto &elem : m.nonzeros) {
        EXPECT_TRUE(-1 <= elem.value && elem.value <= 1);
    }
}


TEST(MatrixData, InitializesFromValueList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    // clang-format off
    gko::matrix_data<double, int> m{
        {2, 3, 5},
        {0, 4}
    };
    // clang-format on

    ASSERT_EQ(m.size, gko::dim<2>(2, 3));
    ASSERT_EQ(m.nonzeros.size(), 4);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 2.0));
    EXPECT_EQ(m.nonzeros[1], nnz(0, 1, 3.0));
    EXPECT_EQ(m.nonzeros[2], nnz(0, 2, 5.0));
    EXPECT_EQ(m.nonzeros[3], nnz(1, 1, 4.0));
}


TEST(MatrixData, InitializesRowVectorFromValueList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    gko::matrix_data<double, int> m{{2, 3, 5}};

    ASSERT_EQ(m.size, gko::dim<2>(1, 3));
    ASSERT_EQ(m.nonzeros.size(), 3);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 2.0));
    EXPECT_EQ(m.nonzeros[1], nnz(0, 1, 3.0));
    EXPECT_EQ(m.nonzeros[2], nnz(0, 2, 5.0));
}


TEST(MatrixData, InitializesColumnVectorFromValueList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    gko::matrix_data<double, int> m{{2}, {3}, {5}};

    ASSERT_EQ(m.size, gko::dim<2>(3, 1));
    ASSERT_EQ(m.nonzeros.size(), 3);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 2.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 0, 3.0));
    EXPECT_EQ(m.nonzeros[2], nnz(2, 0, 5.0));
}


TEST(MatrixData, InitializesFromNonzeroList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    gko::matrix_data<double, int> m(gko::dim<2>{5, 7},
                                    {{0, 0, 2}, {1, 1, 0}, {2, 3, 5}});

    ASSERT_EQ(m.size, gko::dim<2>(5, 7));
    ASSERT_EQ(m.nonzeros.size(), 3);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 2.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 1, 0.0));
    EXPECT_EQ(m.nonzeros[2], nnz(2, 3, 5.0));
}


TEST(MatrixData, InitializesDiagonalMatrix)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    const auto m = gko::matrix_data<double, int>::diag(gko::dim<2>{2, 3}, 5.0);

    ASSERT_EQ(m.size, gko::dim<2>(2, 3));
    ASSERT_EQ(m.nonzeros.size(), 2);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 5.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 1, 5.0));
}


TEST(MatrixData, InitializesFromRange)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;
    double data[]{1.2, 0.0, 1.4, 0.0, 0.0, 2.3, 2.4, 0.0};
    gko::range<gko::accessor::row_major<double, 2>> r(data, 2u, 3u, 4u);

    gko::matrix_data<double> m(r);

    ASSERT_EQ(m.size, gko::dim<2>(2, 3));
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 1.2));
    EXPECT_EQ(m.nonzeros[1], nnz(0, 2, 1.4));
    EXPECT_EQ(m.nonzeros[2], nnz(1, 1, 2.3));
    EXPECT_EQ(m.nonzeros[3], nnz(1, 2, 2.4));
}


TEST(MatrixData, InitializesDiagonalMatrixFromValueList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;

    const auto m =
        gko::matrix_data<double, int>::diag(gko::dim<2>{2, 3}, {3, 5});

    ASSERT_EQ(m.size, gko::dim<2>(2, 3));
    ASSERT_EQ(m.nonzeros.size(), 2);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 3.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 1, 5.0));
}


TEST(MatrixData, InitializesBlockDiagonalMatrix)
{
    using data = gko::matrix_data<double, int>;
    using nnz = data::nonzero_type;

    const auto m = data::diag(gko::dim<2>{2, 3}, {{1.0, 2.0}, {3.0, 4.0}});

    ASSERT_EQ(m.size, gko::dim<2>(4, 6));
    ASSERT_EQ(m.nonzeros.size(), 8);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 1.0));
    EXPECT_EQ(m.nonzeros[1], nnz(0, 1, 2.0));
    EXPECT_EQ(m.nonzeros[2], nnz(1, 0, 3.0));
    EXPECT_EQ(m.nonzeros[3], nnz(1, 1, 4.0));
    EXPECT_EQ(m.nonzeros[4], nnz(2, 2, 1.0));
    EXPECT_EQ(m.nonzeros[5], nnz(2, 3, 2.0));
    EXPECT_EQ(m.nonzeros[6], nnz(3, 2, 3.0));
    EXPECT_EQ(m.nonzeros[7], nnz(3, 3, 4.0));
}


TEST(MatrixData, InitializesCheckeredMatrix)
{
    using data = gko::matrix_data<double, int>;
    using nnz = data::nonzero_type;
    gko::matrix_data<double, int> m{{1., 2.}, {3., 4.}};

    gko::matrix_data<double, int> mm{gko::dim<2>{3, 2}, m};

    ASSERT_EQ(mm.size, gko::dim<2>(6, 4));
    ASSERT_EQ(mm.nonzeros.size(), 24);
    EXPECT_EQ(mm.nonzeros[0], nnz(0, 0, 1.0));
    EXPECT_EQ(mm.nonzeros[1], nnz(0, 1, 2.0));
    EXPECT_EQ(mm.nonzeros[2], nnz(0, 2, 1.0));
    EXPECT_EQ(mm.nonzeros[3], nnz(0, 3, 2.0));
    EXPECT_EQ(mm.nonzeros[4], nnz(1, 0, 3.0));
    EXPECT_EQ(mm.nonzeros[5], nnz(1, 1, 4.0));
    EXPECT_EQ(mm.nonzeros[6], nnz(1, 2, 3.0));
    EXPECT_EQ(mm.nonzeros[7], nnz(1, 3, 4.0));
    EXPECT_EQ(mm.nonzeros[8], nnz(2, 0, 1.0));
    EXPECT_EQ(mm.nonzeros[9], nnz(2, 1, 2.0));
    EXPECT_EQ(mm.nonzeros[10], nnz(2, 2, 1.0));
    EXPECT_EQ(mm.nonzeros[11], nnz(2, 3, 2.0));
    EXPECT_EQ(mm.nonzeros[12], nnz(3, 0, 3.0));
    EXPECT_EQ(mm.nonzeros[13], nnz(3, 1, 4.0));
    EXPECT_EQ(mm.nonzeros[14], nnz(3, 2, 3.0));
    EXPECT_EQ(mm.nonzeros[15], nnz(3, 3, 4.0));
    EXPECT_EQ(mm.nonzeros[16], nnz(4, 0, 1.0));
    EXPECT_EQ(mm.nonzeros[17], nnz(4, 1, 2.0));
    EXPECT_EQ(mm.nonzeros[18], nnz(4, 2, 1.0));
    EXPECT_EQ(mm.nonzeros[19], nnz(4, 3, 2.0));
    EXPECT_EQ(mm.nonzeros[20], nnz(5, 0, 3.0));
    EXPECT_EQ(mm.nonzeros[21], nnz(5, 1, 4.0));
    EXPECT_EQ(mm.nonzeros[22], nnz(5, 2, 3.0));
    EXPECT_EQ(mm.nonzeros[23], nnz(5, 3, 4.0));
}


TEST(MatrixData, InitializesDiagonalWithConditionNumber)
{
    using data = gko::matrix_data<double, int>;

    const auto m =
        data::cond(3, 100.0, std::uniform_real_distribution<double>(-1, 1),
                   std::ranlux48(42), 0);

    ASSERT_EQ(m.size, gko::dim<2>(3, 3));
    ASSERT_NEAR(m.nonzeros[0].value / m.nonzeros[2].value, 100.0, 1e-16);
}


TEST(MatrixData, InitializesBlockDiagonalMatrixFromBlockList)
{
    using data = gko::matrix_data<double, int>;
    using nnz = data::nonzero_type;
    auto list = {data{{1.0}}, data{{2.0, 3.0}, {0.0, 4.0}}};

    const auto m = data::diag(begin(list), end(list));

    ASSERT_EQ(m.size, gko::dim<2>(3, 3));
    ASSERT_EQ(m.nonzeros.size(), 4);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 1.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 1, 2.0));
    EXPECT_EQ(m.nonzeros[2], nnz(1, 2, 3.0));
    EXPECT_EQ(m.nonzeros[3], nnz(2, 2, 4.0));
}


struct dummy_distribution {
    template <typename RandomEngine>
    double operator()(RandomEngine &&) const
    {
        if (last >= 2.0) {
            last = 0.0;
        }
        return last = last + 1.0;
    }
    mutable double last{0};
};


TEST(MatrixData, InitializesGeneralMatrixWithConditionNumber)
{
    /*
     "Randomly" generated reflection vector:

        u = [ 1 2 ]

     Reflector:

        R = I - 2 (u* u) / (u u*)
          = 1/5 [  3 -4 ]
                [ -4 -3 ]

     Reflected matrix:

        M = R [ 2  0   ] R* = 1/25 [  6 -2   ] R* = 1/25 [  26 -18   ]
              [ 0  1/2 ]           [ -8 -3/2 ]           [ -18  73/2 ]
          = 1/100 [  104 -72 ]
                  [ -72  146 ]
     */
    using data = gko::matrix_data<double, int>;
    using nnz = data::nonzero_type;

    const auto m = data::cond(2, 4.0, dummy_distribution{}, std::ranlux48(42));

    ASSERT_EQ(m.size, gko::dim<2>(2, 2));
    ASSERT_EQ(m.nonzeros.size(), 4);
    EXPECT_NEAR(m.nonzeros[0].value, 1.04, 1e-15);
    EXPECT_NEAR(m.nonzeros[1].value, -0.72, 1e-15);
    EXPECT_NEAR(m.nonzeros[2].value, -0.72, 1e-15);
    EXPECT_NEAR(m.nonzeros[3].value, 1.46, 1e-15);
}


}  // namespace
