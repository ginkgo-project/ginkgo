/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/base/matrix_data.hpp>


#include <gtest/gtest.h>


#include <random>


namespace {


TEST(MatrixData, InitializesANullMatrix)
{
    gko::matrix_data<double, int> m;

    ASSERT_EQ(m.num_rows, 0);
    ASSERT_EQ(m.num_cols, 0);
    ASSERT_EQ(m.nonzeros.size(), 0);
}


TEST(MatrixData, InitializesWithZeros)
{
    gko::matrix_data<double, int> m(3, 5);

    ASSERT_EQ(m.num_rows, 3);
    ASSERT_EQ(m.num_cols, 5);
    ASSERT_EQ(m.nonzeros.size(), 0);
}


TEST(MatrixData, InitializesWithValue)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;
    gko::matrix_data<double, int> m(2, 3, 8.3);

    ASSERT_EQ(m.num_rows, 2);
    ASSERT_EQ(m.num_cols, 3);
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
        2, 3, std::uniform_real_distribution<double>(-1, 1), std::ranlux48(19));

    ASSERT_EQ(m.num_rows, 2);
    ASSERT_EQ(m.num_cols, 3);
    ASSERT_LE(m.nonzeros.size(), 6);
    for (const auto &elem : m.nonzeros) {
        EXPECT_TRUE(-1 <= std::get<2>(elem) && std::get<2>(elem) <= 1);
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

    ASSERT_EQ(m.num_rows, 2);
    ASSERT_EQ(m.num_cols, 3);
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

    ASSERT_EQ(m.num_rows, 1);
    ASSERT_EQ(m.num_cols, 3);
    ASSERT_EQ(m.nonzeros.size(), 3);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 2.0));
    EXPECT_EQ(m.nonzeros[1], nnz(0, 1, 3.0));
    EXPECT_EQ(m.nonzeros[2], nnz(0, 2, 5.0));
}


TEST(MatrixData, InitializesColumnVectorFromValueList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;
    gko::matrix_data<double, int> m{{2}, {3}, {5}};

    ASSERT_EQ(m.num_rows, 3);
    ASSERT_EQ(m.num_cols, 1);
    ASSERT_EQ(m.nonzeros.size(), 3);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 2.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 0, 3.0));
    EXPECT_EQ(m.nonzeros[2], nnz(2, 0, 5.0));
}


TEST(MatrixData, InitializesFromNonzeroList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;
    gko::matrix_data<double, int> m(5, 7, {{0, 0, 2}, {1, 1, 0}, {2, 3, 5}});

    ASSERT_EQ(m.num_rows, 5);
    ASSERT_EQ(m.num_cols, 7);
    ASSERT_EQ(m.nonzeros.size(), 3);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 2.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 1, 0.0));
    EXPECT_EQ(m.nonzeros[2], nnz(2, 3, 5.0));
}


TEST(MatrixData, InitializesDiagonalMatrix)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;
    const auto m = gko::matrix_data<double, int>::diag(2, 3, 5.0);

    ASSERT_EQ(m.num_rows, 2);
    ASSERT_EQ(m.num_cols, 3);
    ASSERT_EQ(m.nonzeros.size(), 2);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 5.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 1, 5.0));
}


TEST(MatrixData, InitializesDiagonalMatrixFromValueList)
{
    using nnz = gko::matrix_data<double, int>::nonzero_type;
    const auto m = gko::matrix_data<double, int>::diag(2, 3, {3, 5});

    ASSERT_EQ(m.num_rows, 2);
    ASSERT_EQ(m.num_cols, 3);
    ASSERT_EQ(m.nonzeros.size(), 2);
    EXPECT_EQ(m.nonzeros[0], nnz(0, 0, 3.0));
    EXPECT_EQ(m.nonzeros[1], nnz(1, 1, 5.0));
}


}  // namespace
