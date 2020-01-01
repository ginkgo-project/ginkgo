/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <core/test/utils/matrix_generator.hpp>


#include <gtest/gtest.h>


#include <cmath>
#include <random>


namespace {


class MatrixGenerator : public ::testing::Test {
protected:
    MatrixGenerator()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::generate_random_matrix(
              500, 100, std::normal_distribution<double>(50, 5),
              std::normal_distribution<double>(20.0, 5.0), std::ranlux48(42),
              exec)),
          l_mtx(gko::test::generate_random_lower_triangular_matrix(
              4, 3, true, std::normal_distribution<double>(50, 5),
              std::normal_distribution<double>(20.0, 5.0), std::ranlux48(42),
              exec)),
          u_mtx(gko::test::generate_random_upper_triangular_matrix(
              3, 4, true, std::normal_distribution<double>(50, 5),
              std::normal_distribution<double>(20.0, 5.0), std::ranlux48(42),
              exec)),
          nnz_per_row_sample(500, 0),
          values_sample(0)
    {
        // collect samples of nnz/row and values from the matrix
        for (int row = 0; row < mtx->get_size()[0]; ++row) {
            for (int col = 0; col < mtx->get_size()[1]; ++col) {
                auto val = mtx->at(row, col);
                if (val != 0.0) {
                    ++nnz_per_row_sample[row];
                    values_sample.push_back(val);
                }
            }
        }
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<>> mtx;
    std::unique_ptr<gko::matrix::Dense<>> l_mtx;
    std::unique_ptr<gko::matrix::Dense<>> u_mtx;
    std::vector<int> nnz_per_row_sample;
    std::vector<double> values_sample;

    template <typename InputIterator, typename ValueType>
    ValueType get_nth_moment(int n, ValueType c, InputIterator sample_start,
                             InputIterator sample_end)
    {
        using std::pow;
        ValueType res = 0;
        ValueType num_elems = 0;
        while (sample_start != sample_end) {
            auto tmp = *(sample_start++);
            res += pow(tmp - c, n);
            num_elems += 1;
        }
        return res / num_elems;
    }
};


TEST_F(MatrixGenerator, OutputHasCorrectSize)
{
    ASSERT_EQ(mtx->get_size(), gko::dim<2>(500, 100));
}


TEST_F(MatrixGenerator, OutputHasCorrectNonzeroAverageAndDeviation)
{
    using std::sqrt;
    auto average = get_nth_moment(1, 0.0, begin(nnz_per_row_sample),
                                  end(nnz_per_row_sample));
    auto deviation = sqrt(get_nth_moment(2, average, begin(nnz_per_row_sample),
                                         end(nnz_per_row_sample)));

    // check that average & deviation is within 10% of the required amount
    ASSERT_NEAR(average, 50.0, 5);
    ASSERT_NEAR(deviation, 5.0, 0.5);
}


TEST_F(MatrixGenerator, OutputHasCorrectValuesAverageAndDeviation)
{
    using std::sqrt;
    auto average =
        get_nth_moment(1, 0.0, begin(values_sample), end(values_sample));
    auto deviation = sqrt(
        get_nth_moment(2, average, begin(values_sample), end(values_sample)));

    // check that average and deviation is within 10% of the required amount
    ASSERT_NEAR(average, 20.0, 2.0);
    ASSERT_NEAR(deviation, 5.0, 0.5);
}


TEST_F(MatrixGenerator, CanGenerateLowerTriangularMatrixWithDiagonalOnes)
{
    ASSERT_EQ(l_mtx->at(0, 0), 1.0);
    ASSERT_EQ(l_mtx->at(0, 1), 0.0);
    ASSERT_EQ(l_mtx->at(0, 2), 0.0);
    ASSERT_NE(l_mtx->at(1, 0), 0.0);
    ASSERT_EQ(l_mtx->at(1, 1), 1.0);
    ASSERT_EQ(l_mtx->at(1, 2), 0.0);
    ASSERT_NE(l_mtx->at(2, 0), 0.0);
    ASSERT_NE(l_mtx->at(2, 1), 0.0);
    ASSERT_EQ(l_mtx->at(2, 2), 1.0);
    ASSERT_NE(l_mtx->at(3, 0), 0.0);
    ASSERT_NE(l_mtx->at(3, 1), 0.0);
    ASSERT_NE(l_mtx->at(3, 2), 0.0);
}


TEST_F(MatrixGenerator, CanGenerateUpperTriangularMatrixWithDiagonalOnes)
{
    ASSERT_EQ(u_mtx->at(0, 0), 1.0);
    ASSERT_NE(u_mtx->at(0, 1), 0.0);
    ASSERT_NE(u_mtx->at(0, 2), 0.0);
    ASSERT_NE(u_mtx->at(0, 3), 0.0);
    ASSERT_EQ(u_mtx->at(1, 0), 0.0);
    ASSERT_EQ(u_mtx->at(1, 1), 1.0);
    ASSERT_NE(u_mtx->at(1, 2), 0.0);
    ASSERT_NE(u_mtx->at(1, 3), 0.0);
    ASSERT_EQ(u_mtx->at(2, 0), 0.0);
    ASSERT_EQ(u_mtx->at(2, 1), 0.0);
    ASSERT_EQ(u_mtx->at(2, 2), 1.0);
    ASSERT_NE(u_mtx->at(2, 3), 0.0);
}


}  // namespace
