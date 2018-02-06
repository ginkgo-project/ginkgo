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

#include <core/matrix/dense.hpp>

#include <core/test/utils/assertions.hpp>

#include <gtest/gtest.h>

#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/csr.hpp>
#include <core/matrix/sliced_ell.hpp>

#include <complex>

namespace {


class Dense : public ::testing::Test {
protected:
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::matrix::Dense<>::create(
              exec, 4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}})),
          mtx2(gko::matrix::Dense<>::create(exec, {{1.0, -1.0}, {-2.0, 2.0}})),
          mtx3(gko::matrix::Dense<>::create(
              exec, 4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}})),
          mtx4(gko::matrix::Dense<>::create(
              exec, 4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}})),
          mtx5(gko::matrix::Dense<>::create(
              exec, {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}})),
          mtx6(gko::matrix::Dense<std::complex<double>>::create(
              exec, {{1.0 + 2.0 * i, -1.0 + 2.1 * i},
                     {-2.0 + 1.5 * i, 4.5 + 0.0 * i},
                     {1.0 + 0.0 * i, i}}))
    {}

    std::complex<double> i{0, 1};
    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<>> mtx1;
    std::unique_ptr<gko::matrix::Dense<>> mtx2;
    std::unique_ptr<gko::matrix::Dense<>> mtx3;
    std::unique_ptr<gko::matrix::Dense<>> mtx4;
    std::unique_ptr<gko::matrix::Dense<>> mtx5;
    std::unique_ptr<gko::matrix::Dense<std::complex<double>>> mtx6;
};


TEST_F(Dense, AppliesToDense)
{
    mtx2->apply(mtx1.get(), mtx3.get());

    EXPECT_EQ(mtx3->at(0, 0), -0.5);
    EXPECT_EQ(mtx3->at(0, 1), -0.5);
    EXPECT_EQ(mtx3->at(0, 2), -0.5);
    EXPECT_EQ(mtx3->at(1, 0), 1.0);
    EXPECT_EQ(mtx3->at(1, 1), 1.0);
    ASSERT_EQ(mtx3->at(1, 2), 1.0);
}


TEST_F(Dense, AppliesLinearCombinationToDense)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {-1.0});
    auto beta = gko::matrix::Dense<>::create(exec, {2.0});

    mtx2->apply(alpha.get(), mtx1.get(), beta.get(), mtx3.get());

    EXPECT_EQ(mtx3->at(0, 0), 2.5);
    EXPECT_EQ(mtx3->at(0, 1), 4.5);
    EXPECT_EQ(mtx3->at(0, 2), 6.5);
    EXPECT_EQ(mtx3->at(1, 0), 0.0);
    EXPECT_EQ(mtx3->at(1, 1), 2.0);
    ASSERT_EQ(mtx3->at(1, 2), 4.0);
}


TEST_F(Dense, ApplyFailsOnWrongInnerDimension)
{
    auto res = gko::matrix::Dense<>::create(exec, 2, 2, 2);

    ASSERT_THROW(mtx2->apply(mtx1.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ApplyFailsOnWrongNumberOfRows)
{
    auto res = gko::matrix::Dense<>::create(exec, 3, 3, 3);

    ASSERT_THROW(mtx1->apply(mtx2.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ApplyFailsOnWrongNumberOfCols)
{
    auto res = gko::matrix::Dense<>::create(exec, 2, 2, 3);

    ASSERT_THROW(mtx1->apply(mtx2.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ScalesData)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {{2.0, -2.0}});

    mtx2->scale(alpha.get());

    EXPECT_EQ(mtx2->at(0, 0), 2.0);
    EXPECT_EQ(mtx2->at(0, 1), 2.0);
    EXPECT_EQ(mtx2->at(1, 0), -4.0);
    EXPECT_EQ(mtx2->at(1, 1), -4.0);
}


TEST_F(Dense, ScalesDataWithScalar)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {2.0});

    mtx2->scale(alpha.get());

    EXPECT_EQ(mtx2->at(0, 0), 2.0);
    EXPECT_EQ(mtx2->at(0, 1), -2.0);
    EXPECT_EQ(mtx2->at(1, 0), -4.0);
    EXPECT_EQ(mtx2->at(1, 1), 4.0);
}


TEST_F(Dense, ScalesDataWithPadding)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {{-1.0, 1.0, 2.0}});

    mtx1->scale(alpha.get());

    EXPECT_EQ(mtx1->at(0, 0), -1.0);
    EXPECT_EQ(mtx1->at(0, 1), 2.0);
    EXPECT_EQ(mtx1->at(0, 2), 6.0);
    EXPECT_EQ(mtx1->at(1, 0), -1.5);
    EXPECT_EQ(mtx1->at(1, 1), 2.5);
    ASSERT_EQ(mtx1->at(1, 2), 7.0);
}


TEST_F(Dense, AddsScaled)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {{2.0, 1.0, -2.0}});

    mtx1->add_scaled(alpha.get(), mtx3.get());

    EXPECT_EQ(mtx1->at(0, 0), 3.0);
    EXPECT_EQ(mtx1->at(0, 1), 4.0);
    EXPECT_EQ(mtx1->at(0, 2), -3.0);
    EXPECT_EQ(mtx1->at(1, 0), 2.5);
    EXPECT_EQ(mtx1->at(1, 1), 4.0);
    ASSERT_EQ(mtx1->at(1, 2), -1.5);
}


TEST_F(Dense, AddsScaledWithScalar)
{
    auto alpha = gko::matrix::Dense<>::create(exec, {2.0});

    mtx1->add_scaled(alpha.get(), mtx3.get());

    EXPECT_EQ(mtx1->at(0, 0), 3.0);
    EXPECT_EQ(mtx1->at(0, 1), 6.0);
    EXPECT_EQ(mtx1->at(0, 2), 9.0);
    EXPECT_EQ(mtx1->at(1, 0), 2.5);
    EXPECT_EQ(mtx1->at(1, 1), 5.5);
    ASSERT_EQ(mtx1->at(1, 2), 8.5);
}


TEST_F(Dense, AddScaledFailsOnWrongSizes)
{
    auto alpha = gko::matrix::Dense<>::create(exec, 1, 2, 2);

    ASSERT_THROW(mtx1->add_scaled(alpha.get(), mtx2.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ComputesDot)
{
    auto result = gko::matrix::Dense<>::create(exec, 1, 3, 3);

    mtx1->compute_dot(mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), 1.75);
    EXPECT_EQ(result->at(0, 1), 7.75);
    ASSERT_EQ(result->at(0, 2), 17.75);
}


TEST_F(Dense, ComputDotFailsOnWrongInputSize)
{
    auto result = gko::matrix::Dense<>::create(exec, 1, 3, 3);

    ASSERT_THROW(mtx1->compute_dot(mtx2.get(), result.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ComputDotFailsOnWrongResultSize)
{
    auto result = gko::matrix::Dense<>::create(exec, 1, 2, 2);

    ASSERT_THROW(mtx1->compute_dot(mtx3.get(), result.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ConvertsToCsr)
{
    auto csr_mtx = gko::matrix::Csr<>::create(mtx4->get_executor());

    mtx4->convert_to(csr_mtx.get());

    auto v = csr_mtx->get_const_values();
    auto c = csr_mtx->get_const_col_idxs();
    auto r = csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(csr_mtx->get_num_rows(), 2);
    ASSERT_EQ(csr_mtx->get_num_cols(), 3);
    ASSERT_EQ(csr_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 3.0);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 5.0);
}


TEST_F(Dense, MovesToCsr)
{
    auto csr_mtx = gko::matrix::Csr<>::create(mtx4->get_executor());

    mtx4->move_to(csr_mtx.get());

    auto v = csr_mtx->get_const_values();
    auto c = csr_mtx->get_const_col_idxs();
    auto r = csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(csr_mtx->get_num_rows(), 2);
    ASSERT_EQ(csr_mtx->get_num_cols(), 3);
    ASSERT_EQ(csr_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 3);
    EXPECT_EQ(r[2], 4);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 3.0);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 5.0);
}


TEST_F(Dense, ConvertsToSliced_ell)
{
    auto sliced_ell_mtx = gko::matrix::Sliced_ell<>::create(mtx4->get_executor());

    mtx4->convert_to(sliced_ell_mtx.get());

    auto v = sliced_ell_mtx->get_const_values();
    auto c = sliced_ell_mtx->get_const_col_idxs();
    auto l = sliced_ell_mtx->get_const_slice_lens();
    auto s = sliced_ell_mtx->get_const_slice_sets();

    ASSERT_EQ(sliced_ell_mtx->get_num_rows(), 2);
    ASSERT_EQ(sliced_ell_mtx->get_num_cols(), 3);
    ASSERT_EQ(sliced_ell_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(l[0], 3);
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[default_slice_size], 1);
    EXPECT_EQ(c[default_slice_size+1], 1);
    EXPECT_EQ(c[2*default_slice_size], 2);
    EXPECT_EQ(c[2*default_slice_size+1], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[default_slice_size], 3.0);
    EXPECT_EQ(v[default_slice_size+1], 0.0);
    EXPECT_EQ(v[2*default_slice_size], 2.0);
    EXPECT_EQ(v[2*default_slice_size+1], 0.0);
}


TEST_F(Dense, MovesToSliced_ell)
{
    auto sliced_ell_mtx = gko::matrix::Sliced_ell<>::create(mtx4->get_executor());

    mtx4->move_to(sliced_ell_mtx.get());

    auto v = sliced_ell_mtx->get_const_values();
    auto c = sliced_ell_mtx->get_const_col_idxs();
    auto l = sliced_ell_mtx->get_const_slice_lens();
    auto s = sliced_ell_mtx->get_const_slice_sets();

    ASSERT_EQ(sliced_ell_mtx->get_num_rows(), 2);
    ASSERT_EQ(sliced_ell_mtx->get_num_cols(), 3);
    ASSERT_EQ(sliced_ell_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(l[0], 3);
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[default_slice_size], 1);
    EXPECT_EQ(c[default_slice_size+1], 1);
    EXPECT_EQ(c[2*default_slice_size], 2);
    EXPECT_EQ(c[2*default_slice_size+1], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[default_slice_size], 3.0);
    EXPECT_EQ(v[default_slice_size+1], 0.0);
    EXPECT_EQ(v[2*default_slice_size], 2.0);
    EXPECT_EQ(v[2*default_slice_size+1], 0.0);
}


TEST_F(Dense, SquareMatrixIsTransposable)
{
    auto trans = mtx5->transpose();
    auto trans_as_dense = static_cast<gko::matrix::Dense<> *>(trans.get());

    ASSERT_MTX_NEAR(trans_as_dense,
                    l({{1.0, -2.0, 2.1}, {-1.0, 2.0, 3.4}, {-0.5, 4.5, 1.2}}),
                    0.0);
}


TEST_F(Dense, NonSquareMatrixIsTransposable)
{
    auto trans = mtx4->transpose();
    auto trans_as_dense = static_cast<gko::matrix::Dense<> *>(trans.get());

    ASSERT_MTX_NEAR(trans_as_dense, l({{1.0, 0.0}, {3.0, 5.0}, {2.0, 0.0}}),
                    0.0);
}


TEST_F(Dense, NonSquareMatrixIsConjugateTransposable)
{
    auto trans = mtx6->conj_transpose();
    auto trans_as_dense =
        static_cast<gko::matrix::Dense<std::complex<double>> *>(trans.get());

    ASSERT_MTX_NEAR(trans_as_dense,
                    l({{1.0 - 2.0 * i, -2.0 - 1.5 * i, 1.0 + 0.0 * i},
                       {-1.0 - 2.1 * i, 4.5 + 0.0 * i, -i}}),
                    0.0);
}


}  // namespace
