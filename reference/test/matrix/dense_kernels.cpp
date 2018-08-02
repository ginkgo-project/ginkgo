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


#include <complex>


#include <gtest/gtest.h>


#include <core/base/exception.hpp>
#include <core/base/executor.hpp>
#include <core/matrix/coo.hpp>
#include <core/matrix/csr.hpp>
#include <core/matrix/ell.hpp>
#include <core/matrix/hybrid.hpp>
#include <core/matrix/sellp.hpp>
#include <core/test/utils/assertions.hpp>


namespace {


class Dense : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    Dense()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {1.5, 2.5, 3.5}},
                                    exec)),
          mtx2(gko::initialize<Mtx>({{1.0, -1.0}, {-2.0, 2.0}}, exec)),
          mtx3(gko::initialize<Mtx>(4, {{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}},
                                    exec)),
          mtx4(gko::initialize<Mtx>(4, {{1.0, 3.0, 2.0}, {0.0, 5.0, 0.0}},
                                    exec)),
          mtx5(gko::initialize<Mtx>(
              {{1.0, -1.0, -0.5}, {-2.0, 2.0, 4.5}, {2.1, 3.4, 1.2}}, exec)),
          mtx6(gko::initialize<gko::matrix::Dense<std::complex<double>>>(
              {{1.0 + 2.0 * i, -1.0 + 2.1 * i},
               {-2.0 + 1.5 * i, 4.5 + 0.0 * i},
               {1.0 + 0.0 * i, i}},
              exec)),
          mtx7(gko::initialize<Mtx>({{1.0, 2.0, 0.0}, {0.0, 1.5, 0.0}}, exec)),
          mtx8(gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.0, 1.5, 0.0}}, exec))
    {}

    std::complex<double> i{0, 1};
    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<gko::matrix::Dense<>> mtx1;
    std::unique_ptr<gko::matrix::Dense<>> mtx2;
    std::unique_ptr<gko::matrix::Dense<>> mtx3;
    std::unique_ptr<gko::matrix::Dense<>> mtx4;
    std::unique_ptr<gko::matrix::Dense<>> mtx5;
    std::unique_ptr<gko::matrix::Dense<std::complex<double>>> mtx6;
    std::unique_ptr<gko::matrix::Dense<>> mtx7;
    std::unique_ptr<gko::matrix::Dense<>> mtx8;
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
    auto alpha = gko::initialize<Mtx>({-1.0}, exec);
    auto beta = gko::initialize<Mtx>({2.0}, exec);

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
    auto res = gko::matrix::Dense<>::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx2->apply(mtx1.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ApplyFailsOnWrongNumberOfRows)
{
    auto res = gko::matrix::Dense<>::create(exec, gko::dim<2>{3});

    ASSERT_THROW(mtx1->apply(mtx2.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ApplyFailsOnWrongNumberOfCols)
{
    auto res = gko::matrix::Dense<>::create(exec, gko::dim<2>{2}, 3);

    ASSERT_THROW(mtx1->apply(mtx2.get(), res.get()), gko::DimensionMismatch);
}


TEST_F(Dense, ScalesData)
{
    auto alpha = gko::initialize<Mtx>({{2.0, -2.0}}, exec);

    mtx2->scale(alpha.get());

    EXPECT_EQ(mtx2->at(0, 0), 2.0);
    EXPECT_EQ(mtx2->at(0, 1), 2.0);
    EXPECT_EQ(mtx2->at(1, 0), -4.0);
    EXPECT_EQ(mtx2->at(1, 1), -4.0);
}


TEST_F(Dense, ScalesDataWithScalar)
{
    auto alpha = gko::initialize<Mtx>({2.0}, exec);

    mtx2->scale(alpha.get());

    EXPECT_EQ(mtx2->at(0, 0), 2.0);
    EXPECT_EQ(mtx2->at(0, 1), -2.0);
    EXPECT_EQ(mtx2->at(1, 0), -4.0);
    EXPECT_EQ(mtx2->at(1, 1), 4.0);
}


TEST_F(Dense, ScalesDataWithStride)
{
    auto alpha = gko::initialize<Mtx>({{-1.0, 1.0, 2.0}}, exec);

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
    auto alpha = gko::initialize<Mtx>({{2.0, 1.0, -2.0}}, exec);

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
    auto alpha = gko::initialize<Mtx>({2.0}, exec);

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
    auto alpha = gko::matrix::Dense<>::create(exec, gko::dim<2>{1, 2});

    ASSERT_THROW(mtx1->add_scaled(alpha.get(), mtx2.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ComputesDot)
{
    auto result = gko::matrix::Dense<>::create(exec, gko::dim<2>{1, 3});

    mtx1->compute_dot(mtx3.get(), result.get());

    EXPECT_EQ(result->at(0, 0), 1.75);
    EXPECT_EQ(result->at(0, 1), 7.75);
    ASSERT_EQ(result->at(0, 2), 17.75);
}


TEST_F(Dense, ComputDotFailsOnWrongInputSize)
{
    auto result = gko::matrix::Dense<>::create(exec, gko::dim<2>{1, 3});

    ASSERT_THROW(mtx1->compute_dot(mtx2.get(), result.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ComputDotFailsOnWrongResultSize)
{
    auto result = gko::matrix::Dense<>::create(exec, gko::dim<2>{1, 2});

    ASSERT_THROW(mtx1->compute_dot(mtx3.get(), result.get()),
                 gko::DimensionMismatch);
}


TEST_F(Dense, ConvertsToCoo)
{
    auto coo_mtx = gko::matrix::Coo<>::create(mtx4->get_executor());

    mtx4->convert_to(coo_mtx.get());

    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 3.0);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 5.0);
}


TEST_F(Dense, MovesToCoo)
{
    auto coo_mtx = gko::matrix::Coo<>::create(mtx4->get_executor());

    mtx4->move_to(coo_mtx.get());

    auto v = coo_mtx->get_const_values();
    auto c = coo_mtx->get_const_col_idxs();
    auto r = coo_mtx->get_const_row_idxs();

    ASSERT_EQ(coo_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(coo_mtx->get_num_stored_elements(), 4);
    EXPECT_EQ(r[0], 0);
    EXPECT_EQ(r[1], 0);
    EXPECT_EQ(r[2], 0);
    EXPECT_EQ(r[3], 1);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 2);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 3.0);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 5.0);
}


TEST_F(Dense, ConvertsToCsr)
{
    auto csr_mtx = gko::matrix::Csr<>::create(mtx4->get_executor());

    mtx4->convert_to(csr_mtx.get());

    auto v = csr_mtx->get_const_values();
    auto c = csr_mtx->get_const_col_idxs();
    auto r = csr_mtx->get_const_row_ptrs();

    ASSERT_EQ(csr_mtx->get_size(), gko::dim<2>(2, 3));
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

    ASSERT_EQ(csr_mtx->get_size(), gko::dim<2>(2, 3));
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


TEST_F(Dense, ConvertsToEll)
{
    auto ell_mtx = gko::matrix::Ell<>::create(mtx7->get_executor());

    mtx7->convert_to(ell_mtx.get());

    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 0.0);
}


TEST_F(Dense, MovesToEll)
{
    auto ell_mtx = gko::matrix::Ell<>::create(mtx7->get_executor());

    mtx7->move_to(ell_mtx.get());

    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 4);
    ASSERT_EQ(ell_mtx->get_stride(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 0.0);
}


TEST_F(Dense, ConvertsToEllWithStride)
{
    auto ell_mtx =
        gko::matrix::Ell<>::create(mtx7->get_executor(), gko::dim<2>{}, 0, 3);

    mtx7->convert_to(ell_mtx.get());

    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[2], 0.0);
    EXPECT_EQ(v[3], 2.0);
    EXPECT_EQ(v[4], 0.0);
    EXPECT_EQ(v[5], 0.0);
}


TEST_F(Dense, MovesToEllWithStride)
{
    auto ell_mtx =
        gko::matrix::Ell<>::create(mtx7->get_executor(), gko::dim<2>{}, 0, 3);

    mtx7->move_to(ell_mtx.get());

    auto v = ell_mtx->get_const_values();
    auto c = ell_mtx->get_const_col_idxs();

    ASSERT_EQ(ell_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(ell_mtx->get_num_stored_elements_per_row(), 2);
    ASSERT_EQ(ell_mtx->get_num_stored_elements(), 6);
    ASSERT_EQ(ell_mtx->get_stride(), 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[2], 0.0);
    EXPECT_EQ(v[3], 2.0);
    EXPECT_EQ(v[4], 0.0);
    EXPECT_EQ(v[5], 0.0);
}


TEST_F(Dense, MovesToHybridAutomatically)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(mtx4->get_executor());

    mtx4->move_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 0);
    EXPECT_EQ(n, 3);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 3.0);
    EXPECT_EQ(v[3], 0.0);
    EXPECT_EQ(v[4], 2.0);
    EXPECT_EQ(v[5], 0.0);
}


TEST_F(Dense, ConvertsToHybridAutomatically)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(mtx4->get_executor());

    mtx4->convert_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 0);
    EXPECT_EQ(n, 3);
    EXPECT_EQ(p, 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 3.0);
    EXPECT_EQ(v[3], 0.0);
    EXPECT_EQ(v[4], 2.0);
    EXPECT_EQ(v[5], 0.0);
}


TEST_F(Dense, MovesToHybridWithStrideAutomatically)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(mtx4->get_executor(),
                                                    gko::dim<2>{}, 0, 3);

    mtx4->move_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 9);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 0);
    EXPECT_EQ(n, 3);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(c[6], 2);
    EXPECT_EQ(c[7], 0);
    EXPECT_EQ(c[8], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 0.0);
    EXPECT_EQ(v[3], 3.0);
    EXPECT_EQ(v[4], 0.0);
    EXPECT_EQ(v[5], 0.0);
    EXPECT_EQ(v[6], 2.0);
    EXPECT_EQ(v[7], 0.0);
    EXPECT_EQ(v[8], 0.0);
}


TEST_F(Dense, ConvertsToHybridWithStrideAutomatically)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(mtx4->get_executor(),
                                                    gko::dim<2>{}, 0, 3);

    mtx4->convert_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 9);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 0);
    EXPECT_EQ(n, 3);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(c[6], 2);
    EXPECT_EQ(c[7], 0);
    EXPECT_EQ(c[8], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 0.0);
    EXPECT_EQ(v[3], 3.0);
    EXPECT_EQ(v[4], 0.0);
    EXPECT_EQ(v[5], 0.0);
    EXPECT_EQ(v[6], 2.0);
    EXPECT_EQ(v[7], 0.0);
    EXPECT_EQ(v[8], 0.0);
}


TEST_F(Dense, MovesToHybridWithStrideAndCooLengthByColumns2)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(
        mtx4->get_executor(), gko::dim<2>{}, 0, 3, 3,
        std::make_shared<gko::matrix::Hybrid<>::column_limit>(2));

    mtx4->move_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 3);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 0.0);
    EXPECT_EQ(v[3], 3.0);
    EXPECT_EQ(v[4], 0.0);
    EXPECT_EQ(v[5], 0.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[0], 2.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[1], 0.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[2], 0.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[0], 2);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[2], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[0], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[2], 0);
}


TEST_F(Dense, ConvertsToHybridWithStrideAndCooLengthByColumns2)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(
        mtx4->get_executor(), gko::dim<2>{}, 0, 3, 3,
        std::make_shared<gko::matrix::Hybrid<>::column_limit>(2));

    mtx4->convert_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 6);
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 3);
    EXPECT_EQ(n, 2);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(c[3], 1);
    EXPECT_EQ(c[4], 0);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 0.0);
    EXPECT_EQ(v[3], 3.0);
    EXPECT_EQ(v[4], 0.0);
    EXPECT_EQ(v[5], 0.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[0], 2.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[1], 0.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_values()[2], 0.0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[0], 2);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_col_idxs()[2], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[0], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[1], 0);
    EXPECT_EQ(hybrid_mtx->get_const_coo_row_idxs()[2], 0);
}


TEST_F(Dense, MovesToHybridWithStrideByPercent40)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(
        mtx4->get_executor(), gko::dim<2>{}, 0, 3,
        std::make_shared<gko::matrix::Hybrid<>::imbalance_limit>(0.4));

    mtx4->move_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 3);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 0.0);

    auto coo_v = hybrid_mtx->get_const_coo_values();
    auto coo_c = hybrid_mtx->get_const_coo_col_idxs();
    auto coo_r = hybrid_mtx->get_const_coo_row_idxs();
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], 3.0);
    EXPECT_EQ(coo_v[1], 2.0);
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TEST_F(Dense, ConvertsToHybridWithStrideByPercent40)
{
    auto hybrid_mtx = gko::matrix::Hybrid<>::create(
        mtx4->get_executor(), gko::dim<2>{}, 0, 3,
        std::make_shared<gko::matrix::Hybrid<>::imbalance_limit>(0.4));

    mtx4->convert_to(hybrid_mtx.get());

    auto v = hybrid_mtx->get_const_ell_values();
    auto c = hybrid_mtx->get_const_ell_col_idxs();
    auto n = hybrid_mtx->get_ell_num_stored_elements_per_row();
    auto p = hybrid_mtx->get_ell_stride();
    ASSERT_EQ(hybrid_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(hybrid_mtx->get_ell_num_stored_elements(), 3);
    EXPECT_EQ(n, 1);
    EXPECT_EQ(p, 3);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 5.0);
    EXPECT_EQ(v[2], 0.0);

    auto coo_v = hybrid_mtx->get_const_coo_values();
    auto coo_c = hybrid_mtx->get_const_coo_col_idxs();
    auto coo_r = hybrid_mtx->get_const_coo_row_idxs();
    ASSERT_EQ(hybrid_mtx->get_coo_num_stored_elements(), 2);
    EXPECT_EQ(coo_v[0], 3.0);
    EXPECT_EQ(coo_v[1], 2.0);
    EXPECT_EQ(coo_c[0], 1);
    EXPECT_EQ(coo_c[1], 2);
    EXPECT_EQ(coo_r[0], 0);
    EXPECT_EQ(coo_r[1], 0);
}


TEST_F(Dense, ConvertsToSellp)
{
    auto sellp_mtx = gko::matrix::Sellp<>::create(mtx8->get_executor());

    mtx8->convert_to(sellp_mtx.get());

    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[gko::matrix::default_slice_size], 2.0);
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], 0.0);
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], 3.0);
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], 0.0);
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TEST_F(Dense, MovesToSellp)
{
    auto sellp_mtx = gko::matrix::Sellp<>::create(mtx8->get_executor());

    mtx8->move_to(sellp_mtx.get());

    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 3);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(),
              3 * gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_slice_size(), gko::matrix::default_slice_size);
    ASSERT_EQ(sellp_mtx->get_stride_factor(),
              gko::matrix::default_stride_factor);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size], 1);
    EXPECT_EQ(c[gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size], 2);
    EXPECT_EQ(c[2 * gko::matrix::default_slice_size + 1], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[gko::matrix::default_slice_size], 2.0);
    EXPECT_EQ(v[gko::matrix::default_slice_size + 1], 0.0);
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size], 3.0);
    EXPECT_EQ(v[2 * gko::matrix::default_slice_size + 1], 0.0);
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 3);
    EXPECT_EQ(l[0], 3);
}


TEST_F(Dense, ConvertsToSellpWithSliceSizeAndStrideFactor)
{
    auto sellp_mtx = gko::matrix::Sellp<>::create(mtx8->get_executor(),
                                                  gko::dim<2>{}, 2, 2, 0);

    mtx8->convert_to(sellp_mtx.get());

    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 4);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(), 8);
    ASSERT_EQ(sellp_mtx->get_slice_size(), 2);
    ASSERT_EQ(sellp_mtx->get_stride_factor(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(c[6], 0);
    EXPECT_EQ(c[7], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 0.0);
    EXPECT_EQ(v[4], 3.0);
    EXPECT_EQ(v[5], 0.0);
    EXPECT_EQ(v[6], 0.0);
    EXPECT_EQ(v[7], 0.0);
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(l[0], 4);
}


TEST_F(Dense, MovesToSellpWithSliceSizeAndStrideFactor)
{
    auto sellp_mtx = gko::matrix::Sellp<>::create(mtx8->get_executor(),
                                                  gko::dim<2>{}, 2, 2, 0);

    mtx8->move_to(sellp_mtx.get());

    auto v = sellp_mtx->get_const_values();
    auto c = sellp_mtx->get_const_col_idxs();
    auto s = sellp_mtx->get_const_slice_sets();
    auto l = sellp_mtx->get_const_slice_lengths();

    ASSERT_EQ(sellp_mtx->get_size(), gko::dim<2>(2, 3));
    ASSERT_EQ(sellp_mtx->get_total_cols(), 4);
    ASSERT_EQ(sellp_mtx->get_num_stored_elements(), 8);
    ASSERT_EQ(sellp_mtx->get_slice_size(), 2);
    ASSERT_EQ(sellp_mtx->get_stride_factor(), 2);
    EXPECT_EQ(c[0], 0);
    EXPECT_EQ(c[1], 1);
    EXPECT_EQ(c[2], 1);
    EXPECT_EQ(c[3], 0);
    EXPECT_EQ(c[4], 2);
    EXPECT_EQ(c[5], 0);
    EXPECT_EQ(c[6], 0);
    EXPECT_EQ(c[7], 0);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.5);
    EXPECT_EQ(v[2], 2.0);
    EXPECT_EQ(v[3], 0.0);
    EXPECT_EQ(v[4], 3.0);
    EXPECT_EQ(v[5], 0.0);
    EXPECT_EQ(v[6], 0.0);
    EXPECT_EQ(v[7], 0.0);
    EXPECT_EQ(s[0], 0);
    EXPECT_EQ(s[1], 4);
    EXPECT_EQ(l[0], 4);
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
