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

#include <ginkgo/core/matrix/ell.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <core/test/utils/assertions.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace {


class Ell : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Ell<>;
    using Csr = gko::matrix::Csr<>;
    using Vec = gko::matrix::Dense<>;

    Ell()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(Mtx::create(exec)),
          mtx2(Mtx::create(exec))
    {
        // clang-format off
        mtx1 = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec);
        mtx2 = gko::initialize<Mtx>(
            {{1.0, 3.0, 2.0},
             {0.0, 5.0, 0.0}}, exec, gko::dim<2>{}, 0, 16);
        // clang-format on
    }

    void assert_equal_to_mtx(const Csr *m)
    {
        auto v = m->get_const_values();
        auto c = m->get_const_col_idxs();
        auto r = m->get_const_row_ptrs();

        ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
        ASSERT_EQ(m->get_num_stored_elements(), 4);
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

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
};


TEST_F(Ell, AppliesToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx1->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TEST_F(Ell, AppliesToDenseMatrix)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{2});

    mtx1->apply(x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{13.0,  3.5},
                       { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TEST_F(Ell, AppliesLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx1->apply(alpha.get(), x.get(), beta.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TEST_F(Ell, AppliesLinearCombinationToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>(
        {{1.0, 0.5},
         {2.0, -1.5}}, exec);
    // clang-format on

    mtx1->apply(alpha.get(), x.get(), beta.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{-11.0, -2.5},
                       { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TEST_F(Ell, ApplyFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx1->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Ell, ApplyFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx1->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Ell, ApplyFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx1->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Ell, ConvertsToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx1->get_executor());

    mtx1->convert_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Ell, MovesToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx1->get_executor());

    mtx1->move_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Ell, AppliesWithStrideToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx2->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TEST_F(Ell, AppliesWithStrideToDenseMatrix)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{2});

    mtx2->apply(x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{13.0, 3.5},
                       {5.0, -7.5}}), 0.0);
    // clang-format on
}


TEST_F(Ell, AppliesWithStrideLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TEST_F(Ell, AppliesWithStrideLinearCombinationToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>(
        {{1.0, 0.5},
         {2.0, -1.5}}, exec);
    // clang-format on

    mtx2->apply(alpha.get(), x.get(), beta.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{-11.0, -2.5},
                       {-1.0, 4.5}}), 0.0);
    // clang-format on
}


TEST_F(Ell, ApplyWithStrideFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx2->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Ell, ApplyWithStrideFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx2->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Ell, ApplyWithStrideFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3}, 2);
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx2->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Ell, ConvertsWithStrideToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx2->get_executor());
    // clang-format off
    auto dense_other = gko::initialize<gko::matrix::Dense<>>(
        4, {{1.0, 3.0, 2.0},
            {0.0, 5.0, 0.0}}, exec);
    // clang-format on

    mtx2->convert_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Ell, MovesWithStrideToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx2->get_executor());

    mtx2->move_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Ell, ConvertsToCsr)
{
    auto csr_s_classical = std::make_shared<gko::matrix::Csr<>::classical>();
    auto csr_s_merge = std::make_shared<gko::matrix::Csr<>::merge_path>();
    auto csr_mtx_c =
        gko::matrix::Csr<>::create(mtx1->get_executor(), csr_s_classical);
    auto csr_mtx_m =
        gko::matrix::Csr<>::create(mtx1->get_executor(), csr_s_merge);

    mtx1->convert_to(csr_mtx_c.get());
    mtx1->convert_to(csr_mtx_m.get());

    assert_equal_to_mtx(csr_mtx_c.get());
    assert_equal_to_mtx(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy(), csr_s_classical);
    ASSERT_EQ(csr_mtx_m->get_strategy(), csr_s_merge);
}


TEST_F(Ell, ConvertsWithStrideToCsr)
{
    auto csr_s_classical = std::make_shared<gko::matrix::Csr<>::classical>();
    auto csr_s_merge = std::make_shared<gko::matrix::Csr<>::merge_path>();
    auto csr_mtx_c =
        gko::matrix::Csr<>::create(mtx2->get_executor(), csr_s_classical);
    auto csr_mtx_m =
        gko::matrix::Csr<>::create(mtx2->get_executor(), csr_s_merge);
    auto mtx_clone = mtx2->clone();

    mtx2->move_to(csr_mtx_c.get());
    mtx_clone->move_to(csr_mtx_m.get());

    assert_equal_to_mtx(csr_mtx_c.get());
    assert_equal_to_mtx(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy(), csr_s_classical);
    ASSERT_EQ(csr_mtx_m->get_strategy(), csr_s_merge);
}


}  // namespace
