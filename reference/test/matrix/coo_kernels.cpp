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

#include "core/matrix/coo_kernels.hpp"


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class Coo : public ::testing::Test {
protected:
    using Csr = gko::matrix::Csr<>;
    using Mtx = gko::matrix::Coo<>;
    using Vec = gko::matrix::Dense<>;

    Coo() : exec(gko::ReferenceExecutor::create()), mtx(Mtx::create(exec))
    {
        // clang-format off
        mtx = gko::initialize<Mtx>({{1.0, 3.0, 2.0},
                                     {0.0, 5.0, 0.0}}, exec);
        // clang-format on
    }

    void assert_equal_to_mtx_in_csr_format(const Csr *m)
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
    std::unique_ptr<Mtx> mtx;
};


TEST_F(Coo, ConvertsToCsr)
{
    auto csr_s_classical = std::make_shared<gko::matrix::Csr<>::classical>();
    auto csr_s_merge = std::make_shared<gko::matrix::Csr<>::merge_path>();
    auto csr_mtx_c =
        gko::matrix::Csr<>::create(mtx->get_executor(), csr_s_classical);
    auto csr_mtx_m =
        gko::matrix::Csr<>::create(mtx->get_executor(), csr_s_merge);

    mtx->convert_to(csr_mtx_c.get());
    mtx->convert_to(csr_mtx_m.get());

    assert_equal_to_mtx_in_csr_format(csr_mtx_c.get());
    assert_equal_to_mtx_in_csr_format(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy(), csr_s_classical);
    ASSERT_EQ(csr_mtx_m->get_strategy(), csr_s_merge);
}


TEST_F(Coo, MovesToCsr)
{
    auto csr_s_classical = std::make_shared<gko::matrix::Csr<>::classical>();
    auto csr_s_merge = std::make_shared<gko::matrix::Csr<>::merge_path>();
    auto csr_mtx_c =
        gko::matrix::Csr<>::create(mtx->get_executor(), csr_s_classical);
    auto csr_mtx_m =
        gko::matrix::Csr<>::create(mtx->get_executor(), csr_s_merge);
    auto mtx_clone = mtx->clone();

    mtx->move_to(csr_mtx_c.get());
    mtx_clone->move_to(csr_mtx_m.get());

    assert_equal_to_mtx_in_csr_format(csr_mtx_c.get());
    assert_equal_to_mtx_in_csr_format(csr_mtx_m.get());
    ASSERT_EQ(csr_mtx_c->get_strategy(), csr_s_classical);
    ASSERT_EQ(csr_mtx_m->get_strategy(), csr_s_merge);
}


TEST_F(Coo, ConvertsToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());

    mtx->convert_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Coo, MovesToDense)
{
    auto dense_mtx = gko::matrix::Dense<>::create(mtx->get_executor());

    mtx->move_to(dense_mtx.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(dense_mtx,
                    l({{1.0, 3.0, 2.0},
                       {0.0, 5.0, 0.0}}), 0.0);
    // clang-format on
}


TEST_F(Coo, AppliesToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = Vec::create(exec, gko::dim<2>{2, 1});

    mtx->apply(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({13.0, 5.0}), 0.0);
}


TEST_F(Coo, AppliesToDenseMatrix)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    // clang-format on
    auto y = Vec::create(exec, gko::dim<2>{2, 2});

    mtx->apply(x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{13.0,  3.5},
                       { 5.0, -7.5}}), 0.0);
    // clang-format on
}


TEST_F(Coo, AppliesLinearCombinationToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({-11.0, -1.0}), 0.0);
}


TEST_F(Coo, AppliesLinearCombinationToDenseMatrix)
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

    mtx->apply(alpha.get(), x.get(), beta.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{-11.0, -2.5},
                       { -1.0,  4.5}}), 0.0);
    // clang-format on
}


TEST_F(Coo, ApplyFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Coo, ApplyFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Coo, ApplyFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Coo, AppliesAddToDenseVector)
{
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({2.0, 1.0}, exec);

    mtx->apply2(x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({15.0, 6.0}), 0.0);
}


TEST_F(Coo, AppliesAddToDenseMatrix)
{
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>(
        {{1.0, 0.5},
         {2.0, -1.5}}, exec);
    // clang-format on

    mtx->apply2(x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{14.0,  4.0},
                       { 7.0, -9.0}}), 0.0);
    // clang-format on
}


TEST_F(Coo, AppliesLinearCombinationAddToDenseVector)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    auto x = gko::initialize<Vec>({2.0, 1.0, 4.0}, exec);
    auto y = gko::initialize<Vec>({1.0, 2.0}, exec);

    mtx->apply2(alpha.get(), x.get(), y.get());

    GKO_ASSERT_MTX_NEAR(y, l({-12.0, -3.0}), 0.0);
}


TEST_F(Coo, AppliesLinearCombinationAddToDenseMatrix)
{
    auto alpha = gko::initialize<Vec>({-1.0}, exec);
    // clang-format off
    auto x = gko::initialize<Vec>(
        {{2.0, 3.0},
         {1.0, -1.5},
         {4.0, 2.5}}, exec);
    auto y = gko::initialize<Vec>(
        {{1.0, 0.5},
         {2.0, -1.5}}, exec);
    // clang-format on

    mtx->apply2(alpha.get(), x.get(), y.get());

    // clang-format off
    GKO_ASSERT_MTX_NEAR(y,
                    l({{-12.0, -3.0},
                       { -3.0,  6.0}}), 0.0);
    // clang-format on
}


TEST_F(Coo, ApplyAddFailsOnWrongInnerDimension)
{
    auto x = Vec::create(exec, gko::dim<2>{2});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply2(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Coo, ApplyAddFailsOnWrongNumberOfRows)
{
    auto x = Vec::create(exec, gko::dim<2>{3, 2});
    auto y = Vec::create(exec, gko::dim<2>{3, 2});

    ASSERT_THROW(mtx->apply2(x.get(), y.get()), gko::DimensionMismatch);
}


TEST_F(Coo, ApplyAddFailsOnWrongNumberOfCols)
{
    auto x = Vec::create(exec, gko::dim<2>{3});
    auto y = Vec::create(exec, gko::dim<2>{2});

    ASSERT_THROW(mtx->apply2(x.get(), y.get()), gko::DimensionMismatch);
}


}  // namespace
