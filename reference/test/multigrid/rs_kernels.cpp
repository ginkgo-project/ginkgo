// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/rs_kernels.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace {


using value_type = double;
using index_type = int;
using csr = gko::matrix::Csr<value_type, index_type>;


class Rs : public ::testing::Test {
protected:
    Rs() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<gko::ReferenceExecutor> exec;
};


/**
 * Test matrix (1D Laplacian, 3-point stencil):
 *
 * A =
 * [  2  -1   0 ]
 * [ -1   2  -1 ]
 * [  0  -1   2 ]
 *
 * theta = 0.5
 *
 * consider for each row:
 *
 * row 0:
 *   offdiag = {-1}
 *   max_offdiag = 1
 *   strong if 1 >= 0.5*1 -> true
 *   -> S(0,:) = {1}
 *
 * row 1:
 *   offdiag = {-1,-1}
 *   max_offdiag = 1
 *   both satisfy 1 >= 0.5*1
 *   -> S(1,:) = {0,2}
 *
 * row 2:
 *   offdiag = {-1}
 *   -> S(2,:) = {1}
 *
 * S row_ptrs = {0,1,3,4}
 */


TEST_F(Rs, ComputeSocRowPtrs)
{
    auto A = csr::create(exec, gko::dim<2>{3, 3}, 7);
    A->read({{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}});

    gko::array<index_type> row_ptrs(exec, 4);

    gko::kernels::reference::rs::compute_soc_row_ptrs(exec, A.get(), 0.5,
                                                      row_ptrs.get_data());

    std::vector<index_type> expected{0, 1, 3, 4};

    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(row_ptrs.get_const_data()[i], expected[i]);
    }
}


TEST_F(Rs, FillSoc)
{
    auto A = csr::create(exec, gko::dim<2>{3, 3}, 7);
    A->read({{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}});

    gko::array<index_type> row_ptrs(exec, 4);
    gko::kernels::reference::rs::compute_soc_row_ptrs(exec, A.get(), 0.5,
                                                      row_ptrs.get_data());

    auto S = csr::create(exec, gko::dim<2>{3, 3}, 4);
    exec->copy_from(exec, 4, row_ptrs.get_const_data(), S->get_row_ptrs());

    gko::kernels::reference::rs::fill_soc(exec, A.get(), 0.5, S.get());

    std::vector<index_type> expected_cols{1, 0, 2, 1};

    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(S->get_const_col_idxs()[i], expected_cols[i]);
        ASSERT_EQ(S->get_const_values()[i], 1.0);
    }
}


TEST_F(Rs, ComputeLambda)
{
    auto S = csr::create(exec, gko::dim<2>{3, 3}, 4);
    S->read({{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}});

    gko::array<index_type> lambda(exec, 3);

    gko::kernels::reference::rs::compute_lambda(exec, S.get(),
                                                lambda.get_data());

    ASSERT_EQ(lambda.get_const_data()[0], 1);
    ASSERT_EQ(lambda.get_const_data()[1], 2);
    ASSERT_EQ(lambda.get_const_data()[2], 1);
}


TEST_F(Rs, RsCoarsening)
{
    auto S = csr::create(exec, gko::dim<2>{3, 3}, 4);
    S->read({{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}});

    gko::array<index_type> lambda(exec, 3);
    lambda.get_data()[0] = 1;
    lambda.get_data()[1] = 2;
    lambda.get_data()[2] = 1;

    gko::array<index_type> cf(exec, 3);
    gko::kernels::reference::rs::init_cf(exec, cf);

    gko::kernels::reference::rs::rs_coarsening(exec, S.get(), lambda.get_data(),
                                               cf);

    ASSERT_EQ(cf.get_const_data()[0], -1);
    ASSERT_EQ(cf.get_const_data()[1], 1);
    ASSERT_EQ(cf.get_const_data()[2], -1);
}


TEST_F(Rs, RsCleanup)
{
    gko::array<index_type> cf(exec, 3);
    cf.get_data()[0] = 0;
    cf.get_data()[1] = 1;
    cf.get_data()[2] = -1;

    gko::kernels::reference::rs::rs_cleanup(exec, cf);

    ASSERT_EQ(cf.get_const_data()[0], -1);
    ASSERT_EQ(cf.get_const_data()[1], 1);
    ASSERT_EQ(cf.get_const_data()[2], -1);
}


TEST_F(Rs, CountCoarse)
{
    gko::array<index_type> cf(exec, 3);
    cf.get_data()[0] = -1;
    cf.get_data()[1] = 1;
    cf.get_data()[2] = -1;

    index_type coarse{};
    gko::kernels::reference::rs::count_coarse(exec, cf, &coarse);

    ASSERT_EQ(coarse, 1);
}


TEST_F(Rs, FillCoarseRows)
{
    gko::array<index_type> cf(exec, 3);
    cf.get_data()[0] = -1;
    cf.get_data()[1] = 1;
    cf.get_data()[2] = -1;

    gko::array<index_type> coarse(exec, 1);

    gko::kernels::reference::rs::fill_coarse_rows(exec, cf, coarse.get_data());

    ASSERT_EQ(coarse.get_const_data()[0], 1);
}


}  // namespace
