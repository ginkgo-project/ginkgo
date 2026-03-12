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

    //  * A =
    //  * [  2  -1   0 ]
    //  * [ -1   2  -1 ]
    //  * [  0  -1   2 ]
    // split: 0=C, 1=F, 2=C
    // interpolation row 1: w_10 = -(-1/2) = 0.5, w_12 = -(-1/2) = 0.5
    void setup_test_data()
    {
        A = csr::create(exec, gko::dim<2>{3, 3}, 7);
        A->read({{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}});

        S = csr::create(exec, gko::dim<2>{3, 3}, 4);
        S->read({{0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 0.0}});

        cf = gko::array<index_type>(exec, {1, -1, 1});  // C, F, C
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<csr> A;
    std::shared_ptr<csr> S;
    gko::array<index_type> cf;
};


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


TEST_F(Rs, FillFineToCoarse)
{
    this->setup_test_data();
    gko::array<index_type> f2c(exec, 3);

    gko::kernels::reference::rs::fill_fine_to_coarse(exec, cf, f2c.get_data());

    // C-points get sequential IDs, F-points get -1
    ASSERT_EQ(f2c.get_const_data()[0], 0);
    ASSERT_EQ(f2c.get_const_data()[1], -1);
    ASSERT_EQ(f2c.get_const_data()[2], 1);
}


TEST_F(Rs, ComputeInterpolationRowPtrs)
{
    this->setup_test_data();
    gko::array<index_type> p_row_ptrs(exec, 4);

    gko::kernels::reference::rs::compute_interpolation_row_ptrs(
        exec, S.get(), cf, p_row_ptrs.get_data());

    // Row 0 (C): 1 nz, Row 1 (F): 2 nz (strong C-neighbors 0,2), Row 2 (C): 1
    // nz
    std::vector<index_type> expected{0, 1, 3, 4};
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(p_row_ptrs.get_const_data()[i], expected[i]);
    }
}


TEST_F(Rs, ComputeInterpolation)
{
    this->setup_test_data();
    gko::array<index_type> f2c(exec, {0, -1, 1});
    auto P = csr::create(exec, gko::dim<2>{3, 2}, 4);
    // manual row_ptrs for 0=C, 1=F(0,2), 2=C
    exec->copy_from(exec, 4, std::vector<index_type>{0, 1, 3, 4}.data(),
                    P->get_row_ptrs());

    gko::kernels::reference::rs::compute_interpolation(
        exec, A.get(), S.get(), cf, f2c.get_const_data(), P.get());

    ASSERT_EQ(P->get_const_col_idxs()[0], 0);
    ASSERT_DOUBLE_EQ(P->get_const_values()[0], 1.0);

    ASSERT_EQ(P->get_const_col_idxs()[1], 0);
    ASSERT_DOUBLE_EQ(P->get_const_values()[1], 0.5);
    ASSERT_EQ(P->get_const_col_idxs()[2], 1);
    ASSERT_DOUBLE_EQ(P->get_const_values()[2], 0.5);

    ASSERT_EQ(P->get_const_col_idxs()[3], 1);
    ASSERT_DOUBLE_EQ(P->get_const_values()[3], 1.0);
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
