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

        // mask aligned with A's non-zeros
        is_strong = gko::array<bool>(
            exec, {false, true, true, false, true, true, false});

        cf = gko::array<index_type>(exec, {1, -1, 1});  // C, F, C
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<csr> A;
    gko::array<bool> is_strong;
    gko::array<index_type> cf;
};


TEST_F(Rs, ComputeSocAndRunRs)
{
    auto A = csr::create(exec, gko::dim<2>{3, 3}, 7);
    A->read({{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}});

    gko::array<bool> is_strong(exec, 7);
    gko::array<index_type> lambda(exec, 3);
    gko::array<index_type> cf(exec, 3);
    index_type coarse{};

    gko::kernels::reference::rs::compute_soc_and_run_rs(
        exec, A.get(), 0.5, is_strong, lambda, cf, coarse);

    std::vector<bool> expected{false, true, true, false, true, true, false};

    for (int i = 0; i < 7; ++i) {
        ASSERT_EQ(is_strong.get_const_data()[i], expected[i]);
    }
    ASSERT_EQ(lambda.get_const_data()[0], 1);
    ASSERT_EQ(lambda.get_const_data()[1], 2);
    ASSERT_EQ(lambda.get_const_data()[2], 1);
    ASSERT_EQ(cf.get_const_data()[0], -1);
    ASSERT_EQ(cf.get_const_data()[1], 1);
    ASSERT_EQ(cf.get_const_data()[2], -1);
    ASSERT_EQ(coarse, 1);
}


TEST_F(Rs, FillCoarseAndComputeProlongRowPtrs)
{
    this->setup_test_data();
    gko::array<index_type> cf(exec, 3);
    cf.get_data()[0] = -1;
    cf.get_data()[1] = 1;
    cf.get_data()[2] = -1;
    gko::array<index_type> f2c(exec, 3);
    gko::array<index_type> p_row_ptrs(exec, 4);
    gko::array<index_type> coarse(exec, 1);

    gko::kernels::reference::rs::fill_coarse_and_compute_prolong_row_ptrs(
        exec, cf, coarse, f2c, A.get(), is_strong, p_row_ptrs);

    // C-points get sequential IDs, F-points get -1
    ASSERT_EQ(f2c.get_const_data()[0], -1);
    ASSERT_EQ(f2c.get_const_data()[1], 0);
    ASSERT_EQ(f2c.get_const_data()[2], -1);
    ASSERT_EQ(coarse.get_const_data()[0], 1);
    std::vector<index_type> expected{0, 1, 2, 3};
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
        exec, A.get(), is_strong.get_const_data(), cf, f2c.get_const_data(),
        P.get());

    ASSERT_EQ(P->get_const_col_idxs()[0], 0);
    ASSERT_DOUBLE_EQ(P->get_const_values()[0], 1.0);

    ASSERT_EQ(P->get_const_col_idxs()[1], 0);
    ASSERT_DOUBLE_EQ(P->get_const_values()[1], 0.5);
    ASSERT_EQ(P->get_const_col_idxs()[2], 1);
    ASSERT_DOUBLE_EQ(P->get_const_values()[2], 0.5);

    ASSERT_EQ(P->get_const_col_idxs()[3], 1);
    ASSERT_DOUBLE_EQ(P->get_const_values()[3], 1.0);
}


}  // namespace
