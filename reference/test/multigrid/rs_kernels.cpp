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

#include "core/multigrid/rs_helpers.hpp"


namespace {


using value_type = double;
using index_type = int;
using csr = gko::matrix::Csr<value_type, index_type>;


class Rs : public ::testing::Test {
protected:
    Rs() : exec(gko::ReferenceExecutor::create()) {}

    // A = 5x5 Tridiagonal [ -1  2 -1 ]
    // split: 0=F, 1=C, 2=F, 3=C, 4=F
    // C-indices: 1, 3. Coarse size: 2.
    void setup_test_data()
    {
        A = csr::create(exec, gko::dim<2>{5, 5}, 13);
        A->read({{2.0, -1.0, 0.0, 0.0, 0.0},
                 {-1.0, 2.0, -1.0, 0.0, 0.0},
                 {0.0, -1.0, 2.0, -1.0, 0.0},
                 {0.0, 0.0, -1.0, 2.0, -1.0},
                 {0.0, 0.0, 0.0, -1.0, 2.0}});

        // strength mask: all 8 off-diagonals are strong
        is_strong_prefilled =
            gko::array<bool>(exec, {false, true, true, false, true, true, false,
                                    true, true, false, true, true, false});

        cf = gko::array<index_type>(exec, {-1, 1, -1, 1, -1});
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<csr> A;
    gko::array<bool> is_strong_prefilled;
    gko::array<index_type> cf;
};


TEST_F(Rs, ComputeSocAndRunRs)
{
    auto A = csr::create(exec, gko::dim<2>{5, 5}, 13);
    A->read({{2.0, -1.0, 0.0, 0.0, 0.0},
             {-1.0, 2.0, -1.0, 0.0, 0.0},
             {0.0, -1.0, 2.0, -1.0, 0.0},
             {0.0, 0.0, -1.0, 2.0, -1.0},
             {0.0, 0.0, 0.0, -1.0, 2.0}});

    gko::array<bool> is_strong_empty(exec, 13);
    gko::array<index_type> lambda(exec, 5);
    gko::array<index_type> cf(exec, 5);
    index_type coarse{};
    // this matrix is not distributed, so it has no off-diagonal block
    const auto no_off_diag =
        gko::multigrid::rs::no_off_diag_view<value_type, index_type>();

    gko::kernels::reference::rs::compute_soc_and_run_rs(
        exec, A->get_const_device_view(), no_off_diag, 0.5, is_strong_empty,
        lambda, cf, coarse);

    // all off-diagonals are strong
    std::vector<bool> expected_soc{false, true, true,  false, true, true, false,
                                   true,  true, false, true,  true, false};

    for (int i = 0; i < 13; ++i) {
        ASSERT_EQ(is_strong_empty.get_const_data()[i], expected_soc[i]);
    }
    // initial lambda: [1, 2, 2, 2, 1]. After greedy RS:
    ASSERT_EQ(cf.get_const_data()[0], -1);
    ASSERT_EQ(cf.get_const_data()[1], 1);
    ASSERT_EQ(cf.get_const_data()[2], -1);
    ASSERT_EQ(cf.get_const_data()[3], 1);
    ASSERT_EQ(cf.get_const_data()[4], -1);
    ASSERT_EQ(coarse, 2);
}


TEST_F(Rs, FillCoarseAndComputeProlongRowPtrs)
{
    this->setup_test_data();
    gko::array<index_type> f2c(exec, 5);
    gko::array<index_type> p_row_ptrs(exec, 6);
    gko::array<index_type> coarse(exec, 2);

    gko::kernels::reference::rs::fill_coarse_and_compute_prolong_row_ptrs(
        exec, cf, coarse, f2c, A->get_const_device_view(), is_strong_prefilled,
        p_row_ptrs);

    // C-points (1, 3) map to (0, 1)
    ASSERT_EQ(f2c.get_const_data()[0], -1);
    ASSERT_EQ(f2c.get_const_data()[1], 0);
    ASSERT_EQ(f2c.get_const_data()[2], -1);
    ASSERT_EQ(f2c.get_const_data()[3], 1);
    ASSERT_EQ(f2c.get_const_data()[4], -1);

    std::vector<index_type> expected_ptrs{0, 1, 2, 4, 5, 6};
    for (int i = 0; i < 6; ++i) {
        ASSERT_EQ(p_row_ptrs.get_const_data()[i], expected_ptrs[i]);
    }
}


TEST_F(Rs, ComputeInterpolation)
{
    this->setup_test_data();
    gko::array<index_type> f2c(exec, {-1, 0, -1, 1, -1});
    auto P = csr::create(exec, gko::dim<2>{5, 2}, 6);
    exec->copy_from(exec, 6, std::vector<index_type>{0, 1, 2, 4, 5, 6}.data(),
                    P->get_row_ptrs());

    gko::kernels::reference::rs::compute_interpolation(
        exec, A->get_const_device_view(), is_strong_prefilled.get_const_data(),
        cf, f2c.get_const_data(), P->get_device_view());

    auto p_vals = P->get_const_values();
    auto p_cols = P->get_const_col_idxs();

    ASSERT_EQ(p_cols[0], 0);
    ASSERT_DOUBLE_EQ(p_vals[0], 0.5);

    ASSERT_EQ(p_cols[1], 0);
    ASSERT_DOUBLE_EQ(p_vals[1], 1.0);

    ASSERT_EQ(p_cols[2], 0);
    ASSERT_DOUBLE_EQ(p_vals[2], 0.5);
    ASSERT_EQ(p_cols[3], 1);
    ASSERT_DOUBLE_EQ(p_vals[3], 0.5);

    ASSERT_EQ(p_cols[4], 1);
    ASSERT_DOUBLE_EQ(p_vals[4], 1.0);

    ASSERT_EQ(p_cols[5], 1);
    ASSERT_DOUBLE_EQ(p_vals[5], 0.5);
}


}  // namespace
