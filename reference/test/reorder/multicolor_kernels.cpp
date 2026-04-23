// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/reorder/multicolor_kernels.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"
#include "core/test/utils/reordering.hpp"


namespace {


class Multicolor : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;

    Multicolor() : exec(gko::ReferenceExecutor::create())
    {
        auto mdata5 =
            gko::test::generate_laplacian_2d_5point_matrix_data<v_type, i_type>(
                dims2);
        laplace2d5 = gko::share(CsrMtx::create(exec));
        laplace2d5->read(mdata5);
        auto mdata27 =
            gko::test::generate_laplacian_3d_27point_matrix_data<v_type,
                                                                 i_type>(dims3);
        laplace3d27 = gko::share(CsrMtx::create(exec));
        laplace3d27->read(mdata27);
    }

    gko::dim<2> dims2{4, 4};
    gko::dim<3> dims3{4, 4, 4};
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<CsrMtx> laplace2d5;
    std::shared_ptr<CsrMtx> laplace3d27;

    static bool is_permutation(const perm_type* input_perm)
    {
        const auto perm_size = input_perm->get_size()[0];
        auto perm_sorted = std::vector<i_type>(perm_size);
        std::copy_n(input_perm->get_const_permutation(), perm_size,
                    perm_sorted.begin());
        std::sort(perm_sorted.begin(), perm_sorted.end());
        auto identity = std::vector<i_type>(perm_size);
        std::iota(identity.begin(), identity.end(), 0);
        return identity == perm_sorted;
    }
};


TEST_F(Multicolor, CreatesCorrectColorPtrs2d5p)
{
    const auto nrows = static_cast<i_type>(dims2[0] * dims2[1]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace2d5->get_const_row_ptrs(),
        laplace2d5->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    ASSERT_EQ(color_ptrs.size(), 3);
    EXPECT_EQ(color_ptrs[0], 0);
    EXPECT_EQ(color_ptrs[1], nrows / 2);
    EXPECT_EQ(color_ptrs[2], nrows);
}


TEST_F(Multicolor, CreatesCorrectPermutations2d5p)
{
    const auto nrows = static_cast<i_type>(dims2[0] * dims2[1]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;
    auto expected_ordering =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(dims2);

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace2d5->get_const_row_ptrs(),
        laplace2d5->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    EXPECT_EQ(expected_ordering.old_to_new, perm);
    EXPECT_EQ(expected_ordering.new_to_old, invperm);
}

TEST_F(Multicolor, CreatesCorrectColorPtrs3d27p)
{
    const auto nrows = static_cast<i_type>(dims3[0] * dims3[1] * dims3[2]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace3d27->get_const_row_ptrs(),
        laplace3d27->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    ASSERT_EQ(color_ptrs.size(), 9);
    for (int color = 0; color < 9; color++) {
        EXPECT_EQ(color_ptrs[color], color * nrows / 8);
    }
}


TEST_F(Multicolor, CreatesCorrectPermutations3d27p)
{
    const auto nrows = static_cast<i_type>(dims3[0] * dims3[1] * dims3[2]);
    std::vector<i_type> perm(nrows);
    std::vector<i_type> invperm(nrows);
    std::vector<i_type> color_ptrs;
    auto expected_ordering =
        gko::test::compute_multicolor_ordering_regular_box<i_type>(dims3);

    gko::kernels::reference::multicolor::compute_permutation_csr(
        exec, nrows, laplace3d27->get_const_row_ptrs(),
        laplace3d27->get_const_col_idxs(), color_ptrs, perm.data(),
        invperm.data());

    EXPECT_EQ(expected_ordering.old_to_new, perm);
    EXPECT_EQ(expected_ordering.new_to_old, invperm);
}


}  // namespace
