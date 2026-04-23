// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/test/utils/reordering.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/reorder/multicolor.hpp>

#include "core/test/utils/assertions.hpp"
#include "core/test/utils/matrix_generator.hpp"


namespace {


template <typename i_type>
bool is_permutation(const gko::size_type perm_size, const i_type* input_perm)
{
    auto perm_sorted = std::vector<i_type>(perm_size);
    std::copy_n(input_perm, perm_size, perm_sorted.begin());
    std::sort(perm_sorted.begin(), perm_sorted.end());
    auto identity = std::vector<i_type>(perm_size);
    std::iota(identity.begin(), identity.end(), 0);
    return identity == perm_sorted;
}


class Multicolor2d5pt : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;

    Multicolor2d5pt()
        : exec(gko::ReferenceExecutor::create()),
          mdata5{gko::test::generate_laplacian_2d_5point_matrix_data<v_type,
                                                                     i_type>(
              dims2)}
    {}

    gko::dim<2> dims2{4, 4};
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    gko::matrix_data<v_type, i_type> mdata5;
};

TEST_F(Multicolor2d5pt, GivesCorrectColorPtrs)
{
    const auto nrows = static_cast<i_type>(dims2[0] * dims2[1]);

    auto ordering =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(dims2);

    ASSERT_EQ(ordering.color_ptrs.size(), 3);
    EXPECT_EQ(ordering.color_ptrs[0], 0);
    EXPECT_EQ(ordering.color_ptrs[1], nrows / 2);
    EXPECT_EQ(ordering.color_ptrs[2], nrows);
}

TEST_F(Multicolor2d5pt, GivesConsistentPermutation)
{
    const auto nrows = static_cast<i_type>(dims2[0] * dims2[1]);

    auto ordering =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(dims2);

    EXPECT_TRUE(is_permutation(nrows, ordering.new_to_old.data()));
    EXPECT_TRUE(is_permutation(nrows, ordering.old_to_new.data()));
    for (i_type i = 0; i < nrows; i++) {
        EXPECT_EQ(ordering.old_to_new[ordering.new_to_old[i]], i);
        EXPECT_EQ(ordering.new_to_old[ordering.old_to_new[i]], i);
    }
}

TEST_F(Multicolor2d5pt, GivesCorrectPermutation)
{
    const auto nrows = static_cast<i_type>(dims2[0] * dims2[1]);

    auto ordering =
        gko::test::compute_multicolor_ordering_regular_star<i_type>(dims2);

    // j = 0
    EXPECT_EQ(ordering.old_to_new[0], 0);
    EXPECT_EQ(ordering.old_to_new[1], nrows / 2);
    EXPECT_EQ(ordering.old_to_new[2], 1);
    EXPECT_EQ(ordering.old_to_new[3], nrows / 2 + 1);
    // j = 1
    EXPECT_EQ(ordering.old_to_new[4], nrows / 2 + 2);
    EXPECT_EQ(ordering.old_to_new[5], 2);
    EXPECT_EQ(ordering.old_to_new[6], nrows / 2 + 3);
    EXPECT_EQ(ordering.old_to_new[7], 3);
    // j = 2
    EXPECT_EQ(ordering.old_to_new[8], 4);
    EXPECT_EQ(ordering.old_to_new[9], nrows / 2 + 4);
    EXPECT_EQ(ordering.old_to_new[10], 5);
    EXPECT_EQ(ordering.old_to_new[11], nrows / 2 + 5);
    // j = 3
    EXPECT_EQ(ordering.old_to_new[12], nrows / 2 + 6);
    EXPECT_EQ(ordering.old_to_new[13], 6);
    EXPECT_EQ(ordering.old_to_new[14], nrows / 2 + 7);
    EXPECT_EQ(ordering.old_to_new[15], 7);
}


class Multicolor3d27pt : public ::testing::Test {
protected:
    using v_type = double;
    using i_type = int;
    using CsrMtx = gko::matrix::Csr<v_type, i_type>;
    using reorder_type = gko::reorder::Multicolor<v_type, i_type>;
    using perm_type = gko::matrix::Permutation<i_type>;

    Multicolor3d27pt()
        : exec(gko::ReferenceExecutor::create()),
          mdata27{gko::test::generate_laplacian_3d_27point_matrix_data<v_type,
                                                                       i_type>(
              dims3)}
    {}

    gko::dim<3> dims3{4, 4, 4};
    std::shared_ptr<const gko::ReferenceExecutor> exec;
    gko::matrix_data<v_type, i_type> mdata27;
};

TEST_F(Multicolor3d27pt, GivesCorrectColorPtrs)
{
    const auto nrows = static_cast<i_type>(dims3[0] * dims3[1] * dims3[2]);

    auto ordering =
        gko::test::compute_multicolor_ordering_regular_box<i_type>(dims3);

    ASSERT_EQ(ordering.color_ptrs.size(), 9);
    for (int color = 0; color < 9; color++) {
        EXPECT_EQ(ordering.color_ptrs[color], color * nrows / 8);
    }
}

TEST_F(Multicolor3d27pt, IsConsistentPermutation)
{
    const auto nrows = static_cast<i_type>(dims3[0] * dims3[1] * dims3[2]);
    auto ordering =
        gko::test::compute_multicolor_ordering_regular_box<i_type>(dims3);
    EXPECT_TRUE(is_permutation(nrows, ordering.new_to_old.data()));
    EXPECT_TRUE(is_permutation(nrows, ordering.old_to_new.data()));
    for (i_type i = 0; i < nrows; i++) {
        EXPECT_EQ(ordering.old_to_new[ordering.new_to_old[i]], i);
        EXPECT_EQ(ordering.new_to_old[ordering.old_to_new[i]], i);
    }
}

TEST_F(Multicolor3d27pt, GivesCorrectPermutation)
{
    const auto nrows = static_cast<i_type>(dims3[0] * dims3[1] * dims3[2]);

    auto ordering =
        gko::test::compute_multicolor_ordering_regular_box<i_type>(dims3);

    // k = 0, j = 0
    EXPECT_EQ(ordering.old_to_new[0], 0);
    EXPECT_EQ(ordering.old_to_new[1], nrows / 8);
    EXPECT_EQ(ordering.old_to_new[2], 1);
    EXPECT_EQ(ordering.old_to_new[3], nrows / 8 + 1);
    // k = 0, j = 1
    EXPECT_EQ(ordering.old_to_new[4], 2 * nrows / 8);
    EXPECT_EQ(ordering.old_to_new[5], 3 * nrows / 8);
    EXPECT_EQ(ordering.old_to_new[6], 2 * nrows / 8 + 1);
    EXPECT_EQ(ordering.old_to_new[7], 3 * nrows / 8 + 1);
    // k = 1, j = 0
    EXPECT_EQ(ordering.old_to_new[16], 4 * nrows / 8);
    EXPECT_EQ(ordering.old_to_new[17], 5 * nrows / 8);
    EXPECT_EQ(ordering.old_to_new[18], 4 * nrows / 8 + 1);
    EXPECT_EQ(ordering.old_to_new[19], 5 * nrows / 8 + 1);
    // k = 1, j = 1
    EXPECT_EQ(ordering.old_to_new[20], 6 * nrows / 8);
    EXPECT_EQ(ordering.old_to_new[21], 7 * nrows / 8);
    EXPECT_EQ(ordering.old_to_new[22], 6 * nrows / 8 + 1);
    EXPECT_EQ(ordering.old_to_new[23], 7 * nrows / 8 + 1);
    // k = 2, j = 3
    EXPECT_EQ(ordering.old_to_new[40], 6);
    EXPECT_EQ(ordering.old_to_new[41], nrows / 8 + 6);
    EXPECT_EQ(ordering.old_to_new[42], 7);
    EXPECT_EQ(ordering.old_to_new[43], nrows / 8 + 7);
}


}  // namespace
