// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/multigrid/pmis.hpp>
#include <ginkgo/core/stop/combined.hpp>

#include "core/components/prefix_sum_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/unsort_matrix.hpp"


template <typename ValueIndexType>
class Pmis : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using real_type = gko::remove_complex<value_type>;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;

    Pmis()
        : exec(gko::ReferenceExecutor::create()),
          mtx1(Mtx::create(this->exec)),
          mtx2(Mtx::create(this->exec)),
          row_maxabs1(exec, {4, 0, 8, 6}),
          row_maxabs2(exec, {1, 1, 5, 1, 0}),
          dep_row_ptrs1(this->exec, {0, 3, 3, 4, 5}),
          dep_col_idxs1(this->exec, {1, 2, 3, 1, 2}),
          dep_row_ptrs2(this->exec, {0, 2, 3, 4, 6, 6}),
          dep_col_idxs2(this->exec, {2, 4, 4, 1, 2, 4}),
          prolong_op2(Mtx::create(this->exec)),
          coarse_op2(Mtx::create(this->exec))
    {
        /**
         * 4  -1 4 2
         *     3
         * -1 -8 1
         * 1     6 3
         */
        mtx1->read({{4, 4},
                    {{0, 0, value_type{4}},
                     {0, 1, value_type{-1}},
                     {0, 2, value_type{4}},
                     {0, 3, value_type{2}},
                     {1, 1, value_type{3}},
                     {2, 0, value_type{-1}},
                     {2, 1, value_type{-8}},
                     {2, 2, value_type{1}},
                     {3, 0, value_type{1}},
                     {3, 2, value_type{6}},
                     {3, 3, value_type{3}}}});
        /**
         * 4    1   1
         *   4      1
         *   5  3   1
         *     -1 2 1
         *          1
         */
        mtx2->read({{5, 5},
                    {{0, 0, value_type{4}},
                     {0, 2, value_type{1}},
                     {0, 4, value_type{1}},
                     {1, 1, value_type{4}},
                     {1, 4, value_type{1}},
                     {2, 1, value_type{5}},
                     {2, 2, value_type{3}},
                     {2, 4, value_type{1}},
                     {3, 2, value_type{-1}},
                     {3, 3, value_type{2}},
                     {3, 4, value_type{1}},
                     {4, 4, value_type{1}}}});
        // we only have the following for mtx2.
        // For mtx1, we have same weight before randomization, so there is no
        // determinstic result
        prolong_op2->read({{5, 2},
                           {{0, 0, value_type{-0.25}},
                            {0, 1, value_type{-0.25}},
                            {1, 1, value_type{-0.25}},
                            {2, 0, value_type{1}},
                            {3, 0, value_type{0.5}},
                            {3, 1, value_type{-0.5}},
                            {4, 1, value_type{1}}}});
        coarse_op2->read({{2, 2},
                          {{0, 0, value_type{3}},
                           {0, 1, value_type{-0.25}},
                           {1, 1, value_type{1}}}});
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx1;
    std::shared_ptr<Mtx> mtx2;
    gko::array<real_type> row_maxabs1;
    gko::array<real_type> row_maxabs2;
    gko::array<index_type> dep_row_ptrs1;
    gko::array<index_type> dep_row_ptrs2;
    gko::array<index_type> dep_col_idxs1;
    gko::array<index_type> dep_col_idxs2;
    std::shared_ptr<Mtx> prolong_op2;
    std::shared_ptr<Mtx> coarse_op2;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


// TODO: some copy/move instruction need to be done when the entire setup is
// ready if it merged before removing clone

TYPED_TEST(Pmis, ComputeRowMaxAbs1)
{
    using real_type = typename TestFixture::real_type;
    gko::array<real_type> maxabs(this->exec, 4);

    gko::kernels::reference::pmis::compute_row_maxabs(
        this->exec, this->mtx1.get(), maxabs.get_data());

    GKO_ASSERT_ARRAY_EQ(maxabs, this->row_maxabs1);
}


TYPED_TEST(Pmis, ComputeRowMaxAbs2)
{
    using real_type = typename TestFixture::real_type;
    gko::array<real_type> maxabs(this->exec, 5);

    gko::kernels::reference::pmis::compute_row_maxabs(
        this->exec, this->mtx2.get(), maxabs.get_data());

    GKO_ASSERT_ARRAY_EQ(maxabs, this->row_maxabs2);
}


TYPED_TEST(Pmis, ComputeStrongDepRow1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    gko::array<index_type> sparsity_rows(this->exec,
                                         this->mtx1->get_size()[0] + 1);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        this->exec, this->mtx1.get(), this->row_maxabs1.get_const_data(),
        real_type{0.25}, sparsity_rows.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, sparsity_rows.get_data(), sparsity_rows.get_size());

    GKO_ASSERT_ARRAY_EQ(sparsity_rows, this->dep_row_ptrs1);
}


TYPED_TEST(Pmis, ComputeStrongDepRow2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    gko::array<index_type> sparsity_rows(this->exec,
                                         this->mtx2->get_size()[0] + 1);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        this->exec, this->mtx2.get(), this->row_maxabs2.get_const_data(),
        real_type{0.25}, sparsity_rows.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, sparsity_rows.get_data(), sparsity_rows.get_size());

    GKO_ASSERT_ARRAY_EQ(sparsity_rows, this->dep_row_ptrs2);
}


TYPED_TEST(Pmis, ComputeStrongDep1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto num_rows = this->mtx1->get_size()[0];
    auto sparsity_rows = this->dep_row_ptrs1;
    gko::array<index_type> sparsity_cols(
        this->exec, sparsity_rows.get_const_data()[num_rows]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, this->mtx1->get_size(), std::move(sparsity_cols),
        std::move(sparsity_rows));

    gko::kernels::reference::pmis::compute_strong_dep(
        this->exec, this->mtx1.get(), this->row_maxabs1.get_const_data(),
        real_type{0.25}, strong_dep.get());

    for (int i = 0; i < this->dep_col_idxs1.get_size(); i++) {
        ASSERT_EQ(strong_dep->get_const_col_idxs()[i],
                  this->dep_col_idxs1.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, ComputeStrongDep2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto num_rows = this->mtx2->get_size()[0];
    auto sparsity_rows = this->dep_row_ptrs2;
    gko::array<index_type> sparsity_cols(
        this->exec, sparsity_rows.get_const_data()[num_rows]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, this->mtx2->get_size(), std::move(sparsity_cols),
        std::move(sparsity_rows));

    gko::kernels::reference::pmis::compute_strong_dep(
        this->exec, this->mtx2.get(), this->row_maxabs2.get_const_data(),
        real_type{0.25}, strong_dep.get());

    for (int i = 0; i < this->dep_col_idxs2.get_size(); i++) {
        ASSERT_EQ(strong_dep->get_const_col_idxs()[i],
                  this->dep_col_idxs2.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, InitializeWeightAndStatus1)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx1->get_size(),
                                          std::move(this->dep_col_idxs1),
                                          std::move(this->dep_row_ptrs1));
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    auto num_row = this->mtx1->get_size()[0];
    gko::array<real_type> weight(this->exec, num_row);
    gko::array<int> status(this->exec, num_row);
    gko::array<int> expected_status(this->exec, {0, -1, -1, -1});
    gko::array<real_type> floor_weight(this->exec, {0, 2, 2, 1});

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, trans_strong_dep.get(), weight.get_data(),
        status.get_data());

    GKO_ASSERT_ARRAY_EQ(status, expected_status);
    for (int i = 0; i < num_row; i++) {
        auto val = weight.get_const_data()[i];
        auto ans = floor_weight.get_const_data()[i];
        ASSERT_GE(val, ans);
        ASSERT_LE(val, ans + 1);
    }
    // ensure having same number (after floor)
    ASSERT_EQ(std::floor(weight.get_const_data()[1]),
              std::floor(weight.get_const_data()[2]));
    // check the same number have different random value
    ASSERT_NE(weight.get_const_data()[1], weight.get_const_data()[2]);
}


TYPED_TEST(Pmis, InitializeWeightAndStatus2)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx2->get_size(),
                                          std::move(this->dep_col_idxs2),
                                          std::move(this->dep_row_ptrs2));
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    auto num_row = this->mtx2->get_size()[0];
    gko::array<real_type> weight(this->exec, num_row);
    gko::array<int> status(this->exec, num_row);
    gko::array<int> expected_status(this->exec, {0, -1, -1, 0, -1});
    gko::array<real_type> floor_weight(this->exec, {0, 1, 2, 0, 3});

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, trans_strong_dep.get(), weight.get_data(),
        status.get_data());


    GKO_ASSERT_ARRAY_EQ(status, expected_status);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(std::floor(weight.get_const_data()[i]),
                  floor_weight.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, Classify1)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx1->get_size(),
                                          std::move(this->dep_col_idxs1),
                                          std::move(this->dep_row_ptrs1));
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    gko::array<real_type> weight(this->exec, {0.1, 2.2, 2.1, 1.2});
    gko::array<int> status(this->exec, {0, -1, -1, -1});
    gko::array<int> new_status(this->exec, {0, -1, -1, -1});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), this->mtx1.get(), trans_strong_dep.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 0);
    EXPECT_EQ(new_status.get_const_data()[1], 1);
    EXPECT_EQ(new_status.get_const_data()[2], 0);
    EXPECT_EQ(new_status.get_const_data()[3], -1);
}


TYPED_TEST(Pmis, Classify1Next)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx1->get_size(),
                                          std::move(this->dep_col_idxs1),
                                          std::move(this->dep_row_ptrs1));
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    gko::array<real_type> weight(this->exec, {0.1, 2.2, 2.1, 1.2});
    gko::array<int> status(this->exec, {0, 1, 0, -1});
    gko::array<int> new_status(this->exec, {0, -1, -1, -1});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), this->mtx1.get(), trans_strong_dep.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 0);
    EXPECT_EQ(new_status.get_const_data()[1], 1);
    EXPECT_EQ(new_status.get_const_data()[2], 0);
    EXPECT_EQ(new_status.get_const_data()[3], 1);
}


TYPED_TEST(Pmis, Classify2)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx2->get_size(),
                                          std::move(this->dep_col_idxs2),
                                          std::move(this->dep_row_ptrs2));
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    gko::array<real_type> weight(this->exec, {0.0, 1.0, 2.0, 0.0, 3.0});
    gko::array<int> status(this->exec, {0, -1, -1, 0, -1});
    gko::array<int> new_status(this->exec, {0, -1, -1, 0, -1});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), this->mtx2.get(), trans_strong_dep.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 0);
    EXPECT_EQ(new_status.get_const_data()[1], 0);
    EXPECT_EQ(new_status.get_const_data()[2], -1);
    EXPECT_EQ(new_status.get_const_data()[3], 0);
    EXPECT_EQ(new_status.get_const_data()[4], 1);
}


TYPED_TEST(Pmis, Classify2Next)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx2->get_size(),
                                          std::move(this->dep_col_idxs2),
                                          std::move(this->dep_row_ptrs2));
    auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
    gko::array<real_type> weight(this->exec, {0.0, 1.0, 2.0, 0.0, 3.0});
    gko::array<int> status(this->exec, {0, 0, -1, 0, 1});
    gko::array<int> new_status(this->exec, {0, 0, -1, 0, 1});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), this->mtx2.get(), trans_strong_dep.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 0);
    EXPECT_EQ(new_status.get_const_data()[1], 0);
    EXPECT_EQ(new_status.get_const_data()[2], 1);
    EXPECT_EQ(new_status.get_const_data()[3], 0);
    EXPECT_EQ(new_status.get_const_data()[4], 1);
}


TYPED_TEST(Pmis, Count)
{
    gko::array<int> arr(this->exec, 5);
    auto data = arr.get_data();
    data[0] = -1;
    data[1] = 0;
    data[2] = -1;
    data[3] = 1;
    data[4] = -1;

    gko::size_type num = 0;
    gko::kernels::reference::pmis::count(this->exec, 5, arr.get_const_data(),
                                         &num);

    EXPECT_EQ(num, 3);
}


TYPED_TEST(Pmis, DirectInterpolationRowCount1)
{
    using index_type = typename TestFixture::index_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx1->get_size(),
                                          std::move(this->dep_col_idxs1),
                                          std::move(this->dep_row_ptrs1));
    gko::array<int> status(this->exec, {0, 1, 0, 1});
    gko::array<index_type> prolong_row_count(this->exec, 4);

    gko::kernels::reference::pmis::direct_interpolation_row_count(
        this->exec, strong_dep.get(), status.get_const_data(),
        prolong_row_count.get_data());

    EXPECT_EQ(prolong_row_count.get_const_data()[0], 2);
    EXPECT_EQ(prolong_row_count.get_const_data()[1], 1);
    EXPECT_EQ(prolong_row_count.get_const_data()[2], 1);
    EXPECT_EQ(prolong_row_count.get_const_data()[3], 1);
}


TYPED_TEST(Pmis, DirectInterpolationRowCount2)
{
    using index_type = typename TestFixture::index_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto strong_dep = SparsityCsr::create(this->exec, this->mtx2->get_size(),
                                          std::move(this->dep_col_idxs2),
                                          std::move(this->dep_row_ptrs2));
    gko::array<int> status(this->exec, {0, 0, 1, 0, 1});
    gko::array<index_type> prolong_row_count(this->exec, 5);

    gko::kernels::reference::pmis::direct_interpolation_row_count(
        this->exec, strong_dep.get(), status.get_const_data(),
        prolong_row_count.get_data());

    EXPECT_EQ(prolong_row_count.get_const_data()[0], 2);
    EXPECT_EQ(prolong_row_count.get_const_data()[1], 1);
    EXPECT_EQ(prolong_row_count.get_const_data()[2], 1);
    EXPECT_EQ(prolong_row_count.get_const_data()[3], 2);
    EXPECT_EQ(prolong_row_count.get_const_data()[4], 1);
}


TYPED_TEST(Pmis, DirectInterpolationFill1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    gko::array<index_type> coarse_map(this->exec, {0, 0, 1, 1, 2});
    gko::array<index_type> prolong_row_ptrs(this->exec, {0, 2, 3, 4, 5});
    gko::array<index_type> prolong_col_idxs(this->exec, 5);
    gko::array<value_type> prolong_values(this->exec, 5);

    gko::kernels::reference::pmis::direct_interpolation_fill(
        this->exec, this->mtx1.get(), this->row_maxabs1.get_const_data(),
        real_type{0.25}, coarse_map.get_const_data(),
        prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
        prolong_values.get_data());

    gko::array<index_type> expected_col_idxs(this->exec, {0, 1, 0, 0, 1});
    GKO_ASSERT_ARRAY_EQ(prolong_col_idxs, expected_col_idxs);
    ASSERT_EQ(prolong_values.get_const_data()[0], value_type(0.25));
    ASSERT_EQ(prolong_values.get_const_data()[1], value_type(-1.5));
    ASSERT_EQ(prolong_values.get_const_data()[2], value_type(1));
    ASSERT_EQ(prolong_values.get_const_data()[3], value_type(9));
    ASSERT_EQ(prolong_values.get_const_data()[4], value_type(1));
}


TYPED_TEST(Pmis, DirectInterpolationFill2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    gko::array<index_type> coarse_map(this->exec, {0, 0, 0, 1, 1, 2});
    gko::array<index_type> prolong_row_ptrs(this->exec, {0, 2, 3, 4, 6, 7});
    gko::array<index_type> prolong_col_idxs(this->exec, 7);
    gko::array<value_type> prolong_values(this->exec, 7);

    gko::kernels::reference::pmis::direct_interpolation_fill(
        this->exec, this->mtx2.get(), this->row_maxabs2.get_const_data(),
        real_type{0.25}, coarse_map.get_const_data(),
        prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
        prolong_values.get_data());

    auto prolong = Mtx::create(
        this->exec, this->prolong_op2->get_size(), std::move(prolong_values),
        std::move(prolong_col_idxs), std::move(prolong_row_ptrs));
    GKO_ASSERT_MTX_NEAR(prolong, this->prolong_op2, 0.0);
}


TYPED_TEST(Pmis, GenerateMgLevel)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    auto factory = MgLevel::build().with_skip_sorting(true).on(this->exec);
    auto restrict_op = gko::as<Mtx>(this->prolong_op2->transpose());

    auto mg = factory->generate(this->mtx2);

    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_fine_op()), this->mtx2, 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->coarse_op2,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), this->prolong_op2,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), restrict_op, 0.0);
}

TYPED_TEST(Pmis, GenerateMgLevelOnUnsortedMtx)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    auto factory = MgLevel::build().on(this->exec);
    auto restrict_op = gko::as<Mtx>(this->prolong_op2->transpose());
    std::default_random_engine rng{793643};
    gko::test::unsort_matrix(this->mtx2, rng);

    auto mg = factory->generate(this->mtx2);

    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_fine_op()), this->mtx2, 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->coarse_op2,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), this->prolong_op2,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), restrict_op, 0.0);
}
