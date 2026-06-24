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


template <typename ValueIndexType>
class Pmis : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using MgLevel = gko::multigrid::Pmis<value_type, index_type>;
    using VT = value_type;
    using real_type = gko::remove_complex<value_type>;
    Pmis()
        : exec(gko::ReferenceExecutor::create()),
          pmis_factory(MgLevel::build().with_skip_sorting(true).on(exec)),
          mtx1(Mtx::create(exec)),
          mtx2(Mtx::create(exec)),
          mtx3(Mtx::create(exec)),
          row_maxabs1(exec, {1, 5, 2}),
          row_maxabs2(exec, {2, 1, 2}),
          row_maxabs3(exec, {5, 6, 4, 1, 5}),
          dep_row_ptrs1(this->exec, {0, 1, 3, 4}),
          dep_col_idxs1(this->exec, {1, 0, 2, 1}),
          dep_row_ptrs2(this->exec, {0, 2, 3, 4}),
          dep_col_idxs2(this->exec, {1, 2, 0, 1}),
          dep_row_ptrs3(this->exec, {0, 1, 2, 5, 8, 9}),
          dep_col_idxs3(this->exec, {2, 3, 0, 1, 4, 1, 2, 4, 1})
    {
        /**
         * 2 -1 0
         * 4  3 5
         * 0 -2 1
         */
        mtx1->read({{3, 3},
                    {{0, 0, VT{2}},
                     {0, 1, VT{-1}},
                     {0, 2, VT{0}},
                     {1, 0, VT{4}},
                     {1, 1, VT{3}},
                     {1, 2, VT{5}},
                     {2, 0, VT{0}},
                     {2, 1, VT{-2}},
                     {2, 2, VT{1}}}});
        /**
         *  4 -2   0.5
         * -1  3   0
         *  0 -0.5 2
         */
        mtx2->read({{3, 3},
                    {{0, 0, VT{4}},
                     {0, 1, VT{-2}},
                     {0, 2, VT{0.5}},
                     {1, 0, VT{-1}},
                     {1, 1, VT{3}},
                     {1, 2, VT{0}},
                     {2, 0, VT{0}},
                     {2, 1, VT{-0.5}},
                     {2, 2, VT{2}}}});
        /**
         * 3   5
         *   2   6
         * 1 3     4
         *   4 1 1 1
         *   5     5
         */
        mtx3->read({{5, 5},
                    {{0, 0, value_type{3}},
                     {0, 2, value_type{5}},
                     {1, 1, value_type{2}},
                     {1, 3, value_type{6}},
                     {2, 0, value_type{1}},
                     {2, 1, value_type{3}},
                     {2, 4, value_type{4}},
                     {3, 1, value_type{4}},
                     {3, 2, value_type{1}},
                     {3, 3, value_type{1}},
                     {3, 4, value_type{1}},
                     {4, 1, value_type{5}},
                     {4, 4, value_type{5}}}});
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<typename MgLevel::Factory> pmis_factory;
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    std::unique_ptr<Mtx> mtx3;
    gko::array<real_type> row_maxabs1;
    gko::array<real_type> row_maxabs2;
    gko::array<real_type> row_maxabs3;
    gko::array<index_type> dep_row_ptrs1;
    gko::array<index_type> dep_row_ptrs2;
    gko::array<index_type> dep_row_ptrs3;
    gko::array<index_type> dep_col_idxs1;
    gko::array<index_type> dep_col_idxs2;
    gko::array<index_type> dep_col_idxs3;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


// TODO: some copy/move instruction need to be done when the entire setup is
// ready if it merged before removing clone

TYPED_TEST(Pmis, ComputeRowMaxAbs1)
{
    using real_type = typename TestFixture::real_type;

    gko::array<real_type> maxabs(this->exec, 3);

    gko::kernels::reference::pmis::compute_row_maxabs(
        this->exec, this->mtx1.get(), maxabs.get_data());

    GKO_ASSERT_ARRAY_EQ(maxabs, this->row_maxabs1);
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


TYPED_TEST(Pmis, ComputeStrongDepRow3)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    gko::array<index_type> sparsity_rows(this->exec,
                                         this->mtx3->get_size()[0] + 1);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        this->exec, this->mtx3.get(), this->row_maxabs3.get_const_data(),
        real_type{0.25}, sparsity_rows.get_data());
    gko::kernels::reference::components::prefix_sum_nonnegative(
        this->exec, sparsity_rows.get_data(), sparsity_rows.get_size());

    GKO_ASSERT_ARRAY_EQ(sparsity_rows, this->dep_row_ptrs3);
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


TYPED_TEST(Pmis, ComputeStrongDep3)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    auto num_rows = this->mtx3->get_size()[0];
    auto sparsity_rows = this->dep_row_ptrs3;
    gko::array<index_type> sparsity_cols(
        this->exec, sparsity_rows.get_const_data()[num_rows]);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, this->mtx3->get_size(), std::move(sparsity_cols),
        std::move(sparsity_rows));

    gko::kernels::reference::pmis::compute_strong_dep(
        this->exec, this->mtx3.get(), this->row_maxabs3.get_const_data(),
        real_type{0.25}, strong_dep.get());

    for (int i = 0; i < this->dep_col_idxs3.get_size(); i++) {
        ASSERT_EQ(strong_dep->get_const_col_idxs()[i],
                  this->dep_col_idxs3.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, InitializeWeightAndStatus1)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto S = SparsityCsr::create(this->exec, this->mtx1->get_size(),
                                 std::move(this->dep_col_idxs1),
                                 std::move(this->dep_row_ptrs1));
    gko::array<real_type> weight(this->exec, 3);
    gko::array<int> status(this->exec, 3);
    gko::array<int> expected_status(this->exec, {-1, -1, -1});
    gko::array<real_type> floor_weight(this->exec, {1, 2, 1});

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, S.get(), weight.get_data(), status.get_data());

    GKO_ASSERT_ARRAY_EQ(status, expected_status);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(std::floor(weight.get_const_data()[i]),
                  floor_weight.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, InitializeWeightAndStatus2)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto S = SparsityCsr::create(this->exec, this->mtx2->get_size(),
                                 std::move(this->dep_col_idxs2),
                                 std::move(this->dep_row_ptrs2));
    gko::array<real_type> weight(this->exec, 3);
    gko::array<int> status(this->exec, 3);
    gko::array<int> expected_status(this->exec, {-1, -1, -1});
    gko::array<real_type> floor_weight(this->exec, {1, 2, 1});

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, S.get(), weight.get_data(), status.get_data());

    GKO_ASSERT_ARRAY_EQ(status, expected_status);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(std::floor(weight.get_const_data()[i]),
                  floor_weight.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, InitializeWeightAndStatus3)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto S = SparsityCsr::create(this->exec, this->mtx3->get_size(),
                                 std::move(this->dep_col_idxs3),
                                 std::move(this->dep_row_ptrs3));
    gko::array<real_type> weight(this->exec, 5);
    gko::array<int> status(this->exec, 5);
    gko::array<int> expected_status(this->exec, {-1, -1, -1, -1, -1});
    gko::array<real_type> floor_weight(this->exec, {1, 3, 2, 1, 2});

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, S.get(), weight.get_data(), status.get_data());


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
    auto S = SparsityCsr::create(this->exec, this->mtx1->get_size(),
                                 std::move(this->dep_col_idxs1),
                                 std::move(this->dep_row_ptrs1));
    auto trans_S = gko::as<SparsityCsr>(S->transpose());
    gko::array<real_type> weight(this->exec, {1.2, 2.1, 1.1});
    gko::array<int> status(this->exec, {-1, -1, -1});
    gko::array<int> new_status(this->exec, {-1, -1, -1});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), S.get(), trans_S.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 0);
    EXPECT_EQ(new_status.get_const_data()[1], 1);
    EXPECT_EQ(new_status.get_const_data()[2], 0);
}


TYPED_TEST(Pmis, Classify3)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto S = SparsityCsr::create(this->exec, this->mtx3->get_size(),
                                 std::move(this->dep_col_idxs3),
                                 std::move(this->dep_row_ptrs3));
    auto trans_S = gko::as<SparsityCsr>(S->transpose());
    gko::array<real_type> weight(this->exec, {0.0, 3.1, 2.1, 1.2, 2.2});
    gko::array<int> status(this->exec, {0, -1, -1, -1, -1});
    gko::array<int> new_status(this->exec, {-1, -1, -1, -1, -1});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), S.get(), trans_S.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 0);
    EXPECT_EQ(new_status.get_const_data()[1], 1);
    EXPECT_EQ(new_status.get_const_data()[2], 0);
    EXPECT_EQ(new_status.get_const_data()[3], 0);
    EXPECT_EQ(new_status.get_const_data()[4], 0);
}


// this test seems not to be useful
TYPED_TEST(Pmis, ClassifySameWeight)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto S = SparsityCsr::create(this->exec, this->mtx3->get_size(),
                                 std::move(this->dep_col_idxs3),
                                 std::move(this->dep_row_ptrs3));
    auto trans_S = gko::as<SparsityCsr>(S->transpose());
    gko::array<real_type> weight(this->exec, {0, 3, 2, 1, 2});
    gko::array<int> status(this->exec, {0, -1, -1, -1, -1});
    gko::array<int> new_status(this->exec, {-1, -1, -1, -1, -1});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), S.get(), trans_S.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 0);
    EXPECT_EQ(new_status.get_const_data()[1], 1);
    EXPECT_EQ(new_status.get_const_data()[2], 0);
    EXPECT_EQ(new_status.get_const_data()[3], 0);
    EXPECT_EQ(new_status.get_const_data()[4], 0);
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


TYPED_TEST(Pmis, DirectInterpolationRowCount)
{
    using index_type = typename TestFixture::index_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto S = SparsityCsr::create(this->exec, this->mtx3->get_size(),
                                 std::move(this->dep_col_idxs3),
                                 std::move(this->dep_row_ptrs3));
    gko::array<int> status(this->exec, {0, 1, 0, 0, 0});
    gko::array<index_type> prolong_row_count(this->exec, 5);

    gko::kernels::reference::pmis::direct_interpolation_row_count(
        this->exec, S.get(), status.get_const_data(),
        prolong_row_count.get_data());

    EXPECT_EQ(prolong_row_count.get_const_data()[0], 0);
    EXPECT_EQ(prolong_row_count.get_const_data()[1], 1);
    EXPECT_EQ(prolong_row_count.get_const_data()[2], 1);
    EXPECT_EQ(prolong_row_count.get_const_data()[3], 1);
    EXPECT_EQ(prolong_row_count.get_const_data()[4], 1);
}


TYPED_TEST(Pmis, DirectInterpolationFill)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;

    gko::array<int> coarse_map(this->exec, {0, 0, 1, 1, 1, 1});
    gko::array<index_type> prolong_row_ptrs(this->exec, {0, 0, 1, 2, 3, 4});
    gko::array<index_type> prolong_col_idxs(this->exec, 4);
    gko::array<value_type> prolong_values(this->exec, 4);

    gko::kernels::reference::pmis::direct_interpolation_fill(
        this->exec, this->mtx3.get(), this->row_maxabs3.get_const_data(),
        real_type{0.25}, coarse_map.get_const_data(),
        prolong_row_ptrs.get_const_data(), prolong_col_idxs.get_data(),
        prolong_values.get_data());

    gko::array<index_type> expected_col_idxs(this->exec, {0, 0, 0, 0});
    GKO_ASSERT_ARRAY_EQ(prolong_col_idxs, expected_col_idxs);
    EXPECT_EQ(prolong_values.get_const_data()[0], value_type{1.0});
    EXPECT_EQ(prolong_values.get_const_data()[2], value_type{-6.0});
    EXPECT_EQ(prolong_values.get_const_data()[3], value_type{-1.0});
}
