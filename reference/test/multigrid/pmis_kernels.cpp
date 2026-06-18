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
          pmis_factory(MgLevel::build().with_skip_sorting(true).on(exec))
    {}

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::unique_ptr<typename MgLevel::Factory> pmis_factory;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


// TODO: some copy/move instruction need to be done when the entire setup is
// ready

TYPED_TEST(Pmis, ComputeRowMaxAbs1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{2}},
              {0, 1, value_type{-1}},
              {0, 2, value_type{0}},
              {1, 0, value_type{4}},
              {1, 1, value_type{3}},
              {1, 2, value_type{5}},
              {2, 0, value_type{0}},
              {2, 1, value_type{-2}},
              {2, 2, value_type{1}}}});
    gko::array<real_type> row_maxabs(this->exec, 3);

    gko::kernels::reference::pmis::compute_row_maxabs(this->exec, A.get(),
                                                      row_maxabs.get_data());

    gko::array<real_type> expected(this->exec, {1, 5, 2});
    GKO_ASSERT_ARRAY_EQ(row_maxabs, expected);
}


TYPED_TEST(Pmis, ComputeStrongDepRow1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{2}},
              {0, 1, value_type{-1}},
              {0, 2, value_type{0}},
              {1, 0, value_type{4}},
              {1, 1, value_type{3}},
              {1, 2, value_type{5}},
              {2, 0, value_type{0}},
              {2, 1, value_type{-2}},
              {2, 2, value_type{1}}}});
    gko::array<real_type> row_maxabs(this->exec, {1, 5, 2});
    gko::array<index_type> sparsity_rows(this->exec, 3);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        this->exec, A.get(), row_maxabs.get_const_data(), real_type{0.25},
        sparsity_rows.get_data());

    gko::array<index_type> expected(this->exec, {1, 2, 1});
    GKO_ASSERT_ARRAY_EQ(sparsity_rows, expected);
}


TYPED_TEST(Pmis, ComputeStrongDepRow2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{4}},
              {0, 1, value_type{-2}},
              {0, 2, value_type{0.5}},
              {1, 0, value_type{-1}},
              {1, 1, value_type{3}},
              {1, 2, value_type{0}},
              {2, 0, value_type{0}},
              {2, 1, value_type{-0.5}},
              {2, 2, value_type{2}}}});
    gko::array<real_type> row_maxabs(this->exec, {2, 1, 2});
    gko::array<index_type> sparsity_rows(this->exec, 3);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        this->exec, A.get(), row_maxabs.get_const_data(), real_type{0.25},
        sparsity_rows.get_data());

    gko::array<index_type> expected(this->exec, {2, 1, 1});
    GKO_ASSERT_ARRAY_EQ(sparsity_rows, expected);
}


TYPED_TEST(Pmis, ComputeStrongDepRow3)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    auto A = Mtx::create(this->exec, gko::dim<2>{5, 5}, 13);
    A->read({{5, 5},
             {{0, 0, value_type{3}},
              {0, 2, value_type{5}},
              {1, 0, value_type{1}},
              {1, 1, value_type{2}},
              {1, 3, value_type{6}},
              {2, 1, value_type{3}},
              {2, 4, value_type{4}},
              {3, 1, value_type{4}},
              {3, 2, value_type{1}},
              {3, 3, value_type{1}},
              {3, 4, value_type{1}},
              {4, 1, value_type{5}},
              {4, 4, value_type{5}}}});
    gko::array<real_type> row_maxabs(this->exec, {5, 6, 4, 4, 5});
    gko::array<index_type> sparsity_rows(this->exec, 5);

    gko::kernels::reference::pmis::compute_strong_dep_row(
        this->exec, A.get(), row_maxabs.get_const_data(), real_type{0.25},
        sparsity_rows.get_data());

    gko::array<index_type> expected(this->exec, {1, 1, 2, 3, 1});
    GKO_ASSERT_ARRAY_EQ(sparsity_rows, expected);
}


TYPED_TEST(Pmis, ComputeStrongDep1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{2}},
              {0, 1, value_type{-1}},
              {0, 2, value_type{0}},
              {1, 0, value_type{4}},
              {1, 1, value_type{3}},
              {1, 2, value_type{5}},
              {2, 0, value_type{0}},
              {2, 1, value_type{-2}},
              {2, 2, value_type{1}}}});
    gko::array<real_type> row_maxabs(this->exec, {1, 5, 2});
    gko::array<index_type> sparsity_rows(this->exec, {0, 1, 3, 4});
    gko::array<index_type> sparsity_cols(this->exec, 4);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 3}, std::move(sparsity_cols),
        std::move(sparsity_rows));

    gko::kernels::reference::pmis::compute_strong_dep(
        this->exec, A.get(), row_maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());

    auto ci = strong_dep->get_const_col_idxs();
    ASSERT_EQ(ci[0], 1);
    ASSERT_EQ(ci[1], 0);
    ASSERT_EQ(ci[2], 2);
    ASSERT_EQ(ci[3], 1);
}


TYPED_TEST(Pmis, ComputeStrongDep2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{4}},
              {0, 1, value_type{-2}},
              {0, 2, value_type{0.5}},
              {1, 0, value_type{-1}},
              {1, 1, value_type{3}},
              {1, 2, value_type{0}},
              {2, 0, value_type{0}},
              {2, 1, value_type{-0.5}},
              {2, 2, value_type{2}}}});
    gko::array<real_type> row_maxabs(this->exec, {2, 1, 2});
    gko::array<index_type> sparsity_rows(this->exec, {0, 2, 3, 4});
    gko::array<index_type> sparsity_cols(this->exec, 4);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{3, 3}, std::move(sparsity_cols),
        std::move(sparsity_rows));

    gko::kernels::reference::pmis::compute_strong_dep(
        this->exec, A.get(), row_maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());

    auto ci = strong_dep->get_const_col_idxs();
    ASSERT_EQ(ci[0], 1);
    ASSERT_EQ(ci[1], 2);
    ASSERT_EQ(ci[2], 0);
    ASSERT_EQ(ci[3], 1);
}


TYPED_TEST(Pmis, ComputeStrongDep3)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    auto A = Mtx::create(this->exec, gko::dim<2>{5, 5}, 13);
    A->read({{5, 5},
             {{0, 0, value_type{3}},
              {0, 2, value_type{5}},
              {1, 0, value_type{1}},
              {1, 1, value_type{2}},
              {1, 3, value_type{6}},
              {2, 1, value_type{3}},
              {2, 4, value_type{4}},
              {3, 1, value_type{4}},
              {3, 2, value_type{1}},
              {3, 3, value_type{1}},
              {3, 4, value_type{1}},
              {4, 1, value_type{5}},
              {4, 4, value_type{5}}}});
    gko::array<real_type> row_maxabs(this->exec, {5, 6, 4, 4, 5});
    gko::array<index_type> sparsity_rows(this->exec, {0, 1, 2, 4, 7, 8});
    gko::array<index_type> sparsity_cols(this->exec, 8);
    auto strong_dep = gko::matrix::SparsityCsr<value_type, index_type>::create(
        this->exec, gko::dim<2>{5, 5}, std::move(sparsity_cols),
        std::move(sparsity_rows));

    gko::kernels::reference::pmis::compute_strong_dep(
        this->exec, A.get(), row_maxabs.get_const_data(), real_type{0.25},
        strong_dep.get());

    auto ci = strong_dep->get_const_col_idxs();
    ASSERT_EQ(ci[0], 2);
    ASSERT_EQ(ci[1], 3);
    ASSERT_EQ(ci[2], 1);
    ASSERT_EQ(ci[3], 4);
    ASSERT_EQ(ci[4], 1);
    ASSERT_EQ(ci[5], 2);
    ASSERT_EQ(ci[6], 4);
    ASSERT_EQ(ci[7], 1);
}


TYPED_TEST(Pmis, InitializeWeightAndStatus1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{2}},
              {0, 1, value_type{-1}},
              {0, 2, value_type{0}},
              {1, 0, value_type{4}},
              {1, 1, value_type{3}},
              {1, 2, value_type{5}},
              {2, 0, value_type{0}},
              {2, 1, value_type{-2}},
              {2, 2, value_type{1}}}});
    gko::array<index_type> sparsity_rows(this->exec, {0, 1, 3, 4});
    gko::array<index_type> sparsity_cols(this->exec, {1, 0, 2, 1});
    auto S =
        SparsityCsr::create(this->exec, gko::dim<2>{3, 3},
                            std::move(sparsity_cols), std::move(sparsity_rows));

    gko::array<real_type> weight(this->exec, 3);
    gko::array<int> status(this->exec, 3);

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, S.get(), weight.get_data(), status.get_data());

    gko::array<int> expected_status(this->exec, {0, 0, 0});
    gko::array<real_type> floor_weight(this->exec, {1, 2, 1});
    GKO_ASSERT_ARRAY_EQ(status, expected_status);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(std::floor(weight.get_const_data()[i]),
                  floor_weight.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, InitializeWeightAndStatus2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{4}},
              {0, 1, value_type{-2}},
              {0, 2, value_type{0.5}},
              {1, 0, value_type{-1}},
              {1, 1, value_type{3}},
              {2, 1, value_type{-0.5}},
              {2, 2, value_type{2}}}});
    gko::array<index_type> sparsity_rows(this->exec, {0, 2, 3, 4});
    gko::array<index_type> sparsity_cols(this->exec, {1, 2, 0, 1});
    auto S =
        SparsityCsr::create(this->exec, gko::dim<2>{3, 3},
                            std::move(sparsity_cols), std::move(sparsity_rows));
    gko::array<real_type> weight(this->exec, 3);
    gko::array<int> status(this->exec, 3);

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, S.get(), weight.get_data(), status.get_data());

    gko::array<int> expected_status(this->exec, {0, 0, 0});
    gko::array<real_type> floor_weight(this->exec, {1, 2, 1});
    GKO_ASSERT_ARRAY_EQ(status, expected_status);
    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(std::floor(weight.get_const_data()[i]),
                  floor_weight.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, InitializeWeightAndStatus3)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto A = Mtx::create(this->exec, gko::dim<2>{5, 5}, 13);
    A->read({{5, 5},
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
    gko::array<index_type> sparsity_rows(this->exec, {0, 1, 2, 4, 7, 8});
    gko::array<index_type> sparsity_cols(this->exec, {2, 3, 1, 4, 1, 2, 4, 1});
    auto S =
        SparsityCsr::create(this->exec, gko::dim<2>{5, 5},
                            std::move(sparsity_cols), std::move(sparsity_rows));
    gko::array<real_type> weight(this->exec, 5);
    gko::array<int> status(this->exec, 5);

    gko::kernels::reference::pmis::initialize_weight_and_status(
        this->exec, S.get(), weight.get_data(), status.get_data());

    gko::array<int> expected_status(this->exec, {1, 0, 0, 0, 0});
    gko::array<real_type> floor_weight(this->exec, {0, 3, 2, 1, 2});
    GKO_ASSERT_ARRAY_EQ(status, expected_status);
    for (int i = 0; i < 5; i++) {
        ASSERT_EQ(std::floor(weight.get_const_data()[i]),
                  floor_weight.get_const_data()[i]);
    }
}


TYPED_TEST(Pmis, Classify1)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto A = Mtx::create(this->exec, gko::dim<2>{3, 3}, 6);
    A->read({{3, 3},
             {{0, 0, value_type{4}},
              {0, 1, value_type{-2}},
              {0, 2, value_type{0.5}},
              {1, 0, value_type{-1}},
              {1, 1, value_type{3}},
              {2, 1, value_type{-0.5}},
              {2, 2, value_type{2}}}});
    gko::array<index_type> sparsity_rows(this->exec, {0, 1, 3, 4});
    gko::array<index_type> sparsity_cols(this->exec, {1, 0, 2, 1});
    auto S =
        SparsityCsr::create(this->exec, gko::dim<2>{3, 3},
                            std::move(sparsity_cols), std::move(sparsity_rows));
    auto trans_S = gko::as<SparsityCsr>(S->transpose());
    gko::array<real_type> weight(this->exec, {1.2, 2.1, 1.1});
    gko::array<int> status(this->exec, {0, 0, 0});
    gko::array<int> new_status(this->exec, {0, 0, 0});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), S.get(), trans_S.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 1);
    EXPECT_EQ(new_status.get_const_data()[1], 2);
    EXPECT_EQ(new_status.get_const_data()[2], 1);
}


TYPED_TEST(Pmis, Classify2)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto A = Mtx::create(this->exec, gko::dim<2>{5, 5}, 13);
    A->read({{5, 5},
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
    gko::array<index_type> sparsity_rows(this->exec, {0, 1, 2, 4, 7, 8});
    gko::array<index_type> sparsity_cols(this->exec, {2, 3, 1, 4, 1, 2, 4, 1});
    auto S =
        SparsityCsr::create(this->exec, gko::dim<2>{5, 5},
                            std::move(sparsity_cols), std::move(sparsity_rows));
    auto trans_S = gko::as<SparsityCsr>(S->transpose());
    gko::array<real_type> weight(this->exec, {0.0, 3.1, 2.1, 1.2, 2.2});
    gko::array<int> status(this->exec, {1, 0, 0, 0, 0});
    gko::array<int> new_status(this->exec, {0, 0, 0, 0, 0});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), S.get(), trans_S.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 1);
    EXPECT_EQ(new_status.get_const_data()[1], 2);
    EXPECT_EQ(new_status.get_const_data()[2], 1);
    EXPECT_EQ(new_status.get_const_data()[2], 1);
    EXPECT_EQ(new_status.get_const_data()[2], 1);
}


// this test seems not to be useful
TYPED_TEST(Pmis, ClassifySameWeight)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    auto A = Mtx::create(this->exec, gko::dim<2>{5, 5}, 13);
    A->read({{5, 5},
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
    gko::array<index_type> sparsity_rows(this->exec, {0, 1, 2, 4, 7, 8});
    gko::array<index_type> sparsity_cols(this->exec, {2, 3, 1, 4, 1, 2, 4, 1});
    auto S =
        SparsityCsr::create(this->exec, gko::dim<2>{5, 5},
                            std::move(sparsity_cols), std::move(sparsity_rows));
    auto trans_S = gko::as<SparsityCsr>(S->transpose());
    gko::array<real_type> weight(this->exec, {0, 3, 2, 1, 2});
    gko::array<int> status(this->exec, {1, 0, 0, 0, 0});
    gko::array<int> new_status(this->exec, {0, 0, 0, 0, 0});

    gko::kernels::reference::pmis::classify(
        this->exec, weight.get_data(), S.get(), trans_S.get(),
        status.get_const_data(), new_status.get_data());

    EXPECT_EQ(new_status.get_const_data()[0], 1);
    EXPECT_EQ(new_status.get_const_data()[1], 2);
    EXPECT_EQ(new_status.get_const_data()[2], 1);
    EXPECT_EQ(new_status.get_const_data()[3], 1);
    EXPECT_EQ(new_status.get_const_data()[4], 1);
}


TYPED_TEST(Pmis, Count)
{
    gko::array<int> arr(this->exec, 5);
    auto data = arr.get_data();
    data[0] = 0;
    data[1] = 1;
    data[2] = 0;
    data[3] = 2;
    data[4] = 0;

    gko::size_type num = 0;
    gko::kernels::reference::pmis::count(this->exec, 5, arr.get_const_data(),
                                         &num);

    EXPECT_EQ(num, 3);
}
