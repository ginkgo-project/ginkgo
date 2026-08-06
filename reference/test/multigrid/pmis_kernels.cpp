// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pmis_kernels.hpp"

#include <array>
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
          mtx{Mtx::create(this->exec), Mtx::create(this->exec)},
          row_maxabs{gko::array<real_type>(this->exec, {4, 0, 8, 6}),
                     gko::array<real_type>(this->exec, {1, 1, 5, 1, 0})},
          dep_row_ptrs{gko::array<index_type>(this->exec, {0, 3, 3, 4, 5}),
                       gko::array<index_type>(this->exec, {0, 2, 3, 4, 6, 6})},
          dep_col_idxs{gko::array<index_type>(this->exec, {1, 2, 3, 1, 2}),
                       gko::array<index_type>(this->exec, {2, 4, 4, 1, 2, 4})},
          expected_status{gko::array<int>(this->exec, {0, -1, -1, -1}),
                          gko::array<int>(this->exec, {0, -1, -1, 0, -1})},
          floor_weight{gko::array<real_type>(this->exec, {0, 2, 2, 1}),
                       gko::array<real_type>(this->exec, {0, 1, 2, 0, 3})},
          prolong_op1(Mtx::create(this->exec)),
          coarse_op1(Mtx::create(this->exec))
    {
        /**
         * 4  -1 4 2
         *     3
         * -1 -8 1
         * 1     6 3
         */
        mtx.at(0)->read({{4, 4},
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
        mtx.at(1)->read({{5, 5},
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
        // we only have the following for mtx.at(1).
        // For mtx.at(0), we have same weight before randomization, so there is
        // no determinstic result
        prolong_op1->read({{5, 2},
                           {{0, 0, value_type{-0.25}},
                            {0, 1, value_type{-0.25}},
                            {1, 1, value_type{-0.25}},
                            {2, 0, value_type{1}},
                            {3, 0, value_type{0.5}},
                            {3, 1, value_type{-0.5}},
                            {4, 1, value_type{1}}}});
        coarse_op1->read({{2, 2},
                          {{0, 0, value_type{3}},
                           {0, 1, value_type{-0.25}},
                           {1, 1, value_type{1}}}});
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::array<std::shared_ptr<Mtx>, 2> mtx;
    std::array<gko::array<real_type>, 2> row_maxabs;
    std::array<gko::array<index_type>, 2> dep_row_ptrs;
    std::array<gko::array<index_type>, 2> dep_col_idxs;
    std::array<gko::array<int>, 2> expected_status;
    std::array<gko::array<real_type>, 2> floor_weight;
    std::shared_ptr<Mtx> prolong_op1;
    std::shared_ptr<Mtx> coarse_op1;
};

TYPED_TEST_SUITE(Pmis, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Pmis, ComputeRowMaxAbs)
{
    using real_type = typename TestFixture::real_type;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        gko::array<real_type> maxabs(this->exec,
                                     this->mtx.at(i)->get_size()[0]);

        gko::kernels::reference::pmis::compute_row_maxabs(
            this->exec, this->mtx.at(i).get(), maxabs.get_data());

        GKO_ASSERT_ARRAY_EQ(maxabs, this->row_maxabs.at(i));
    }
}


TYPED_TEST(Pmis, ComputeStrongDepRow)
{
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        gko::array<index_type> sparsity_rows(
            this->exec, this->mtx.at(i)->get_size()[0] + 1);

        gko::kernels::reference::pmis::compute_strong_dep_row(
            this->exec, this->mtx.at(i).get(),
            this->row_maxabs.at(i).get_const_data(), real_type{0.25},
            sparsity_rows.get_data());
        gko::kernels::reference::components::prefix_sum_nonnegative(
            this->exec, sparsity_rows.get_data(), sparsity_rows.get_size());

        GKO_ASSERT_ARRAY_EQ(sparsity_rows, this->dep_row_ptrs.at(i));
    }
}


TYPED_TEST(Pmis, ComputeStrongDep)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto num_rows = this->mtx.at(i)->get_size()[0];
        auto sparsity_rows = this->dep_row_ptrs.at(i);
        auto strong_dep_ans =
            gko::matrix::SparsityCsr<value_type, index_type>::create(
                this->exec, this->mtx.at(i)->get_size(),
                std::move(this->dep_col_idxs.at(i)),
                std::move(this->dep_row_ptrs.at(i)));
        gko::array<index_type> sparsity_cols(
            this->exec, sparsity_rows.get_const_data()[num_rows]);
        auto strong_dep =
            gko::matrix::SparsityCsr<value_type, index_type>::create(
                this->exec, this->mtx.at(i)->get_size(),
                std::move(sparsity_cols), std::move(sparsity_rows));

        gko::kernels::reference::pmis::compute_strong_dep(
            this->exec, this->mtx.at(i).get(),
            this->row_maxabs.at(i).get_const_data(), real_type{0.25},
            strong_dep.get());

        GKO_ASSERT_MTX_EQ_SPARSITY(strong_dep, strong_dep_ans);
    }
}


TYPED_TEST(Pmis, InitializeWeightAndStatus)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto strong_dep =
            SparsityCsr::create(this->exec, this->mtx.at(i)->get_size(),
                                std::move(this->dep_col_idxs.at(i)),
                                std::move(this->dep_row_ptrs.at(i)));
        auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
        auto num_row = this->mtx.at(i)->get_size()[0];
        gko::array<real_type> weight(this->exec, num_row);
        gko::array<int> status(this->exec, num_row);

        gko::kernels::reference::pmis::initialize_weight_and_status(
            this->exec, trans_strong_dep.get(), weight.get_data(),
            status.get_data());

        GKO_ASSERT_ARRAY_EQ(status, this->expected_status.at(i));
        for (int row = 0; row < num_row; row++) {
            auto val = weight.get_const_data()[row];
            auto ans = this->floor_weight.at(i).get_const_data()[row];
            ASSERT_GE(val, ans);
            ASSERT_LE(val, ans + 1);
        }
    }
}


TYPED_TEST(Pmis, Classify)
{
    using real_type = typename TestFixture::real_type;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    std::array<gko::array<real_type>, 2> weight{
        gko::array<real_type>(this->exec, {0.1, 2.2, 2.1, 1.2}),
        gko::array<real_type>(this->exec, {0.0, 1.0, 2.0, 0.0, 3.0})};
    std::array<gko::array<int>, 4> status_ans{
        gko::array<int>(this->exec, {0, 1, 0, -1}),
        gko::array<int>(this->exec, {0, 1, 0, 1}),
        gko::array<int>(this->exec, {0, 0, 1, 0, 1})};
    std::array<int, 2> required_step{2, 1};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto strong_dep =
            SparsityCsr::create(this->exec, this->mtx.at(i)->get_size(),
                                std::move(this->dep_col_idxs.at(i)),
                                std::move(this->dep_row_ptrs.at(i)));
        auto trans_strong_dep = gko::as<SparsityCsr>(strong_dep->transpose());
        auto new_status = this->expected_status.at(i);
        for (int step = 0; step < required_step.at(i); step++) {
            SCOPED_TRACE(step);
            auto status = new_status;
            gko::kernels::reference::pmis::classify(
                this->exec, weight.at(i).get_data(), strong_dep.get(),
                trans_strong_dep.get(), status.get_const_data(),
                new_status.get_data());

            GKO_ASSERT_ARRAY_EQ(new_status, status_ans.at(i * 2 + step));
        }
    }
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
    std::array<gko::array<int>, 2> status{
        gko::array<int>(this->exec, {0, 1, 0, 1}),
        gko::array<int>(this->exec, {0, 0, 1, 0, 1})};
    std::array<gko::array<index_type>, 2> row_count_ans{
        gko::array<index_type>(this->exec, {2, 1, 1, 1}),
        gko::array<index_type>(this->exec, {2, 1, 1, 2, 1})};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto strong_dep =
            SparsityCsr::create(this->exec, this->mtx.at(i)->get_size(),
                                std::move(this->dep_col_idxs.at(i)),
                                std::move(this->dep_row_ptrs.at(i)));
        gko::array<index_type> prolong_row_count(
            this->exec, this->mtx.at(i)->get_size()[0]);

        gko::kernels::reference::pmis::direct_interpolation_row_count(
            this->exec, strong_dep.get(), status.at(i).get_const_data(),
            prolong_row_count.get_data());

        GKO_ASSERT_ARRAY_EQ(prolong_row_count, row_count_ans.at(i));
    }
}


TYPED_TEST(Pmis, DirectInterpolationFill)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using real_type = typename TestFixture::real_type;
    std::array<gko::array<index_type>, 2> coarse_map{
        gko::array<index_type>(this->exec, {0, 0, 1, 1, 2}),
        gko::array<index_type>(this->exec, {0, 0, 0, 1, 1, 2})};
    std::array<gko::array<index_type>, 2> prolong_row_ptrs{
        gko::array<index_type>(this->exec, {0, 2, 3, 4, 5}),
        gko::array<index_type>(this->exec, {0, 2, 3, 4, 6, 7})};
    std::array<gko::array<index_type>, 2> expected_col_idxs{
        gko::array<index_type>(this->exec, {0, 1, 0, 0, 1}),
        gko::array<index_type>(this->exec, {0, 1, 1, 0, 0, 1, 1})};
    std::array<gko::array<value_type>, 2> expected_values{
        gko::array<value_type>(
            this->exec, {value_type(0.25), value_type(-1.5), value_type(1),
                         value_type(9), value_type(1)}),
        gko::array<value_type>(
            this->exec,
            {value_type{-0.25}, value_type{-0.25}, value_type{-0.25},
             value_type{1}, value_type{0.5}, value_type{-0.5}, value_type{1}})};
    for (int i = 0; i < 2; i++) {
        SCOPED_TRACE(i);
        auto prolong_nnz =
            prolong_row_ptrs.at(i)
                .get_const_data()[this->mtx.at(i)->get_size()[0]];
        gko::array<index_type> prolong_col_idxs(this->exec, prolong_nnz);
        gko::array<value_type> prolong_values(this->exec, prolong_nnz);

        gko::kernels::reference::pmis::direct_interpolation_fill(
            this->exec, this->mtx.at(i).get(),
            this->row_maxabs.at(i).get_const_data(), real_type{0.25},
            coarse_map.at(i).get_const_data(),
            prolong_row_ptrs.at(i).get_const_data(),
            prolong_col_idxs.get_data(), prolong_values.get_data());

        GKO_ASSERT_ARRAY_EQ(prolong_col_idxs, expected_col_idxs.at(i));
        GKO_ASSERT_ARRAY_EQ(prolong_values, expected_values.at(i));
    }
}


TYPED_TEST(Pmis, GenerateMgLevel)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    auto factory = MgLevel::build().with_skip_sorting(true).on(this->exec);
    auto restrict_op = gko::as<Mtx>(this->prolong_op1->transpose());

    auto mg = factory->generate(this->mtx.at(1));

    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_fine_op()), this->mtx.at(1), 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->coarse_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), this->prolong_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), restrict_op, 0.0);
}


TYPED_TEST(Pmis, GenerateMgLevelOnUnsortedMtx)
{
    using MgLevel = typename TestFixture::MgLevel;
    using Mtx = typename TestFixture::Mtx;
    auto factory = MgLevel::build().on(this->exec);
    auto restrict_op = gko::as<Mtx>(this->prolong_op1->transpose());
    std::default_random_engine rng{793643};
    gko::test::unsort_matrix(this->mtx.at(1), rng);

    auto mg = factory->generate(this->mtx.at(1));

    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_fine_op()), this->mtx.at(1), 0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_coarse_op()), this->coarse_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_prolong_op()), this->prolong_op1,
                        0.0);
    GKO_EXPECT_MTX_NEAR(gko::as<Mtx>(mg->get_restrict_op()), restrict_op, 0.0);
}
