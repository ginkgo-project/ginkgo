// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/pgm.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/multigrid/pgm_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Pgm : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using MgLevel = gko::multigrid::Pgm<value_type, index_type>;
    using RowGatherer = gko::matrix::RowGatherer<index_type>;
    using VT = value_type;
    using real_type = gko::remove_complex<value_type>;
    using WeightMtx = gko::matrix::Csr<real_type, index_type>;
    Pgm()
        : exec(gko::ReferenceExecutor::create()),
          pgm_factory(MgLevel::build()
                          .with_max_iterations(2u)
                          .with_max_unassigned_ratio(0.1)
                          .with_skip_sorting(true)
                          .on(exec)),
          fine_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({-1.0, 2.0}), I<VT>({0.0, -1.0}),
               I<VT>({3.0, -2.0}), I<VT>({-2.0, 1.0})},
              exec)),
          coarse_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, -1.0})}, exec)),
          restrict_ans((gko::initialize<Vec>(
              {I<VT>({0.0, -1.0}), I<VT>({2.0, 0.0})}, exec))),
          prolong_ans(gko::initialize<Vec>(
              {I<VT>({0.0, -2.0}), I<VT>({1.0, -2.0}), I<VT>({1.0, -2.0}),
               I<VT>({0.0, -1.0}), I<VT>({2.0, 1.0})},
              exec)),
          prolong_applyans(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, -1.0}), I<VT>({2.0, -1.0}),
               I<VT>({0.0, -1.0}), I<VT>({2.0, -1.0})},
              exec)),
          fine_x(gko::initialize<Vec>(
              {I<VT>({-2.0, -1.0}), I<VT>({1.0, -1.0}), I<VT>({-1.0, -1.0}),
               I<VT>({0.0, 0.0}), I<VT>({0.0, 2.0})},
              exec)),
          mtx(Mtx::create(exec, gko::dim<2>(5, 5), 15,
                          std::make_shared<typename Mtx::classical>())),
          weight(WeightMtx::create(
              exec, gko::dim<2>(5, 5), 15,
              std::make_shared<typename WeightMtx::classical>())),
          coarse(Mtx::create(exec, gko::dim<2>(2, 2), 4,
                             std::make_shared<typename Mtx::classical>())),
          agg(exec, 5)
    {
        this->create_mtx(mtx.get(), weight.get(), &agg, coarse.get());
        mg_level = pgm_factory->generate(mtx);
        mtx_diag = weight->extract_diagonal();
    }

    void create_mtx(Mtx* fine, WeightMtx* weight, gko::array<index_type>* agg,
                    Mtx* coarse)
    {
        auto agg_val = agg->get_data();
        agg_val[0] = 0;
        agg_val[1] = 1;
        agg_val[2] = 0;
        agg_val[3] = 1;
        agg_val[4] = 0;

        /* this matrix is stored:
         *  5 -3 -3  0  0
         * -3  5  0 -2 -1
         * -3  0  5  0 -1
         *  0 -3  0  5  0
         *  0 -2 -2  0  5
         */
        fine->read({{5, 5},
                    {{0, 0, 5},
                     {0, 1, -3},
                     {0, 2, -3},
                     {1, 0, -3},
                     {1, 1, 5},
                     {1, 3, -2},
                     {1, 4, -1},
                     {2, 0, -3},
                     {2, 2, 5},
                     {2, 4, -1},
                     {3, 1, -3},
                     {3, 3, 5},
                     {4, 1, -2},
                     {4, 2, -2},
                     {4, 4, 5}}});

        /* weight matrix is stored:
         * 5   3   3   0   0
         * 3   5   0   2.5 1.5
         * 3   0   5   0   1.5
         * 0   2.5 0   5   0
         * 0   1.5 1.5 0   5
         */
        weight->read({{5, 5},
                      {{0, 0, 5.0},
                       {0, 1, 3.0},
                       {0, 2, 3.0},
                       {1, 0, 3.0},
                       {1, 1, 5.0},
                       {1, 3, 2.5},
                       {1, 4, 1.5},
                       {2, 0, 3.0},
                       {2, 2, 5.0},
                       {2, 4, 1.5},
                       {3, 1, 2.5},
                       {3, 3, 5.0},
                       {4, 1, 1.5},
                       {4, 2, 1.5},
                       {4, 4, 5.0}}});

        /* this coarse is stored:
         *  6 -5
         * -4  5
         */
        coarse->read({{2, 2}, {{0, 0, 6}, {0, 1, -5}, {1, 0, -4}, {1, 1, 5}}});
    }

    static void assert_same_matrices(const Mtx* m1, const Mtx* m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        ASSERT_EQ(m1->get_num_stored_elements(), m2->get_num_stored_elements());
        for (gko::size_type i = 0; i < m1->get_size()[0] + 1; i++) {
            ASSERT_EQ(m1->get_const_row_ptrs()[i], m2->get_const_row_ptrs()[i]);
        }
        for (gko::size_type i = 0; i < m1->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m1->get_const_values()[i], m2->get_const_values()[i]);
            EXPECT_EQ(m1->get_const_col_idxs()[i], m2->get_const_col_idxs()[i]);
        }
    }

    static void assert_same_agg(const index_type* m1, const index_type* m2,
                                gko::size_type len)
    {
        for (gko::size_type i = 0; i < len; ++i) {
            EXPECT_EQ(m1[i], m2[i]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> coarse;
    std::shared_ptr<WeightMtx> weight;
    std::shared_ptr<gko::matrix::Diagonal<real_type>> mtx_diag;
    gko::array<index_type> agg;
    std::shared_ptr<Vec> coarse_b;
    std::shared_ptr<Vec> fine_b;
    std::shared_ptr<Vec> restrict_ans;
    std::shared_ptr<Vec> prolong_ans;
    std::shared_ptr<Vec> prolong_applyans;
    std::shared_ptr<Vec> fine_x;
    std::unique_ptr<typename MgLevel::Factory> pgm_factory;
    std::unique_ptr<MgLevel> mg_level;
};

TYPED_TEST_SUITE(Pgm, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Pgm, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy = this->pgm_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->mg_level);
    auto copy_mtx = copy->get_system_matrix();
    auto copy_agg = copy->get_const_agg();
    auto copy_coarse = copy->get_coarse_op();

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
    this->assert_same_agg(copy_agg, this->agg.get_data(), this->agg.get_size());
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_coarse), this->coarse, 0.0);
}


TYPED_TEST(Pgm, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy = this->pgm_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->mg_level);
    auto copy_mtx = copy->get_system_matrix();
    auto copy_agg = copy->get_const_agg();
    auto copy_coarse = copy->get_coarse_op();

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
    this->assert_same_agg(copy_agg, this->agg.get_data(), this->agg.get_size());
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_coarse), this->coarse, 0.0);
}


TYPED_TEST(Pgm, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto clone = this->mg_level->clone();
    auto clone_mtx = clone->get_system_matrix();
    auto clone_agg = clone->get_const_agg();
    auto clone_coarse = clone->get_coarse_op();

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx), this->mtx, 0.0);
    this->assert_same_agg(clone_agg, this->agg.get_data(),
                          this->agg.get_size());
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_coarse), this->coarse, 0.0);
}


TYPED_TEST(Pgm, CanBeCleared)
{
    using MgLevel = typename TestFixture::MgLevel;

    this->mg_level->clear();
    auto mtx = this->mg_level->get_system_matrix();
    auto coarse = this->mg_level->get_coarse_op();
    auto agg = this->mg_level->get_agg();

    ASSERT_EQ(mtx, nullptr);
    ASSERT_EQ(coarse, nullptr);
    ASSERT_EQ(agg, nullptr);
}


TYPED_TEST(Pgm, MatchEdge)
{
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> agg(this->exec, 5);
    gko::array<index_type> snb(this->exec, 5);
    auto agg_val = agg.get_data();
    auto snb_val = snb.get_data();
    for (int i = 0; i < 5; i++) {
        agg_val[i] = -1;
    }
    snb_val[0] = 2;
    snb_val[1] = 0;
    snb_val[2] = 0;
    snb_val[3] = 1;
    // isolated item
    snb_val[4] = 4;

    gko::kernels::reference::pgm::match_edge(this->exec, snb, agg);

    ASSERT_EQ(agg_val[0], 0);
    ASSERT_EQ(agg_val[1], -1);
    ASSERT_EQ(agg_val[2], 0);
    ASSERT_EQ(agg_val[3], -1);
    // isolated item should be self aggregation
    ASSERT_EQ(agg_val[4], 4);
}


TYPED_TEST(Pgm, CountUnagg)
{
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> agg(this->exec, 5);
    auto agg_val = agg.get_data();
    index_type num_unagg = 0;
    agg_val[0] = 0;
    agg_val[1] = -1;
    agg_val[2] = 0;
    agg_val[3] = -1;
    agg_val[4] = -1;

    gko::kernels::reference::pgm::count_unagg(this->exec, agg, &num_unagg);

    ASSERT_EQ(num_unagg, 3);
}


TYPED_TEST(Pgm, Renumber)
{
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> agg(this->exec, 5);
    auto agg_val = agg.get_data();
    index_type num_agg = 0;
    agg_val[0] = 0;
    agg_val[1] = 1;
    agg_val[2] = 0;
    agg_val[3] = 1;
    agg_val[4] = 4;

    gko::kernels::reference::pgm::renumber(this->exec, agg, &num_agg);

    ASSERT_EQ(num_agg, 3);
    ASSERT_EQ(agg_val[0], 0);
    ASSERT_EQ(agg_val[1], 1);
    ASSERT_EQ(agg_val[2], 0);
    ASSERT_EQ(agg_val[3], 1);
    ASSERT_EQ(agg_val[4], 2);
}


TYPED_TEST(Pgm, Generate)
{
    auto coarse_fine = this->pgm_factory->generate(this->mtx);

    auto agg_result = coarse_fine->get_const_agg();

    ASSERT_EQ(agg_result[0], 0);
    ASSERT_EQ(agg_result[1], 1);
    ASSERT_EQ(agg_result[2], 0);
    ASSERT_EQ(agg_result[3], 1);
    ASSERT_EQ(agg_result[4], 0);
}


TYPED_TEST(Pgm, CoarseFineRestrictApply)
{
    auto pgm = this->pgm_factory->generate(this->mtx);
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = Vec::create_with_config_of(this->coarse_b);

    pgm->get_restrict_op()->apply(this->fine_b, x);

    GKO_ASSERT_MTX_NEAR(x, this->restrict_ans, r<value_type>::value);
}


TYPED_TEST(Pgm, CoarseFineProlongApplyadd)
{
    using value_type = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto pgm = this->pgm_factory->generate(this->mtx);
    auto one = gko::initialize<Vec>({value_type{1.0}}, this->exec);
    auto x = gko::clone(this->fine_x);

    pgm->get_prolong_op()->apply(one, this->coarse_b, one, x);

    GKO_ASSERT_MTX_NEAR(x, this->prolong_ans, r<value_type>::value);
}


TYPED_TEST(Pgm, CoarseFineProlongApply)
{
    using value_type = typename TestFixture::value_type;
    auto pgm = this->pgm_factory->generate(this->mtx);
    auto x = gko::clone(this->fine_x);

    pgm->get_prolong_op()->apply(this->coarse_b, x);

    GKO_ASSERT_MTX_NEAR(x, this->prolong_applyans, r<value_type>::value);
}


TYPED_TEST(Pgm, Apply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto pgm = this->pgm_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = pgm->get_executor();
    auto answer = gko::initialize<Vec>(
        {I<VT>({-23.0, 5.0}), I<VT>({17.0, -5.0}), I<VT>({-23.0, 5.0}),
         I<VT>({17.0, -5.0}), I<VT>({-23.0, 5.0})},
        exec);

    pgm->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(Pgm, AdvancedApply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto pgm = this->pgm_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = pgm->get_executor();
    auto alpha = gko::initialize<Vec>({1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto answer = gko::initialize<Vec>(
        {I<VT>({-27.0, 3.0}), I<VT>({19.0, -7.0}), I<VT>({-25.0, 3.0}),
         I<VT>({17.0, -5.0}), I<VT>({-23.0, 9.0})},
        exec);

    pgm->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(Pgm, FindStrongestNeighbor)
{
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> strongest_neighbor(this->exec, 5);
    gko::array<index_type> agg(this->exec, 5);
    auto snb_vals = strongest_neighbor.get_data();
    auto agg_vals = agg.get_data();
    for (int i = 0; i < 5; i++) {
        snb_vals[i] = -1;
        agg_vals[i] = -1;
    }

    gko::kernels::reference::pgm::find_strongest_neighbor(
        this->exec, this->weight.get(), this->mtx_diag.get(), agg,
        strongest_neighbor);

    ASSERT_EQ(snb_vals[0], 2);
    ASSERT_EQ(snb_vals[1], 0);
    ASSERT_EQ(snb_vals[2], 0);
    ASSERT_EQ(snb_vals[3], 1);
    ASSERT_EQ(snb_vals[4], 2);
}


TYPED_TEST(Pgm, AssignToExistAgg)
{
    using index_type = typename TestFixture::index_type;
    gko::array<index_type> agg(this->exec, 5);
    gko::array<index_type> intermediate_agg(this->exec, 0);
    auto agg_vals = agg.get_data();
    // 0 - 2, 1 - 3
    agg_vals[0] = 0;
    agg_vals[1] = 1;
    agg_vals[2] = 0;
    agg_vals[3] = 1;
    agg_vals[4] = -1;

    gko::kernels::reference::pgm::assign_to_exist_agg(
        this->exec, this->weight.get(), this->mtx_diag.get(), agg,
        intermediate_agg);

    ASSERT_EQ(agg_vals[0], 0);
    ASSERT_EQ(agg_vals[1], 1);
    ASSERT_EQ(agg_vals[2], 0);
    ASSERT_EQ(agg_vals[3], 1);
    ASSERT_EQ(agg_vals[4], 0);
}


TYPED_TEST(Pgm, GenerateMgLevel)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    using RowGatherer = typename TestFixture::RowGatherer;
    auto prolong_op = gko::share(Mtx::create(this->exec, gko::dim<2>{5, 2}, 0));
    // 0-2-4, 1-3
    prolong_op->read(
        {{5, 2}, {{0, 0, 1}, {1, 1, 1}, {2, 0, 1}, {3, 1, 1}, {4, 0, 1}}});
    auto restrict_op = gko::share(gko::as<Mtx>(prolong_op->transpose()));

    auto coarse_fine = this->pgm_factory->generate(this->mtx);
    auto row_gatherer = gko::as<RowGatherer>(coarse_fine->get_prolong_op());
    auto row_gather_view = gko::array<index_type>::const_view(
        this->exec, row_gatherer->get_size()[0],
        row_gatherer->get_const_row_idxs());
    auto expected_row_gather =
        gko::array<index_type>(this->exec, {0, 1, 0, 1, 0});

    GKO_ASSERT_MTX_NEAR(gko::as<SparsityCsr>(coarse_fine->get_restrict_op()),
                        restrict_op, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_coarse_op()),
                        this->coarse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(row_gather_view, expected_row_gather);
}


TYPED_TEST(Pgm, GenerateMgLevelOnUnsortedMatrix)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using SparsityCsr = typename TestFixture::SparsityCsr;
    using MgLevel = typename TestFixture::MgLevel;
    using RowGatherer = typename TestFixture::RowGatherer;
    auto mglevel_sort = MgLevel::build()
                            .with_max_iterations(2u)
                            .with_max_unassigned_ratio(0.1)
                            .on(this->exec);
    /* this unsorted matrix is stored as this->fine:
     *  5 -3 -3  0  0
     * -3  5  0 -2 -1
     * -3  0  5  0 -1
     *  0 -3  0  5  0
     *  0 -2 -2  0  5
     */
    auto mtx_values = {-3, -3, 5, -3, -2, -1, 5, -3, -1, 5, 5, -3, -2, -2, 5};
    auto mtx_col_idxs = {1, 2, 0, 0, 3, 4, 1, 0, 4, 2, 1, 3, 1, 2, 4};
    auto mtx_row_ptrs = {0, 3, 7, 10, 12, 15};
    auto matrix = gko::share(
        Mtx::create(this->exec, gko::dim<2>{5, 5}, std::move(mtx_values),
                    std::move(mtx_col_idxs), std::move(mtx_row_ptrs)));
    auto prolong_op = gko::share(Mtx::create(this->exec, gko::dim<2>{5, 2}, 0));
    // 0-2-4, 1-3
    prolong_op->read(
        {{5, 2}, {{0, 0, 1}, {1, 1, 1}, {2, 0, 1}, {3, 1, 1}, {4, 0, 1}}});
    auto restrict_op = gko::share(gko::as<Mtx>(prolong_op->transpose()));

    auto coarse_fine = mglevel_sort->generate(matrix);
    auto row_gatherer = gko::as<RowGatherer>(coarse_fine->get_prolong_op());
    auto row_gather_view = gko::array<index_type>::const_view(
        this->exec, row_gatherer->get_size()[0],
        row_gatherer->get_const_row_idxs());
    auto expected_row_gather =
        gko::array<index_type>(this->exec, {0, 1, 0, 1, 0});

    GKO_ASSERT_MTX_NEAR(gko::as<SparsityCsr>(coarse_fine->get_restrict_op()),
                        restrict_op, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_coarse_op()),
                        this->coarse, r<value_type>::value);
    GKO_ASSERT_ARRAY_EQ(row_gather_view, expected_row_gather);
}


}  // namespace
