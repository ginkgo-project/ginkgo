// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/fixed_coarsening.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/row_gatherer.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class FixedCoarsening : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using CooMtx = gko::matrix::Coo<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using SparsityCsr = gko::matrix::SparsityCsr<value_type, index_type>;
    using MgLevel = gko::multigrid::FixedCoarsening<value_type, index_type>;
    using VT = value_type;
    using real_type = gko::remove_complex<value_type>;
    FixedCoarsening()
        : exec(gko::ReferenceExecutor::create()),
          coarse_rows(exec, {0, 2, 3}),
          fixed_coarsening_factory(MgLevel::build()
                                       .with_coarse_rows(coarse_rows)
                                       .with_skip_sorting(true)
                                       .on(exec)),
          fine_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({-1.0, 2.0}), I<VT>({0.0, -1.0}),
               I<VT>({3.0, -2.0}), I<VT>({-2.0, 1.0})},
              exec)),
          coarse_b(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({3.0, 1.0}), I<VT>({0.0, -1.0})},
              exec)),
          restrict_ans(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, -1.0}), I<VT>({3.0, -2.0})},
              exec)),
          prolong_applyans(gko::initialize<Vec>(
              {I<VT>({2.0, -1.0}), I<VT>({0.0, 0.0}), I<VT>({3.0, 1.0}),
               I<VT>({0.0, -1.0}), I<VT>({0.0, 0.0})},
              exec)),
          fine_x(gko::initialize<Vec>(
              {I<VT>({-2.0, -1.0}), I<VT>({1.0, -1.0}), I<VT>({-1.0, -1.0}),
               I<VT>({0.0, 0.0}), I<VT>({0.0, 2.0})},
              exec)),
          mtx(Mtx::create(exec, gko::dim<2>(5, 5), 15,
                          std::make_shared<typename Mtx::classical>())),
          coarse(Mtx::create(exec, gko::dim<2>(3, 3), 5,
                             std::make_shared<typename Mtx::classical>())),
          gen_coarse_rows(exec, 5)
    {
        this->create_mtx(mtx.get(), &gen_coarse_rows, coarse.get());
        mg_level = fixed_coarsening_factory->generate(mtx);
    }

    void create_mtx(Mtx* fine, gko::array<index_type>* coarse_rows, Mtx* coarse)
    {
        auto coarse_rows_val = coarse_rows->get_data();
        coarse_rows_val[0] = 0;
        coarse_rows_val[1] = -1;
        coarse_rows_val[2] = 1;
        coarse_rows_val[3] = 2;
        coarse_rows_val[4] = -1;

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


        /* this coarse is stored:
         *  5 -3  0
         * -3  5  0
         *  0  0  5
         */
        coarse->read(
            {{3, 3},
             {{0, 0, 5}, {0, 1, -3}, {1, 0, -3}, {1, 1, 5}, {2, 2, 5}}});
    }

    static void assert_same_coarse_rows(const index_type* m1,
                                        const index_type* m2,
                                        gko::size_type len)
    {
        for (gko::size_type i = 0; i < len; ++i) {
            EXPECT_EQ(m1[i], m2[i]);
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> coarse;
    gko::array<index_type> coarse_rows;
    gko::array<index_type> gen_coarse_rows;
    std::shared_ptr<Vec> coarse_b;
    std::shared_ptr<Vec> fine_b;
    std::shared_ptr<Vec> restrict_ans;
    std::shared_ptr<Vec> prolong_applyans;
    std::shared_ptr<Vec> fine_x;
    std::unique_ptr<typename MgLevel::Factory> fixed_coarsening_factory;
    std::unique_ptr<MgLevel> mg_level;
};

TYPED_TEST_SUITE(FixedCoarsening, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(FixedCoarsening, Generate)
{
    ASSERT_NO_THROW(this->fixed_coarsening_factory->generate(this->mtx));
}


TYPED_TEST(FixedCoarsening, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy =
        this->fixed_coarsening_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->mg_level);
    auto copy_mtx = copy->get_system_matrix();
    auto copy_coarse = copy->get_coarse_op();

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_coarse), this->coarse, 0.0);
}


TYPED_TEST(FixedCoarsening, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto copy =
        this->fixed_coarsening_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->mg_level);
    auto copy_mtx = copy->get_system_matrix();
    auto copy_coarse = copy->get_coarse_op();

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_mtx), this->mtx, 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(copy_coarse), this->coarse, 0.0);
}


TYPED_TEST(FixedCoarsening, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto clone = this->mg_level->clone();
    auto clone_mtx = clone->get_system_matrix();
    auto clone_coarse = clone->get_coarse_op();

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_mtx), this->mtx, 0.0);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(clone_coarse), this->coarse, 0.0);
}


TYPED_TEST(FixedCoarsening, CanBeCleared)
{
    using MgLevel = typename TestFixture::MgLevel;

    this->mg_level->clear();
    auto mtx = this->mg_level->get_system_matrix();
    auto coarse = this->mg_level->get_coarse_op();

    ASSERT_EQ(mtx, nullptr);
    ASSERT_EQ(coarse, nullptr);
}


TYPED_TEST(FixedCoarsening, CoarseFineRestrictApply)
{
    auto fixed_coarsening = this->fixed_coarsening_factory->generate(this->mtx);
    using Vec = typename TestFixture::Vec;
    using value_type = typename TestFixture::value_type;
    auto x = Vec::create_with_config_of(this->coarse_b);

    fixed_coarsening->get_restrict_op()->apply(this->fine_b, x);

    GKO_ASSERT_MTX_NEAR(x, this->restrict_ans, r<value_type>::value);
}


TYPED_TEST(FixedCoarsening, CoarseFineProlongApply)
{
    using value_type = typename TestFixture::value_type;
    auto fixed_coarsening = this->fixed_coarsening_factory->generate(this->mtx);
    auto x = gko::clone(this->fine_x);

    fixed_coarsening->get_prolong_op()->apply(this->coarse_b, x);

    GKO_ASSERT_MTX_NEAR(x, this->prolong_applyans, r<value_type>::value);
}


TYPED_TEST(FixedCoarsening, Apply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto fixed_coarsening = this->fixed_coarsening_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = fixed_coarsening->get_executor();
    auto answer = gko::initialize<Vec>(
        {I<VT>({-7.0, -2.0}), I<VT>({0.0, 0.0}), I<VT>({1.0, -2.0}),
         I<VT>({0.0, 0.0}), I<VT>({0.0, 0.0})},
        exec);

    fixed_coarsening->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(FixedCoarsening, AdvancedApply)
{
    using VT = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto fixed_coarsening = this->fixed_coarsening_factory->generate(this->mtx);
    auto b = gko::clone(this->fine_x);
    auto x = gko::clone(this->fine_x);
    auto exec = fixed_coarsening->get_executor();
    auto alpha = gko::initialize<Vec>({1.0}, exec);
    auto beta = gko::initialize<Vec>({2.0}, exec);
    auto answer = gko::initialize<Vec>(
        {I<VT>({-11.0, -4.0}), I<VT>({2.0, -2.0}), I<VT>({-1.0, -4.0}),
         I<VT>({0.0, 0.0}), I<VT>({0.0, 4.0})},
        exec);

    fixed_coarsening->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, answer, r<VT>::value);
}


TYPED_TEST(FixedCoarsening, GenerateMgLevel)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    auto prolong_op = gko::share(Mtx::create(this->exec, gko::dim<2>{5, 3}, 0));
    // 0-2-3
    prolong_op->read({{5, 3}, {{0, 0, 1}, {2, 1, 1}, {3, 2, 1}}});
    auto restrict_op = gko::share(gko::as<Mtx>(prolong_op->transpose()));

    auto coarse_fine = this->fixed_coarsening_factory->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_restrict_op()),
                        restrict_op, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_prolong_op()), prolong_op,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_coarse_op()),
                        this->coarse, r<value_type>::value);
}


TYPED_TEST(FixedCoarsening, GenerateMgLevelOnUnsortedCsrMatrix)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto coarse_rows = gko::array<index_type>(this->exec, {0, 2, 3});
    auto mglevel_sort =
        MgLevel::build().with_coarse_rows(coarse_rows).on(this->exec);
    /* this unsorted matrix is stored as this->fine:
     *  5 -3 -3  0  0
     * -3  5  0 -2 -1
     * -3  0  5  0 -1
     *  0 -3  0  5  0
     *  0 -2 -2  0  5
     */
    auto mtx_values = {-3, -3, 5, -3, -2, -1, 5, -3, -1, 5, -3, 5, -2, -2, 5};
    auto mtx_col_idxs = {1, 2, 0, 0, 3, 4, 1, 0, 4, 2, 1, 3, 1, 2, 4};
    auto mtx_row_ptrs = {0, 3, 7, 10, 12, 15};
    auto matrix = gko::share(
        Mtx::create(this->exec, gko::dim<2>{5, 5}, std::move(mtx_values),
                    std::move(mtx_col_idxs), std::move(mtx_row_ptrs)));
    auto prolong_op = gko::share(Mtx::create(this->exec, gko::dim<2>{5, 3}, 0));
    // 0-2-3
    prolong_op->read({{5, 3}, {{0, 0, 1}, {2, 1, 1}, {3, 2, 1}}});
    auto restrict_op = gko::share(gko::as<Mtx>(prolong_op->transpose()));

    auto coarse_fine = mglevel_sort->generate(matrix);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_restrict_op()),
                        restrict_op, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_prolong_op()), prolong_op,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_coarse_op()),
                        this->coarse, r<value_type>::value);
}


TYPED_TEST(FixedCoarsening, GenerateMgLevelOnUnsortedCooMatrix)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using CooMtx = typename TestFixture::CooMtx;
    using Mtx = typename TestFixture::Mtx;
    using MgLevel = typename TestFixture::MgLevel;
    auto coarse_rows = gko::array<index_type>(this->exec, {0, 2, 3});
    auto mglevel_sort =
        MgLevel::build().with_coarse_rows(coarse_rows).on(this->exec);
    /* this unsorted matrix is stored as this->fine:
     *  5 -3 -3  0  0
     * -3  5  0 -2 -1
     * -3  0  5  0 -1
     *  0 -3  0  5  0
     *  0 -2 -2  0  5
     */
    auto mtx_values = {-3, -3, 5, -3, -2, -1, 5, -3, -1, 5, -3, 5, -2, -2, 5};
    auto mtx_col_idxs = {1, 2, 0, 0, 3, 4, 1, 0, 4, 2, 1, 3, 1, 2, 4};
    auto mtx_row_idxs = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4};
    auto matrix = gko::share(
        CooMtx::create(this->exec, gko::dim<2>{5, 5}, std::move(mtx_values),
                       std::move(mtx_col_idxs), std::move(mtx_row_idxs)));
    auto prolong_op = gko::share(Mtx::create(this->exec, gko::dim<2>{5, 3}, 0));
    // 0-2-3
    prolong_op->read({{5, 3}, {{0, 0, 1}, {2, 1, 1}, {3, 2, 1}}});
    auto restrict_op = gko::share(gko::as<Mtx>(prolong_op->transpose()));

    auto coarse_fine = mglevel_sort->generate(matrix);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_restrict_op()),
                        restrict_op, r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_prolong_op()), prolong_op,
                        r<value_type>::value);
    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(coarse_fine->get_coarse_op()),
                        this->coarse, r<value_type>::value);
}


}  // namespace
