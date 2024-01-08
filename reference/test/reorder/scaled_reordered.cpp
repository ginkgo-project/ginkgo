// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/reorder/scaled_reordered.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/reorder/rcm.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


namespace {


template <typename ValueIndexType>
class ScaledReordered : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Diag = gko::matrix::Diagonal<value_type>;
    using SR =
        gko::experimental::reorder::ScaledReordered<value_type, index_type>;
    using Cg = gko::solver::Cg<value_type>;
    using Bicgstab = gko::solver::Bicgstab<value_type>;
    using Rcm = gko::reorder::Rcm<value_type, index_type>;

    ScaledReordered()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>({{1., 1., 0.}, {1., 2., 1.}, {0., 1., 2.}},
                                   exec)),
          row_scaled(gko::initialize<Mtx>(
              {{1., 1., 0.}, {.5, 1., .5}, {0., .25, .5}}, exec)),
          col_scaled(gko::initialize<Mtx>(
              {{1., .5, 0.}, {1., 1., .25}, {0., .5, .5}}, exec)),
          row_col_scaled(gko::initialize<Mtx>(
              {{1., .5, 0.}, {.5, .5, .125}, {0., .125, .125}}, exec)),
          rectangular_mtx(
              gko::initialize<Mtx>({{1., 1., 0}, {1., 2., 1.}}, exec)),
          // clang-format off
          rcm_mtx(gko::initialize<Mtx>(
                                        {{1.0, 2.0, 0.0, -1.3, 2.1},
                                         {2.0, 5.0, 1.5, 0.0, 0.0},
                                         {0.0, 1.5, 1.5, 1.1, 0.0},
                                         {-1.3, 0.0, 1.1, 2.0, 0.0},
                                         {2.1, 0.0, 0.0, 0.0, 1.0}},
                                        exec)),
          rcm_mtx_reordered(gko::initialize<Mtx>(
                                        {{1.5, 1.1, 1.5, 0.0, 0.0},
                                         {1.1, 2.0, 0.0, -1.3, 0.0},
                                         {1.5, 0.0, 5.0, 2.0, 0.0},
                                         {0.0, -1.3, 2.0, 1.0, 2.1},
                                         {0.0, 0.0, 0.0, 2.1, 1.0}},
                                        exec)),
          // clang-format on
          diag(Diag::create(exec, 3)),
          diag2(Diag::create(exec, 5)),
          diag3(Diag::create(exec, 5)),
          x(gko::initialize<Vec>({1., 2., 3., 4., 5.}, exec)),
          b(gko::initialize<Vec>({10.3, 16.5, 11.9, 10., 7.1}, exec)),
          rcm_factory(Rcm::build().on(exec)),
          solver_factory(
              Bicgstab::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(100u),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(r<value_type>::value))
                  .on(exec)),
          tol{r<value_type>::value}
    {
        this->create_diag(diag.get());
        this->create_equilibration(diag2.get(), diag3.get());
    }

    void create_diag(Diag* d)
    {
        auto* v = d->get_values();
        v[0] = 1.;
        v[1] = .5;
        v[2] = .25;
    }

    void create_equilibration(Diag* d1, Diag* d2)
    {
        auto* v1 = d1->get_values();
        v1[0] = 1.0648;
        v1[1] = 0.4472;
        v1[2] = 1.4907;
        v1[3] = 1.1180;
        v1[4] = 1.0648;
        auto* v2 = d2->get_values();
        v2[0] = 0.4472;
        v2[1] = 0.4472;
        v2[2] = 0.4472;
        v2[3] = 0.4472;
        v2[4] = 0.4472;
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> row_scaled;
    std::shared_ptr<Mtx> col_scaled;
    std::shared_ptr<Mtx> row_col_scaled;
    std::shared_ptr<Mtx> rcm_mtx;
    std::shared_ptr<Mtx> rcm_mtx_reordered;
    std::shared_ptr<Mtx> rectangular_mtx;
    std::shared_ptr<Diag> diag;
    std::shared_ptr<Diag> diag2;
    std::shared_ptr<Diag> diag3;
    std::shared_ptr<typename Rcm::Factory> rcm_factory;
    std::shared_ptr<typename Bicgstab::Factory> solver_factory;
    std::shared_ptr<Vec> b;
    std::shared_ptr<Vec> x;
    gko::remove_complex<value_type> tol;
};

TYPED_TEST_SUITE(ScaledReordered, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(ScaledReordered, BuildsWithOnlySystemMatrix)
{
    using SR = typename TestFixture::SR;
    using Mtx = typename TestFixture::Mtx;
    auto scaled_reordered_fact = SR::build().on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(scaled_reordered->get_system_matrix()),
                        this->mtx, this->tol);
}


TYPED_TEST(ScaledReordered, BuildsWithRowScaling)
{
    using SR = typename TestFixture::SR;
    using Mtx = typename TestFixture::Mtx;
    auto scaled_reordered_fact =
        SR::build().with_row_scaling(this->diag).on(this->exec);

    auto scaled_reordered = scaled_reordered_fact->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(scaled_reordered->get_system_matrix()),
                        this->row_scaled, this->tol);
}


TYPED_TEST(ScaledReordered, BuildsWithColScaling)
{
    using SR = typename TestFixture::SR;
    using Mtx = typename TestFixture::Mtx;
    auto scaled_reordered_fact =
        SR::build().with_col_scaling(this->diag).on(this->exec);

    auto scaled_reordered = scaled_reordered_fact->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(scaled_reordered->get_system_matrix()),
                        this->col_scaled, this->tol);
}


TYPED_TEST(ScaledReordered, BuildsWithRowAndColScaling)
{
    using SR = typename TestFixture::SR;
    using Mtx = typename TestFixture::Mtx;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag)
                                     .with_col_scaling(this->diag)
                                     .on(this->exec);

    auto scaled_reordered = scaled_reordered_fact->generate(this->mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(scaled_reordered->get_system_matrix()),
                        this->row_col_scaled, this->tol);
}


TYPED_TEST(ScaledReordered, BuildsWithRcmReordering)
{
    using SR = typename TestFixture::SR;
    using Mtx = typename TestFixture::Mtx;
    auto scaled_reordered_fact =
        SR::build().with_reordering(this->rcm_factory).on(this->exec);

    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);

    GKO_ASSERT_MTX_NEAR(gko::as<Mtx>(scaled_reordered->get_system_matrix()),
                        this->rcm_mtx_reordered, this->tol);
}


TYPED_TEST(ScaledReordered, ThrowOnNonSquareMatrix)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build().on(this->exec);

    ASSERT_THROW(scaled_reordered_fact->generate(this->rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(ScaledReordered, CanBeCopied)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag)
                                     .with_col_scaling(this->diag)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->mtx);
    auto before_system_matrix = scaled_reordered->get_system_matrix();
    auto before_inner_operator = scaled_reordered->get_inner_operator();
    auto copied = SR::build().on(this->exec)->generate(this->rcm_mtx);

    copied->copy_from(scaled_reordered);

    ASSERT_EQ(before_system_matrix, copied->get_system_matrix());
    ASSERT_EQ(before_inner_operator, copied->get_inner_operator());
}


TYPED_TEST(ScaledReordered, CanBeMoved)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag)
                                     .with_col_scaling(this->diag)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->mtx);
    auto before_system_matrix = scaled_reordered->get_system_matrix();
    auto before_inner_operator = scaled_reordered->get_inner_operator();
    auto moved = SR::build().on(this->exec)->generate(this->rcm_mtx);

    moved->move_from(scaled_reordered);

    ASSERT_EQ(before_system_matrix, moved->get_system_matrix());
    ASSERT_EQ(before_inner_operator, moved->get_inner_operator());
}


TYPED_TEST(ScaledReordered, CanBeCloned)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag)
                                     .with_col_scaling(this->diag)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->mtx);
    auto before_system_matrix = scaled_reordered->get_system_matrix();
    auto before_inner_operator = scaled_reordered->get_inner_operator();

    auto cloned = scaled_reordered->clone();

    ASSERT_EQ(before_system_matrix, cloned->get_system_matrix());
    ASSERT_EQ(before_inner_operator, cloned->get_inner_operator());
}


TYPED_TEST(ScaledReordered, AppliesWithoutOperators)
{
    using SR = typename TestFixture::SR;
    using Vec = typename TestFixture::Vec;
    auto scaled_reordered_fact = SR::build().on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->b, this->tol);
}


TYPED_TEST(ScaledReordered, AppliesWithOnlyRowScaling)
{
    using SR = typename TestFixture::SR;
    using Vec = typename TestFixture::Vec;
    auto scaled_reordered_fact =
        SR::build().with_row_scaling(this->diag2).on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);
    auto expected_result = this->b->clone();
    this->diag2->apply(this->b, expected_result);

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, expected_result, this->tol);
}


TYPED_TEST(ScaledReordered, AppliesWithOnlyColScaling)
{
    using SR = typename TestFixture::SR;
    using Vec = typename TestFixture::Vec;
    auto scaled_reordered_fact =
        SR::build().with_col_scaling(this->diag2).on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);
    auto expected_result = this->b->clone();
    this->diag2->apply(this->b, expected_result);

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, expected_result, this->tol);
}


TYPED_TEST(ScaledReordered, AppliesWithRowAndColScaling)
{
    using SR = typename TestFixture::SR;
    using Vec = typename TestFixture::Vec;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag2)
                                     .with_col_scaling(this->diag3)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);
    auto expected_result = this->b->clone();
    this->diag2->apply(this->b, expected_result);
    this->diag3->apply(expected_result, expected_result);

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, expected_result, this->tol);
}


TYPED_TEST(ScaledReordered, AppliesWithRcmReordering)
{
    using SR = typename TestFixture::SR;
    using Vec = typename TestFixture::Vec;
    auto scaled_reordered_fact =
        SR::build().with_reordering(this->rcm_factory).on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->b, this->tol);
}


TYPED_TEST(ScaledReordered, SolvesSingleRhsWithOnlyInnerOperator)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact =
        SR::build().with_inner_operator(this->solver_factory).on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = this->b->clone();

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->x, 15 * this->tol);
}


TYPED_TEST(ScaledReordered, SolvesSingleRhsWithRowScaling)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag2)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = this->b->clone();

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->x, 15 * this->tol);
}


TYPED_TEST(ScaledReordered, SolvesSingleRhsWithColScaling)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build()
                                     .with_col_scaling(this->diag3)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = this->b->clone();

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->x, 15 * this->tol);
}


TYPED_TEST(ScaledReordered, SolvesSingleRhsWithRcmReordering)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build()
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = this->b->clone();

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->x, 15 * this->tol);
}


TYPED_TEST(ScaledReordered, SolvesSingleRhsWithScalingAndRcmReordering)
{
    using SR = typename TestFixture::SR;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag2)
                                     .with_col_scaling(this->diag3)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = this->b->clone();

    scaled_reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->x, 15 * this->tol);
}


TYPED_TEST(ScaledReordered, SolvesSingleRhsWithScalingAndRcmReorderingMixed)
{
    using SR = typename TestFixture::SR;
    using T = typename TestFixture::value_type;
    using Vec = gko::matrix::Dense<gko::next_precision<T>>;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag2)
                                     .with_col_scaling(this->diag3)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto x = gko::initialize<Vec>({1., 2., 3., 4., 5.}, this->exec);
    auto b = gko::initialize<Vec>({10.3, 16.5, 11.9, 10., 7.1}, this->exec);
    auto res = b->clone();

    scaled_reordered->apply(b, res);

    GKO_ASSERT_MTX_NEAR(res, x, 1e-5);
}


TYPED_TEST(ScaledReordered, AdvancedSolvesSingleRhsWithScalingAndRcmReordering)
{
    using SR = typename TestFixture::SR;
    using Vec = typename TestFixture::Vec;
    const auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    const auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag2)
                                     .with_col_scaling(this->diag3)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto res = this->b->clone();

    scaled_reordered->apply(alpha, this->b, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({-8.3, -12.5, -5.9, -2., 2.9}), 15 * this->tol);
}


TYPED_TEST(ScaledReordered,
           AdvancedSolvesSingleRhsWithScalingAndRcmReorderingMixed)
{
    using SR = typename TestFixture::SR;
    using T = typename TestFixture::value_type;
    using value_type = gko::next_precision<T>;
    using Vec = gko::matrix::Dense<value_type>;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag2)
                                     .with_col_scaling(this->diag3)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    const auto alpha = gko::initialize<Vec>({2.0}, this->exec);
    const auto beta = gko::initialize<Vec>({-1.0}, this->exec);
    auto x = gko::initialize<Vec>({1., 2., 3., 4., 5.}, this->exec);
    auto b = gko::initialize<Vec>({10.3, 16.5, 11.9, 10., 7.1}, this->exec);
    auto res = b->clone();

    scaled_reordered->apply(alpha, b, beta, res);

    GKO_ASSERT_MTX_NEAR(res, l({-8.3, -12.5, -5.9, -2., 2.9}), 1e-5);
}


TYPED_TEST(ScaledReordered, SolvesMultipleRhs)
{
    using SR = typename TestFixture::SR;
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto scaled_reordered_fact = SR::build()
                                     .with_row_scaling(this->diag2)
                                     .with_col_scaling(this->diag3)
                                     .with_reordering(this->rcm_factory)
                                     .with_inner_operator(this->solver_factory)
                                     .on(this->exec);
    auto scaled_reordered = scaled_reordered_fact->generate(this->rcm_mtx);
    auto x = gko::initialize<Vec>(
        {I<T>{1., 2.}, I<T>{2., 4.}, I<T>{3., 6.}, I<T>{4., 8.}, I<T>{5., 10.}},
        this->exec);
    auto b = gko::initialize<Vec>(
        {I<T>{10.3, 20.6}, I<T>{16.5, 33.}, I<T>{11.9, 23.8}, I<T>{10., 20.},
         I<T>{7.1, 14.2}},
        this->exec);
    auto res = b->clone();

    scaled_reordered->apply(b, res);

    GKO_ASSERT_MTX_NEAR(res, x, 15 * this->tol);
}


}  // namespace


GKO_END_DISABLE_DEPRECATION_WARNINGS
