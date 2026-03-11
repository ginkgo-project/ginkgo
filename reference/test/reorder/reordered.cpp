// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/reorder/rcm.hpp>
#include <ginkgo/core/reorder/reordered.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"
#include "core/test/utils/factories.hpp"


template <typename ValueIndexType>
class Reordered : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using Op = gko::experimental::reorder::Reordered<value_type, index_type>;
    using Cg = gko::solver::Cg<value_type>;
    using Stop = gko::stop::Iteration;
    using Rcm = gko::experimental::reorder::Rcm<index_type>;
    using Perm = gko::matrix::Permutation<index_type>;

    Reordered()
        : exec(gko::ReferenceExecutor::create()),
          empty_mtx(Mtx::create(exec)),
          rcm_mtx(gko::initialize<Mtx>({{1.0, 2.0, 0.0, -1.3, 2.1},
                                        {2.0, 5.0, 1.5, 0.0, 0.0},
                                        {0.0, 1.5, 1.5, 1.1, 0.0},
                                        {-1.3, 0.0, 1.1, 2.0, 0.0},
                                        {2.1, 0.0, 0.0, 0.0, 1.0}},
                                       exec)),
          rcm_mtx_reordered(gko::initialize<Mtx>({{1.5, 1.1, 1.5, 0.0, 0.0},
                                                  {1.1, 2.0, 0.0, -1.3, 0.0},
                                                  {1.5, 0.0, 5.0, 2.0, 0.0},
                                                  {0.0, -1.3, 2.0, 1.0, 2.1},
                                                  {0.0, 0.0, 0.0, 2.1, 1.0}},
                                                 exec)),
          zero_rcm_mtx(rcm_mtx->clone()),
          empty_rcm_mtx(Mtx::create(exec, rcm_mtx->get_size())),
          rectangular_mtx(
              gko::initialize<Mtx>({{1., 1., 0}, {1., 2., 1.}}, exec)),
          rcm_perm(Perm::create(exec,
                                gko::array<index_type>(exec, {2, 3, 1, 0, 4}))),
          stop(Stop::build().with_max_iters(10).on(exec)),
          x(gko::initialize<Vec>({1., 2., 3., 4., 5.}, exec)),
          b(gko::initialize<Vec>({10.3, 16.5, 11.9, 10., 7.1}, exec)),
          reordered_fact{
              Op::build()
                  .with_reordering(Rcm::build())
                  .with_inner_operator(Cg::build().with_criteria(stop))
                  .on(exec)}
    {
        std::fill_n(empty_rcm_mtx->get_values(),
                    empty_rcm_mtx->get_num_stored_elements(),
                    gko::zero<value_type>());
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> empty_mtx;
    std::shared_ptr<Mtx> rcm_mtx;
    std::shared_ptr<Mtx> rcm_mtx_reordered;
    std::shared_ptr<Mtx> zero_rcm_mtx;
    std::shared_ptr<Mtx> empty_rcm_mtx;
    std::shared_ptr<Mtx> rectangular_mtx;
    std::shared_ptr<Perm> rcm_perm;
    std::shared_ptr<Stop::Factory> stop;
    std::shared_ptr<Vec> b;
    std::shared_ptr<Vec> x;
    std::shared_ptr<typename Op::Factory> reordered_fact;
};

TYPED_TEST_SUITE(Reordered, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Reordered, GeneratesCorrectly)
{
    using Mtx = typename TestFixture::Mtx;
    using Cg = typename TestFixture::Cg;

    auto reordered = this->reordered_fact->generate(this->rcm_mtx);

    GKO_ASSERT_MTX_EQ_SPARSITY(reordered->get_permutation(), this->rcm_perm);
    auto inner_op = gko::as<Cg>(reordered->get_inner_operator());
    auto inner_mtx = gko::as<Mtx>(inner_op->get_system_matrix());
    GKO_ASSERT_MTX_NEAR(inner_mtx, this->rcm_mtx_reordered, 0);
    ASSERT_EQ(inner_op->get_parameters().criteria.front(), this->stop);
}


TYPED_TEST(Reordered, GeneratesReuseCorrectly)
{
    using Mtx = typename TestFixture::Mtx;
    using Cg = typename TestFixture::Cg;

    auto data = this->reordered_fact->create_empty_reuse_data();
    auto reordered = this->reordered_fact->generate_reuse(this->rcm_mtx, *data);

    GKO_ASSERT_MTX_EQ_SPARSITY(reordered->get_permutation(), this->rcm_perm);
    auto inner_op = gko::as<Cg>(reordered->get_inner_operator());
    auto inner_mtx = gko::as<Mtx>(inner_op->get_system_matrix());
    GKO_ASSERT_MTX_NEAR(inner_mtx, this->rcm_mtx_reordered, 0);
    ASSERT_EQ(inner_op->get_parameters().criteria.front(), this->stop);
}


TYPED_TEST(Reordered, GeneratesReuseUpdateCorrectly)
{
    using Mtx = typename TestFixture::Mtx;
    using Cg = typename TestFixture::Cg;

    auto data = this->reordered_fact->create_empty_reuse_data();
    this->reordered_fact->generate_reuse(this->zero_rcm_mtx, *data);
    auto reordered = this->reordered_fact->generate_reuse(this->rcm_mtx, *data);

    GKO_ASSERT_MTX_EQ_SPARSITY(reordered->get_permutation(), this->rcm_perm);
    auto inner_op = gko::as<Cg>(reordered->get_inner_operator());
    auto inner_mtx = gko::as<Mtx>(inner_op->get_system_matrix());
    GKO_ASSERT_MTX_NEAR(inner_mtx, this->rcm_mtx_reordered, 0);
    ASSERT_EQ(inner_op->get_parameters().criteria.front(), this->stop);
}


TYPED_TEST(Reordered, ThrowOnNonSquareMatrix)
{
    ASSERT_THROW(this->reordered_fact->generate(this->rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(Reordered, ThrowOnNonSquareMatrixReuse)
{
    auto data = this->reordered_fact->create_empty_reuse_data();
    ASSERT_THROW(
        this->reordered_fact->generate_reuse(this->rectangular_mtx, *data),
        gko::DimensionMismatch);
}


TYPED_TEST(Reordered, ThrowOnIncorrectReuseType)
{
    auto data = gko::LinOpFactory::ReuseData{};
    ASSERT_THROW(
        this->reordered_fact->generate_reuse(this->rectangular_mtx, data),
        gko::NotSupported);
}


TYPED_TEST(Reordered, ThrowOnIncorrectReuseDimensions)
{
    auto data = this->reordered_fact->create_empty_reuse_data();
    this->reordered_fact->generate_reuse(this->rcm_mtx, *data);

    ASSERT_THROW(this->reordered_fact->generate_reuse(this->empty_mtx, *data),
                 gko::DimensionMismatch);
}


TYPED_TEST(Reordered, ThrowOnIncorrectReuseNnz)
{
    auto data = this->reordered_fact->create_empty_reuse_data();
    this->reordered_fact->generate_reuse(this->rcm_mtx, *data);

    ASSERT_THROW(
        this->reordered_fact->generate_reuse(this->empty_rcm_mtx, *data),
        gko::ValueMismatch);
}


TYPED_TEST(Reordered, ThrowOnDifferentExecutorReuse)
{
    using Op = typename TestFixture::Op;
    using Cg = typename TestFixture::Cg;
    using Rcm = typename TestFixture::Rcm;
    using Stop = typename TestFixture::Stop;
    auto data = this->reordered_fact->create_empty_reuse_data();
    this->reordered_fact->generate_reuse(this->rcm_mtx, *data);
    auto reordered_fact2 = Op::build()
                               .with_reordering(Rcm::build())
                               .with_inner_operator(Cg::build().with_criteria(
                                   Stop::build().with_max_iters(0)))
                               .on(gko::ReferenceExecutor::create());

    ASSERT_THROW(reordered_fact2->generate_reuse(this->rcm_mtx, *data),
                 gko::NotSupported);
}


TYPED_TEST(Reordered, CanBeCopied)
{
    auto reordered = this->reordered_fact->generate(this->rcm_mtx);
    auto before_inner_operator = reordered->get_inner_operator();
    auto before_permutation = reordered->get_permutation();
    auto copied = this->reordered_fact->generate(this->empty_mtx);

    copied->copy_from(reordered);

    ASSERT_EQ(before_inner_operator, copied->get_inner_operator());
    ASSERT_EQ(before_permutation, copied->get_permutation());
}


TYPED_TEST(Reordered, CanBeMoved)
{
    auto reordered = this->reordered_fact->generate(this->rcm_mtx);
    auto before_inner_operator = reordered->get_inner_operator();
    auto before_permutation = reordered->get_permutation();
    auto moved = this->reordered_fact->generate(this->empty_mtx);

    moved->move_from(reordered);

    ASSERT_EQ(before_inner_operator, moved->get_inner_operator());
    ASSERT_EQ(before_permutation, moved->get_permutation());
    ASSERT_FALSE(reordered->get_size());
}


TYPED_TEST(Reordered, CanBeCloned)
{
    auto reordered = this->reordered_fact->generate(this->rcm_mtx);
    auto before_inner_operator = reordered->get_inner_operator();
    auto before_permutation = reordered->get_permutation();

    auto cloned = reordered->clone();

    ASSERT_EQ(before_inner_operator, cloned->get_inner_operator());
    ASSERT_EQ(before_permutation, cloned->get_permutation());
}


TYPED_TEST(Reordered, AppliesWithPassthru)
{
    using Op = typename TestFixture::Op;
    using Rcm = typename TestFixture::Rcm;
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto passthru_factory = share(PassthruLinOpFactory::create(this->exec));
    auto passthru_reordered_fact = Op::build()
                                       .with_reordering(Rcm::build())
                                       .with_inner_operator(passthru_factory)
                                       .on(this->exec);
    auto reordered = passthru_reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);
    auto res2 = Vec::create_with_config_of(this->b);

    reordered->apply(this->b, res);
    this->rcm_mtx->apply(this->b, res2);

    GKO_ASSERT_MTX_NEAR(res, res2, r<value_type>::value);
    auto inner_op = gko::as<Mtx>(reordered->get_inner_operator());
    auto inner_op2 = gko::as<Mtx>(passthru_factory->get_last_op());
    GKO_ASSERT_MTX_NEAR(inner_op, inner_op2, 0);
    GKO_ASSERT_MTX_NEAR(inner_op, this->rcm_mtx_reordered, 0);
}


TYPED_TEST(Reordered, AppliesWithSolver)
{
    using value_type = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto reordered = this->reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);

    reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->x, 100 * r<value_type>::value);
}


TYPED_TEST(Reordered, AppliesWithSolverMultipleRhs)
{
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto reordered = this->reordered_fact->generate(this->rcm_mtx);
    auto res = Vec::create_with_config_of(this->b);
    auto x = gko::initialize<Vec>(
        {I<T>{1., 2.}, I<T>{2., 4.}, I<T>{3., 6.}, I<T>{4., 8.}, I<T>{5., 10.}},
        this->exec);
    auto b = gko::initialize<Vec>(
        {I<T>{10.3, 20.6}, I<T>{16.5, 33.}, I<T>{11.9, 23.8}, I<T>{10., 20.},
         I<T>{7.1, 14.2}},
        this->exec);

    reordered->apply(this->b, res);

    GKO_ASSERT_MTX_NEAR(res, this->x, 100 * r<T>::value);
}


TYPED_TEST(Reordered, AdvancedAppliesWithSolver)
{
    using T = typename TestFixture::value_type;
    using Vec = typename TestFixture::Vec;
    auto reordered = this->reordered_fact->generate(this->rcm_mtx);
    auto x = gko::initialize<Vec>({1., 2., 3., 4., 5.}, this->exec);
    auto b = gko::initialize<Vec>({10.3, 16.5, 11.9, 10., 7.1}, this->exec);
    auto alpha = gko::initialize<Vec>({3.0}, this->exec);
    auto beta = gko::initialize<Vec>({-1.0}, this->exec);

    reordered->apply(alpha, b, beta, x);

    GKO_ASSERT_MTX_NEAR(x, l({2.0, 4.0, 6.0, 8.0, 10.0}), 100 * r<T>::value);
}
