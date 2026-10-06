// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/preconditioner/nullspace_projection.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/nullspace.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>

#include "core/test/utils.hpp"


namespace {


template <typename T>
class NullspaceProjection : public ::testing::Test {
protected:
    using value_type = T;
    using vec = gko::matrix::Dense<value_type>;
    using Csr = gko::matrix::Csr<value_type, int>;
    using Nullspace = gko::solver::Nullspace<value_type>;
    using Projection = gko::preconditioner::NullspaceProjection<value_type>;
    using Jacobi = gko::preconditioner::Jacobi<value_type, int>;

    NullspaceProjection()
        : exec(gko::ReferenceExecutor::create()),
          mtx(laplacian(6)),
          constant(gko::share(Nullspace::create_from_constant(exec))),
          jacobi(gko::share(
              Jacobi::build().with_max_block_size(1u).on(exec)->generate(mtx)))
    {}

    std::shared_ptr<Csr> laplacian(int n)
    {
        gko::matrix_data<value_type, int> data(gko::dim<2>(n, n));
        for (int i = 0; i < n; ++i) {
            value_type diag{};
            if (i > 0) {
                data.nonzeros.emplace_back(i, i - 1, -gko::one<value_type>());
                diag += gko::one<value_type>();
            }
            if (i < n - 1) {
                data.nonzeros.emplace_back(i, i + 1, -gko::one<value_type>());
                diag += gko::one<value_type>();
            }
            data.nonzeros.emplace_back(i, i, diag);
        }
        data.sort_row_major();
        auto result = gko::share(Csr::create(exec));
        result->read(data);
        return result;
    }

    std::unique_ptr<vec> test_vector(gko::size_type num_cols)
    {
        auto result =
            vec::create(exec, gko::dim<2>{mtx->get_size()[0], num_cols});
        for (gko::size_type i = 0; i < result->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < num_cols; ++j) {
                result->at(i, j) =
                    static_cast<value_type>((3 * i + 5 * j) % 7) -
                    gko::one<value_type>();
            }
        }
        return result;
    }

    std::shared_ptr<const gko::ReferenceExecutor> exec;
    std::shared_ptr<Csr> mtx;
    std::shared_ptr<Nullspace> constant;
    std::shared_ptr<Jacobi> jacobi;
};

TYPED_TEST_SUITE(NullspaceProjection, gko::test::ValueTypes,
                 TypenameNameGenerator);


struct AllocationCounter : gko::log::Logger {
    AllocationCounter()
        : gko::log::Logger(gko::log::Logger::allocation_completed_mask)
    {}

    void on_allocation_completed(const gko::Executor*, const gko::size_type&,
                                 const gko::uintptr&) const override
    {
        ++count;
    }

    mutable int count = 0;
};


TYPED_TEST(NullspaceProjection, ProjectsPreconditionedVector)
{
    using value_type = typename TestFixture::value_type;
    using Projection = typename TestFixture::Projection;
    auto projection = Projection::build()
                          .with_generated_preconditioner(this->jacobi)
                          .with_nullspace(this->constant)
                          .on(this->exec)
                          ->generate(this->mtx);
    auto in = this->test_vector(2);
    auto z = this->test_vector(2);
    auto expected = this->test_vector(2);
    this->jacobi->apply(in, expected);
    projection->get_nullspace()->project(expected);

    projection->apply(in, z);

    GKO_ASSERT_MTX_NEAR(z, expected, r<value_type>::value);
}


TYPED_TEST(NullspaceProjection, ProjectsInputAndPreconditionedVector)
{
    using value_type = typename TestFixture::value_type;
    using Projection = typename TestFixture::Projection;
    auto projection = Projection::build()
                          .with_generated_preconditioner(this->jacobi)
                          .with_nullspace(this->constant)
                          .with_left_nullspace(this->constant)
                          .on(this->exec)
                          ->generate(this->mtx);
    auto in = this->test_vector(2);
    auto z = this->test_vector(2);
    auto projected_in = gko::clone(in);
    projection->get_left_nullspace()->project(projected_in);
    auto expected = this->test_vector(2);
    this->jacobi->apply(projected_in, expected);
    projection->get_nullspace()->project(expected);

    projection->apply(in, z);

    GKO_ASSERT_MTX_NEAR(z, expected, r<value_type>::value);
}


TYPED_TEST(NullspaceProjection, UsesIdentityWithoutPreconditioner)
{
    using value_type = typename TestFixture::value_type;
    using Projection = typename TestFixture::Projection;
    auto projection = Projection::build()
                          .with_nullspace(this->constant)
                          .on(this->exec)
                          ->generate(this->mtx);
    auto in = this->test_vector(1);
    auto z = this->test_vector(1);
    auto expected = gko::clone(in);
    projection->get_nullspace()->project(expected);

    projection->apply(in, z);

    GKO_ASSERT_MTX_NEAR(z, expected, r<value_type>::value);
}


TYPED_TEST(NullspaceProjection, AdvancedApplyMatchesApply)
{
    using value_type = typename TestFixture::value_type;
    using vec = typename TestFixture::vec;
    using Projection = typename TestFixture::Projection;
    auto projection = Projection::build()
                          .with_generated_preconditioner(this->jacobi)
                          .with_nullspace(this->constant)
                          .with_left_nullspace(this->constant)
                          .on(this->exec)
                          ->generate(this->mtx);
    auto alpha = gko::initialize<vec>({value_type{2}}, this->exec);
    auto beta = gko::initialize<vec>({value_type{-1}}, this->exec);
    auto in = this->test_vector(2);
    auto x = this->test_vector(2);
    auto expected = this->test_vector(2);
    projection->apply(in, expected);
    expected->scale(alpha);
    expected->sub_scaled(
        gko::initialize<vec>({gko::one<value_type>()}, this->exec), x);

    projection->apply(alpha, in, beta, x);

    GKO_ASSERT_MTX_NEAR(x, expected, r<value_type>::value);
}


TYPED_TEST(NullspaceProjection, AdaptsConstantNullspaceToMatrixSize)
{
    using Projection = typename TestFixture::Projection;
    auto factory = Projection::build()
                       .with_preconditioner(TestFixture::Jacobi::build())
                       .with_nullspace(this->constant)
                       .on(this->exec);

    auto projection4 = factory->generate(this->laplacian(4));
    auto projection9 = factory->generate(this->laplacian(9));

    ASSERT_EQ(projection4->get_nullspace()->get_size(), (gko::dim<2>{4, 4}));
    ASSERT_EQ(projection9->get_nullspace()->get_size(), (gko::dim<2>{9, 9}));
}


TYPED_TEST(NullspaceProjection, ConjTransposeSwapsNullspaces)
{
    using Projection = typename TestFixture::Projection;
    auto projection = Projection::build()
                          .with_generated_preconditioner(this->jacobi)
                          .with_nullspace(this->constant)
                          .on(this->exec)
                          ->generate(this->mtx);

    auto trans = gko::as<Projection>(projection->conj_transpose());

    ASSERT_EQ(trans->get_nullspace(), nullptr);
    ASSERT_EQ(trans->get_left_nullspace(), projection->get_nullspace());
}


TYPED_TEST(NullspaceProjection, ApplyDoesNotAllocateAfterFirstUse)
{
    using Projection = typename TestFixture::Projection;
    auto exec = gko::ReferenceExecutor::create();
    auto projection = Projection::build()
                          .with_nullspace(this->constant)
                          .with_left_nullspace(this->constant)
                          .on(exec)
                          ->generate(this->mtx);
    auto in = gko::clone(exec, this->test_vector(2));
    auto z = gko::clone(exec, this->test_vector(2));
    projection->apply(in, z);
    auto logger = std::make_shared<AllocationCounter>();
    exec->add_logger(logger);

    projection->apply(in, z);
    projection->apply(in, z);

    exec->remove_logger(logger);
    ASSERT_EQ(logger->count, 0);
}


TYPED_TEST(NullspaceProjection, WorksAsPreconditionerOfAnySolver)
{
    using value_type = typename TestFixture::value_type;
    using vec = typename TestFixture::vec;
    using Projection = typename TestFixture::Projection;
    auto x_star =
        gko::initialize<vec>({value_type{3}, value_type{-1}, value_type{-4},
                              value_type{2}, value_type{-2}, value_type{2}},
                             this->exec);
    auto b = gko::clone(x_star);
    this->mtx->apply(x_star, b);
    auto ones = vec::create(this->exec, b->get_size());
    ones->fill(gko::one<value_type>());
    b->add_scaled(gko::initialize<vec>({value_type{3}}, this->exec), ones);
    auto x = this->test_vector(1);
    auto solver =
        gko::solver::Cg<value_type>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(100u),
                           gko::stop::ResidualNorm<value_type>::build()
                               .with_reduction_factor(r<value_type>::value))
            .with_preconditioner(
                Projection::build()
                    .with_preconditioner(TestFixture::Jacobi::build())
                    .with_nullspace(this->constant)
                    .with_left_nullspace(this->constant))
            .on(this->exec)
            ->generate(this->mtx);
    auto ns = gko::solver::Nullspace<value_type>::create_from_constant(
        this->exec, this->mtx->get_size());

    ns->project(b);
    ns->project(x);
    solver->apply(b, x);

    GKO_ASSERT_MTX_NEAR(x, x_star, 1000 * r<value_type>::value);
}


}  // namespace
