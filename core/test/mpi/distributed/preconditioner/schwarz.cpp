// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/config.hpp>
#include <ginkgo/core/distributed/preconditioner/schwarz.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/multigrid/pgm.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


template <typename ValueLocalGlobalIndexType>
class SchwarzFactory : public ::testing::Test {
protected:
    using value_type = typename std::tuple_element<
        0, decltype(ValueLocalGlobalIndexType())>::type;
    using local_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using global_index_type = typename std::tuple_element<
        1, decltype(ValueLocalGlobalIndexType())>::type;
    using Schwarz = gko::experimental::distributed::preconditioner::Schwarz<
        value_type, local_index_type, global_index_type>;
    using Jacobi = gko::preconditioner::Jacobi<value_type, local_index_type>;
    using Pgm = gko::multigrid::Pgm<value_type, local_index_type>;
    using Cg = gko::solver::Cg<value_type>;
    using Mtx =
        gko::experimental::distributed::Matrix<value_type, local_index_type,
                                               global_index_type>;

    SchwarzFactory()
        : exec(gko::ReferenceExecutor::create()),
          jacobi_factory(Jacobi::build().on(exec)),
          pgm_factory(Pgm::build().on(exec)),
          cg_factory(Cg::build().on(exec)),
          mtx(Mtx::create(exec, MPI_COMM_WORLD))
    {
        schwarz = Schwarz::build()
                      .with_local_solver(jacobi_factory)
                      .with_galerkin_ops(pgm_factory)
                      .with_coarse_solver(cg_factory)
                      .on(exec)
                      ->generate(mtx);
    }


    template <typename T>
    void init_array(T* arr, std::initializer_list<T> vals)
    {
        std::copy(std::begin(vals), std::end(vals), arr);
    }

    void assert_same_precond(gko::ptr_param<const Schwarz> a,
                             gko::ptr_param<const Schwarz> b)
    {
        ASSERT_EQ(a->get_size(), b->get_size());
        ASSERT_EQ(a->get_parameters().local_solver,
                  b->get_parameters().local_solver);
        ASSERT_EQ(a->get_parameters().galerkin_ops,
                  b->get_parameters().galerkin_ops);
        ASSERT_EQ(a->get_parameters().coarse_solver,
                  b->get_parameters().coarse_solver);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Schwarz> schwarz;
    std::shared_ptr<typename Jacobi::Factory> jacobi_factory;
    std::shared_ptr<typename Pgm::Factory> pgm_factory;
    std::shared_ptr<typename Cg::Factory> cg_factory;
    std::shared_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(SchwarzFactory, gko::test::ValueLocalGlobalIndexTypesBase,
                 TupleTypenameNameGenerator);


TYPED_TEST(SchwarzFactory, KnowsItsExecutor)
{
    ASSERT_EQ(this->schwarz->get_executor(), this->exec);
}


TYPED_TEST(SchwarzFactory, CanSetLocalFactory)
{
    ASSERT_EQ(this->schwarz->get_parameters().local_solver,
              this->jacobi_factory);
}


TYPED_TEST(SchwarzFactory, CanSetGalerkinOpsFactory)
{
    ASSERT_EQ(this->schwarz->get_parameters().galerkin_ops, this->pgm_factory);
}


TYPED_TEST(SchwarzFactory, CanSetCoarseSolverFactory)
{
    ASSERT_EQ(this->schwarz->get_parameters().coarse_solver, this->cg_factory);
}


TYPED_TEST(SchwarzFactory, CanBeCloned)
{
    auto schwarz_clone = clone(this->schwarz);

    this->assert_same_precond(schwarz_clone, this->schwarz);
}


TYPED_TEST(SchwarzFactory, CanBeCopied)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Pgm = typename TestFixture::Pgm;
    using Cg = typename TestFixture::Cg;
    using Schwarz = typename TestFixture::Schwarz;
    using Mtx = typename TestFixture::Mtx;
    auto bj = gko::share(Jacobi::build().on(this->exec));
    auto pgm = gko::share(Pgm::build().on(this->exec));
    auto cg = gko::share(Cg::build().on(this->exec));
    auto copy = Schwarz::build()
                    .with_local_solver(bj)
                    .with_galerkin_ops(pgm)
                    .with_coarse_solver(cg)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->copy_from(this->schwarz);

    this->assert_same_precond(copy, this->schwarz);
}


TYPED_TEST(SchwarzFactory, CanBeMoved)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Pgm = typename TestFixture::Pgm;
    using Cg = typename TestFixture::Cg;
    using Schwarz = typename TestFixture::Schwarz;
    using Mtx = typename TestFixture::Mtx;
    auto tmp = clone(this->schwarz);
    auto bj = gko::share(Jacobi::build().on(this->exec));
    auto pgm = gko::share(Pgm::build().on(this->exec));
    auto cg = gko::share(Cg::build().on(this->exec));
    auto copy = Schwarz::build()
                    .with_local_solver(bj)
                    .with_galerkin_ops(pgm)
                    .with_coarse_solver(cg)
                    .on(this->exec)
                    ->generate(Mtx::create(this->exec, MPI_COMM_WORLD));

    copy->move_from(this->schwarz);

    this->assert_same_precond(copy, tmp);
}


TYPED_TEST(SchwarzFactory, CanBeCleared)
{
    this->schwarz->clear();

    ASSERT_EQ(this->schwarz->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(this->schwarz->get_parameters().local_solver, nullptr);
    ASSERT_EQ(this->schwarz->get_parameters().galerkin_ops, nullptr);
    ASSERT_EQ(this->schwarz->get_parameters().coarse_solver, nullptr);
}


TYPED_TEST(SchwarzFactory, PassExplicitFactory)
{
    using Jacobi = typename TestFixture::Jacobi;
    using Schwarz = typename TestFixture::Schwarz;
    auto jacobi_factory = gko::share(Jacobi::build().on(this->exec));

    auto factory =
        Schwarz::build().with_local_solver(jacobi_factory).on(this->exec);

    ASSERT_EQ(factory->get_parameters().local_solver, jacobi_factory);
}


TYPED_TEST(SchwarzFactory, ApplyUsesInitialGuessAsLocalSolver)
{
    using value_type = typename TestFixture::value_type;
    using Cg = typename gko::solver::Cg<value_type>;
    using Jacobi = typename TestFixture::Jacobi;
    using Schwarz = typename TestFixture::Schwarz;

    auto schwarz_with_jacobi = Schwarz::build()
                                   .with_local_solver(Jacobi::build())
                                   .on(this->exec)
                                   ->generate(this->mtx);
    auto schwarz_with_cg =
        Schwarz::build()
            .with_local_solver(Cg::build().with_criteria(
                gko::stop::Iteration::build().with_max_iters(1u)))
            .on(this->exec)
            ->generate(this->mtx);

    ASSERT_EQ(schwarz_with_jacobi->apply_uses_initial_guess(), false);
    ASSERT_EQ(schwarz_with_cg->apply_uses_initial_guess(), true);
}
