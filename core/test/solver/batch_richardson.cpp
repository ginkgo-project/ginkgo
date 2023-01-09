/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/batch_richardson.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"


namespace {


template <typename T>
class BatchRich : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<T>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using Dense = gko::matrix::BatchDense<value_type>;
    using Solver = gko::solver::BatchRichardson<value_type>;

    BatchRich()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::test::create_poisson1d_batch<Mtx>(
              std::static_pointer_cast<const gko::ReferenceExecutor>(exec),
              nrows, nbatch)),
          prec_factory(
              gko::share(gko::preconditioner::BatchJacobi<T>::build().on(exec)))
    {}

    std::shared_ptr<const gko::Executor> exec;
    const gko::size_type nbatch = 3;
    const int nrows = 5;
    std::shared_ptr<Mtx> mtx;
    const int def_max_iters = 10;
    const real_type def_rel_res_tol = 1e-4;
    const real_type def_relax = 1.2;
    const gko::stop::batch::ToleranceType def_tol_type =
        gko::stop::batch::ToleranceType::relative;
    std::shared_ptr<gko::BatchLinOpFactory> prec_factory;

    std::unique_ptr<Solver> generate_solver()
    {
        auto batchrich_factory =
            Solver::build()
                .with_default_max_iterations(this->def_max_iters)
                .with_default_residual_tol(this->def_rel_res_tol)
                .with_preconditioner(prec_factory)
                .with_relaxation_factor(this->def_relax)
                .on(this->exec);
        auto solver = batchrich_factory->generate(this->mtx);
        return std::unique_ptr<Solver>(static_cast<Solver*>(solver.release()));
    }

    void assert_size(const gko::BatchLinOp* const solver)
    {
        for (size_t i = 0; i < nbatch; i++) {
            ASSERT_EQ(solver->get_size().at(i), gko::dim<2>(nrows, nrows));
        }
    }

    // Checks equality of the matrix and parameters with defaults
    void assert_solver_params(const Solver* const a)
    {
        ASSERT_EQ(a->get_parameters().default_max_iterations, def_max_iters);
        ASSERT_EQ(a->get_parameters().preconditioner, prec_factory);
        ASSERT_EQ(a->get_parameters().relaxation_factor, def_relax);
        ASSERT_EQ(a->get_parameters().default_residual_tol, def_rel_res_tol);
        ASSERT_EQ(a->get_parameters().tolerance_type, def_tol_type);
    }

    // Checks equality of the matrix and parameters with defaults
    void assert_solver_with_mtx(const Solver* const a)
    {
        auto a_copy_mtx = a->get_system_matrix();
        const auto a_copy_batch_mtx = static_cast<const Mtx*>(a_copy_mtx.get());
        GKO_ASSERT_BATCH_MTX_NEAR(a_copy_batch_mtx, mtx, 0.0);
        assert_solver_params(a);
    }
};

TYPED_TEST_SUITE(BatchRich, gko::test::ValueTypes);


TYPED_TEST(BatchRich, FactoryKnowsItsExecutor)
{
    using Solver = typename TestFixture::Solver;
    auto batchrich_factory = Solver::build().on(this->exec);

    ASSERT_EQ(batchrich_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchRich, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    using real_type = typename TestFixture::real_type;

    auto batchrich_solver = this->generate_solver();

    ASSERT_EQ(batchrich_solver->get_system_matrix(), this->mtx);
    this->assert_solver_params(batchrich_solver.get());
}


TYPED_TEST(BatchRich, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto solver = this->generate_solver();
    auto batchrich_factory = Solver::build().on(this->exec);
    auto copy = batchrich_factory->generate(Mtx::create(this->exec));

    copy->copy_from(solver.get());

    this->assert_size(copy.get());
    this->assert_solver_with_mtx(static_cast<Solver*>(copy.get()));
}


TYPED_TEST(BatchRich, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto solver = this->generate_solver();
    auto batchrich_factory = Solver::build().on(this->exec);
    auto copy = batchrich_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(solver));

    this->assert_size(copy.get());
    this->assert_solver_with_mtx(static_cast<Solver*>(copy.get()));
    ASSERT_EQ(solver.get(), nullptr);
}


TYPED_TEST(BatchRich, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto solver = this->generate_solver();

    auto clone = solver->clone();

    this->assert_size(clone.get());
    this->assert_solver_with_mtx(static_cast<Solver*>(clone.get()));
}


TYPED_TEST(BatchRich, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    auto solver = this->generate_solver();

    solver->clear();

    ASSERT_EQ(solver->get_num_batch_entries(), 0);
    ASSERT_EQ(solver->get_size().at(0), gko::dim<2>(0, 0));
    auto solver_mtx = static_cast<Solver*>(solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchRich, ApplyUsesInitialGuessReturnsTrue)
{
    auto solver = this->generate_solver();

    ASSERT_TRUE(solver->apply_uses_initial_guess());
}


TYPED_TEST(BatchRich, CanSetCriteriaInFactory)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;

    auto batchrich_factory =
        Solver::build()
            .with_default_max_iterations(22)
            .with_default_residual_tol(static_cast<RT>(0.25))
            .with_relaxation_factor(static_cast<RT>(0.28))
            .on(this->exec);
    auto solver = batchrich_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_parameters().default_max_iterations, 22);
    ASSERT_EQ(solver->get_parameters().relaxation_factor,
              static_cast<RT>(0.28));
    const RT tol = std::numeric_limits<RT>::epsilon();
    ASSERT_NEAR(solver->get_parameters().default_residual_tol, 0.25, tol);
}


TYPED_TEST(BatchRich, CanSetResidualTol)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;
    auto factory =
        Solver::build()
            .with_default_max_iterations(22)
            .with_default_residual_tol(static_cast<RT>(0.25))
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(this->exec);
    auto solver = factory->generate(this->mtx);

    solver->set_residual_tolerance(0.5);

    ASSERT_EQ(solver->get_parameters().default_residual_tol, 0.25);
    ASSERT_EQ(solver->get_residual_tolerance(), 0.5);
}


TYPED_TEST(BatchRich, CanSetPreconditionerFactory)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    auto prec_factory = gko::share(
        gko::preconditioner::BatchJacobi<value_type>::build().on(this->exec));

    auto batchrich_factory =
        Solver::build().with_preconditioner(prec_factory).on(this->exec);
    auto solver = batchrich_factory->generate(this->mtx);
    auto precond = solver->get_parameters().preconditioner;

    ASSERT_EQ(precond, prec_factory);
}


TYPED_TEST(BatchRich, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto batchrich_factory = Solver::build().on(this->exec);
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, 2, gko::dim<2>{3, 5}, 3);

    ASSERT_THROW(batchrich_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchRich, CanSetScalingOps)
{
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    using Dense = typename TestFixture::Dense;
    using Diag = gko::matrix::BatchDiagonal<value_type>;
    auto left_scale = gko::share(Diag::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, this->nrows))));
    auto right_scale = gko::share(Diag::create(
        this->exec,
        gko::batch_dim<>(this->nbatch, gko::dim<2>(this->nrows, this->nrows))));
    auto batchrich_factory = Solver::build()
                                 .with_left_scaling_op(left_scale)
                                 .with_right_scaling_op(right_scale)
                                 .on(this->exec);
    auto solver = batchrich_factory->generate(this->mtx);

    ASSERT_EQ(solver->get_left_scaling_op(), left_scale);
    ASSERT_EQ(solver->get_right_scaling_op(), right_scale);
}

// TODO: Enable after BatchCsr::transpose is implemented.
// TYPED_TEST(BatchRich, SolverTransposeRetainsFactoryParameters)
// {
//     using Solver = typename TestFixture::Solver;

//     auto batchrich_factory =
//         Solver::build().with_default_max_iterations(3).with_default_residual_tol(0.25f)
// 		.with_relaxation_factor(2.0f).with_preconditioner(gpb::none).on(this->exec);
//     auto solver = batchrich_factory->generate(this->mtx);
// 	auto solver_trans = gko::as<Solver>(solver->transpose());
// 	auto params = solver_trans->get_parameters();

// 	ASSERT_EQ(params.preconditioner, gpb::none);
// 	ASSERT_EQ(params.default_max_iterations, 3);
// 	ASSERT_EQ(params.default_residual_tol, 0.25);
// 	ASSERT_EQ(params.relaxation_factor, 2.0);
// }


}  // namespace
