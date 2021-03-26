/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <ginkgo/core/matrix/batch_dense.hpp>


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
          mtx(gko::test::create_poisson1d_batch<value_type>(
              std::static_pointer_cast<const gko::ReferenceExecutor>(
                  this->exec),
              nrows, nbatch)),
          batchrich_factory(Solver::build()
                                .with_max_iterations(def_max_iters)
                                .with_rel_residual_tol(def_rel_res_tol)
                                .with_preconditioner("jacobi")
                                .on(exec)),
          solver(batchrich_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    const gko::size_type nbatch = 3;
    const int nrows = 5;
    const gko::size_type cumul_sz = nbatch * nrows;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> batchrich_factory;
    std::unique_ptr<gko::LinOp> solver;
    const int def_max_iters = 10;
    const real_type def_rel_res_tol = 1e-4;
};

TYPED_TEST_SUITE(BatchRich, gko::test::ValueTypes);


TYPED_TEST(BatchRich, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->batchrich_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchRich, FactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;
    ASSERT_EQ(this->solver->get_size(),
              gko::dim<2>(this->cumul_sz, this->cumul_sz));
    auto batchrich_solver = static_cast<Solver *>(this->solver.get());
    ASSERT_NE(batchrich_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(batchrich_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchRich, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchrich_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(this->cumul_sz, this->cumul_sz));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx *>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchRich, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batchrich_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(this->cumul_sz, this->cumul_sz));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    const auto copy_batch_mtx = static_cast<const Mtx *>(copy_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), copy_batch_mtx, 0.0);
}


TYPED_TEST(BatchRich, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(this->cumul_sz, this->cumul_sz));
    auto clone_mtx = static_cast<Solver *>(clone.get())->get_system_matrix();
    const auto clone_batch_mtx = static_cast<const Mtx *>(clone_mtx.get());
    GKO_ASSERT_BATCH_MTX_NEAR(this->mtx.get(), clone_batch_mtx, 0.0);
}


TYPED_TEST(BatchRich, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchRich, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(BatchRich, CanSetCriteria)
{
    using Solver = typename TestFixture::Solver;
    using RT = typename TestFixture::real_type;

    auto batchrich_factory = Solver::build()
                                 .with_max_iterations(22)
                                 .with_rel_residual_tol(static_cast<RT>(0.25))
                                 .on(this->exec);
    auto solver = batchrich_factory->generate(this->mtx);

    ASSERT_EQ(batchrich_factory->get_parameters().max_iterations, 22);
    const RT tol = std::numeric_limits<RT>::epsilon();
    ASSERT_NEAR(batchrich_factory->get_parameters().rel_residual_tol, 0.25,
                tol);
    ASSERT_NE(solver.get(), nullptr);
}


TYPED_TEST(BatchRich, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    const std::string batchrich_precond = "none";

    auto batchrich_factory =
        Solver::build().with_max_iterations(3).with_preconditioner("none").on(
            this->exec);
    auto solver = batchrich_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond, "");
    ASSERT_EQ(precond, batchrich_precond);
}


TYPED_TEST(BatchRich, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::string unavailable_prec = "smoothed_aggregation";
    auto batchrich_factory =
        Solver::build().with_preconditioner(unavailable_prec).on(this->exec);

    ASSERT_THROW(batchrich_factory->generate(this->mtx), gko::NotImplemented);
}


TYPED_TEST(BatchRich, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, 2, gko::dim<2>{3, 5}, 3);

    ASSERT_THROW(this->batchrich_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


}  // namespace
