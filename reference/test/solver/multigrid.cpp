/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/solver/multigrid.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/multigrid/amgx_pgm.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm_reduction.hpp>
#include <ginkgo/core/stop/time.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class Multigrid : public ::testing::Test {
protected:
    using value_type = T;
    using Csr = gko::matrix::Csr<value_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Multigrid<value_type>;
    using Coarse = gko::multigrid::AmgxPgm<value_type>;
    using Smoother = gko::preconditioner::Jacobi<value_type>;
    using CoarsestSolver = gko::solver::Cg<value_type>;
    Multigrid()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Csr>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          coarse_factory(Coarse::build()
                             .with_max_iterations(2u)
                             .with_max_unassigned_percentage(0.1)
                             .on(exec)),
          smoother_factory(
              gko::give(Smoother::build().with_max_block_size(3u).on(exec))),
          coarsest_factory(
              CoarsestSolver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(4u).on(exec),
                      gko::stop::Time::build()
                          .with_time_limit(std::chrono::seconds(6))
                          .on(exec),
                      gko::stop::ResidualNormReduction<value_type>::build()
                          .with_reduction_factor(r<value_type>::value)
                          .on(exec))
                  .on(exec))
    {
        multigrid_factory =
            Solver::build()
                .with_pre_smoother(smoother_factory)
                .with_post_smoother(smoother_factory)
                .with_coarsest_solver(coarsest_factory)
                .with_max_levels(2u)
                .with_rstr_prlg(coarse_factory)
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(4u).on(exec),
                    gko::stop::Time::build()
                        .with_time_limit(std::chrono::seconds(100))
                        .on(exec),
                    gko::stop::ResidualNormReduction<value_type>::build()
                        .with_reduction_factor(r<value_type>::value)
                        .on(exec))
                .on(exec);
    }

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Csr> mtx;
    std::unique_ptr<typename Solver::Factory> multigrid_factory;
    std::shared_ptr<typename Coarse::Factory> coarse_factory;
    std::shared_ptr<typename Smoother::Factory> smoother_factory;
    std::shared_ptr<typename CoarsestSolver::Factory> coarsest_factory;
};

TYPED_TEST_CASE(Multigrid, gko::test::ValueTypes);


TYPED_TEST(Multigrid, SolvesStencilSystem)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    auto solver = this->multigrid_factory->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->exec);
    solver->apply(b.get(), x.get());
    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


}  // namespace
