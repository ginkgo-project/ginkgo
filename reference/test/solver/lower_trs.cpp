/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/solver/lower_trs.hpp>


#include <memory>
#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class LowerTrs : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using CsrMtx = gko::matrix::Csr<>;
    using Solver = gko::solver::LowerTrs<>;

    LowerTrs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          csr_mtx(gko::initialize<CsrMtx>(
              {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, exec)),
          lower_trs_factory(Solver::build().on(exec)),
          solver(lower_trs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<Solver::Factory> lower_trs_factory;
    std::unique_ptr<Solver> solver;
};


TEST_F(LowerTrs, LowerTrsFactoryCreatesCorrectSolver)
{
    auto sys_mtx = solver->get_system_matrix();

    ASSERT_EQ(solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_NE(sys_mtx, nullptr);
    GKO_ASSERT_MTX_NEAR(sys_mtx, csr_mtx, 0);
}


TEST_F(LowerTrs, CanBeCopied)
{
    auto copy = Solver::build().on(exec)->generate(Mtx::create(exec));

    copy->copy_from(gko::lend(solver));
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(LowerTrs, CanBeMoved)
{
    auto copy = Solver::build().on(exec)->generate(Mtx::create(exec));

    copy->copy_from(std::move(solver));
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(LowerTrs, CanBeCloned)
{
    auto clone = solver->clone();

    auto clone_mtx = clone->get_system_matrix();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(clone_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(LowerTrs, CanBeCleared)
{
    solver->clear();

    auto solver_mtx = solver->get_system_matrix();

    ASSERT_EQ(solver_mtx, nullptr);
    ASSERT_EQ(solver->get_size(), gko::dim<2>(0, 0));
}


TEST_F(LowerTrs, CanSetPreconditionerGenerator)
{
    auto lower_trs_factory =
        Solver::build().with_preconditioner(Solver::build().on(exec)).on(exec);
    auto solver = lower_trs_factory->generate(mtx);

    auto precond = dynamic_cast<const gko::solver::LowerTrs<> *>(
        static_cast<gko::solver::LowerTrs<> *>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(
        static_cast<const CsrMtx *>(precond->get_system_matrix().get()),
        csr_mtx.get(), 0);
}


}  // namespace
