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

#include <ginkgo/core/solver/upper_trs.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class UpperTrs : public ::testing::Test {
protected:
    using CsrMtx = gko::matrix::Csr<double, int>;
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::UpperTrs<>;

    UpperTrs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{1, 3.0, 1.0}, {0.0, 1, 2.0}, {0.0, 0.0, 1}}, exec)),
          csr_mtx(gko::initialize<CsrMtx>(
              {{1, 3.0, 1.0}, {0.0, 1, 2.0}, {0.0, 0.0, 1}}, exec)),
          upper_trs_factory(Solver::build().on(exec)),
          upper_trs_solver(upper_trs_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<Solver::Factory> upper_trs_factory;
    std::unique_ptr<Solver> upper_trs_solver;
};


TEST_F(UpperTrs, UpperTrsFactoryCreatesCorrectSolver)
{
    auto sys_mtx = upper_trs_solver->get_system_matrix();

    ASSERT_EQ(upper_trs_solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_NE(sys_mtx, nullptr);
    GKO_ASSERT_MTX_NEAR(sys_mtx, csr_mtx, 0);
}


TEST_F(UpperTrs, CanBeCopied)
{
    auto copy = Solver::build().on(exec)->generate(Mtx::create(exec));

    copy->copy_from(gko::lend(upper_trs_solver));
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(UpperTrs, CanBeMoved)
{
    auto copy = upper_trs_factory->generate(Mtx::create(exec));

    copy->copy_from(std::move(upper_trs_solver));
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(UpperTrs, CanBeCloned)
{
    auto clone = upper_trs_solver->clone();

    auto clone_mtx = clone->get_system_matrix();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(clone_mtx.get(), csr_mtx.get(), 0);
}


TEST_F(UpperTrs, CanBeCleared)
{
    upper_trs_solver->clear();

    auto solver_mtx = upper_trs_solver->get_system_matrix();

    ASSERT_EQ(upper_trs_solver->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(solver_mtx, nullptr);
}


}  // namespace
