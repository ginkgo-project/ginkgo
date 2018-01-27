/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/solver/fcg.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>
#include <core/test/utils.hpp>


namespace {


class Fcg : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::Fcg<>;

    Fcg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec,
                          {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}})),
          fcg_factory(gko::solver::FcgFactory<>::create(exec, 3, 1e-6)),
          solver(fcg_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<gko::solver::FcgFactory<>> fcg_factory;
    std::unique_ptr<gko::LinOp> solver;
};


TEST_F(Fcg, FcgFactoryKnowsItsExecutor)
{
    ASSERT_EQ(fcg_factory->get_executor(), exec);
}


TEST_F(Fcg, FcgFactoryKnowsItsIterationLimit)
{
    ASSERT_EQ(fcg_factory->get_max_iters(), 3);
}


TEST_F(Fcg, FcgFactoryKnowsItsRelResidualGoal)
{
    ASSERT_EQ(fcg_factory->get_rel_residual_goal(), 1e-6);
}


TEST_F(Fcg, FcgFactoryCreatesCorrectSolver)
{
    ASSERT_EQ(solver->get_num_rows(), 3);
    ASSERT_EQ(solver->get_num_cols(), 3);
    ASSERT_EQ(solver->get_num_stored_elements(), 9);
    auto fcg_solver = dynamic_cast<Solver *>(solver.get());
    ASSERT_EQ(fcg_solver->get_max_iters(), 3);
    ASSERT_EQ(fcg_solver->get_rel_residual_goal(), 1e-6);
    ASSERT_NE(fcg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(fcg_solver->get_system_matrix(), mtx);
}


TEST_F(Fcg, CanBeCopied)
{
    auto copy = fcg_factory->generate(Mtx::create(exec));

    copy->copy_from(solver.get());

    ASSERT_EQ(copy->get_num_rows(), 3);
    ASSERT_EQ(copy->get_num_cols(), 3);
    ASSERT_EQ(copy->get_num_stored_elements(), 9);
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    ASSERT_NE(copy_mtx.get(), mtx.get());
    ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(copy_mtx.get()), mtx.get(),
                    1e-14);
}


TEST_F(Fcg, CanBeMoved)
{
    auto copy = fcg_factory->generate(Mtx::create(exec));

    copy->copy_from(std::move(solver));

    ASSERT_EQ(copy->get_num_rows(), 3);
    ASSERT_EQ(copy->get_num_cols(), 3);
    ASSERT_EQ(copy->get_num_stored_elements(), 9);
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(copy_mtx.get()), mtx.get(),
                    1e-14);
}


TEST_F(Fcg, CanBeCloned)
{
    auto clone = solver->clone();

    ASSERT_EQ(clone->get_num_rows(), 3);
    ASSERT_EQ(clone->get_num_cols(), 3);
    ASSERT_EQ(clone->get_num_stored_elements(), 9);
    auto clone_mtx = dynamic_cast<Solver *>(clone.get())->get_system_matrix();
    ASSERT_NE(clone_mtx.get(), mtx.get());
    ASSERT_MTX_NEAR(dynamic_cast<const Mtx *>(clone_mtx.get()), mtx.get(),
                    1e-14);
}


TEST_F(Fcg, CanBeCleared)
{
    solver->clear();

    ASSERT_EQ(solver->get_num_rows(), 0);
    ASSERT_EQ(solver->get_num_cols(), 0);
    ASSERT_EQ(solver->get_num_stored_elements(), 0);
    auto solver_mtx = dynamic_cast<Solver *>(solver.get())->get_system_matrix();
    ASSERT_NE(solver_mtx, nullptr);
    ASSERT_EQ(solver_mtx->get_num_rows(), 0);
    ASSERT_EQ(solver_mtx->get_num_cols(), 0);
    ASSERT_EQ(solver_mtx->get_num_stored_elements(), 0);
}


}  // namespace
