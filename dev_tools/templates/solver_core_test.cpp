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

#include <core/solver/xxsolverxx.hpp>


#include <gtest/gtest.h>


#include <core/base/executor.hpp>
#include <core/matrix/dense.hpp>


namespace {


class Xxsolverxx : public ::testing::Test {
protected:
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::Xxsolverxx<>;

    Xxsolverxx()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec,
                          {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}})),
          xxsolverxx_factory(
              gko::solver::XxsolverxxFactory<>::create(exec, 3, 1e-6)),
          solver(xxsolverxx_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<gko::solver::XxsolverxxFactory<>> xxsolverxx_factory;
    std::unique_ptr<gko::LinOp> solver;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_num_rows(), m2->get_num_rows());
        ASSERT_EQ(m1->get_num_cols(), m2->get_num_cols());
        for (gko::size_type i = 0; i < m1->get_num_rows(); ++i) {
            for (gko::size_type j = 0; j < m2->get_num_cols(); ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }
};


TEST_F(Xxsolverxx, XxsolverxxFactoryKnowsItsExecutor)
{
    ASSERT_EQ(xxsolverxx_factory->get_executor(), exec);
}


TEST_F(Xxsolverxx, XxsolverxxFactoryKnowsItsIterationLimit)
{
    ASSERT_EQ(xxsolverxx_factory->get_max_iters(), 3);
}


TEST_F(Xxsolverxx, XxsolverxxFactoryKnowsItsRelResidualGoal)
{
    ASSERT_EQ(xxsolverxx_factory->get_rel_residual_goal(), 1e-6);
}


TEST_F(Xxsolverxx, XxsolverxxFactoryCreatesCorrectSolver)
{
    ASSERT_EQ(solver->get_num_rows(), 3);
    ASSERT_EQ(solver->get_num_cols(), 3);
    ASSERT_EQ(solver->get_num_nonzeros(), 9);
    auto xxsolverxx_solver = dynamic_cast<Solver *>(solver.get());
    ASSERT_EQ(xxsolverxx_solver->get_max_iters(), 3);
    ASSERT_EQ(xxsolverxx_solver->get_rel_residual_goal(), 1e-6);
    ASSERT_NE(xxsolverxx_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(xxsolverxx_solver->get_system_matrix(), mtx);
}


TEST_F(Xxsolverxx, CanBeCopied)
{
    auto copy = xxsolverxx_factory->generate(Mtx::create(exec));

    copy->copy_from(solver.get());

    ASSERT_EQ(copy->get_num_rows(), 3);
    ASSERT_EQ(copy->get_num_cols(), 3);
    ASSERT_EQ(copy->get_num_nonzeros(), 9);
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    ASSERT_NE(copy_mtx.get(), mtx.get());
    assert_same_matrices(dynamic_cast<const Mtx *>(copy_mtx.get()), mtx.get());
}


TEST_F(Xxsolverxx, CanBeMoved)
{
    auto copy = xxsolverxx_factory->generate(Mtx::create(exec));

    copy->copy_from(std::move(solver));

    ASSERT_EQ(copy->get_num_rows(), 3);
    ASSERT_EQ(copy->get_num_cols(), 3);
    ASSERT_EQ(copy->get_num_nonzeros(), 9);
    auto copy_mtx = dynamic_cast<Solver *>(copy.get())->get_system_matrix();
    assert_same_matrices(dynamic_cast<const Mtx *>(copy_mtx.get()), mtx.get());
}


TEST_F(Xxsolverxx, CanBeCloned)
{
    auto clone = solver->clone();

    ASSERT_EQ(clone->get_num_rows(), 3);
    ASSERT_EQ(clone->get_num_cols(), 3);
    ASSERT_EQ(clone->get_num_nonzeros(), 9);
    auto clone_mtx = dynamic_cast<Solver *>(clone.get())->get_system_matrix();
    ASSERT_NE(clone_mtx.get(), mtx.get());
    assert_same_matrices(dynamic_cast<const Mtx *>(clone_mtx.get()), mtx.get());
}


TEST_F(Xxsolverxx, CanBeCleared)
{
    solver->clear();

    ASSERT_EQ(solver->get_num_rows(), 0);
    ASSERT_EQ(solver->get_num_cols(), 0);
    ASSERT_EQ(solver->get_num_nonzeros(), 0);
    auto solver_mtx = dynamic_cast<Solver *>(solver.get())->get_system_matrix();
    ASSERT_NE(solver_mtx, nullptr);
    ASSERT_EQ(solver_mtx->get_num_rows(), 0);
    ASSERT_EQ(solver_mtx->get_num_cols(), 0);
    ASSERT_EQ(solver_mtx->get_num_nonzeros(), 0);
}


}  // namespace
