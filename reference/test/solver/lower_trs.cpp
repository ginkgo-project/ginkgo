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


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/assertions.hpp"


namespace {


class LowerTrs : public ::testing::Test {
protected:
    using CsrMtx = gko::matrix::Csr<double, int>;
    using Mtx = gko::matrix::Dense<>;
    using Solver = gko::solver::LowerTrs<>;

    LowerTrs()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::initialize<Mtx>(
              {{2, 0.0, 0.0}, {3.0, 1, 0.0}, {1.0, 2.0, 3}}, exec)),
          b(gko::initialize<Mtx>({{2, 0.0, 0.0}}, exec)),
          csr_mtx(gko::copy_and_convert_to<CsrMtx>(exec, mtx.get())),
          lower_trs_factory(Solver::build().on(exec)),
          lower_trs_solver(lower_trs_factory->generate(mtx, b))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::shared_ptr<Mtx> b;
    std::shared_ptr<CsrMtx> csr_mtx;
    std::unique_ptr<Solver::Factory> lower_trs_factory;
    std::unique_ptr<Solver> lower_trs_solver;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }
};


TEST_F(LowerTrs, LowerTrsFactoryCreatesCorrectSolver)
{
    ASSERT_EQ(lower_trs_solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_NE(lower_trs_solver->get_system_matrix(), nullptr);
    ASSERT_NE(lower_trs_solver->get_rhs(), nullptr);
    ASSERT_EQ(lower_trs_solver->get_system_matrix(), mtx);
    ASSERT_EQ(lower_trs_solver->get_rhs(), b);
}


TEST_F(LowerTrs, CanBeCopied)
{
    auto copy = Solver::build().on(exec)->generate(Mtx::create(exec),
                                                   Mtx::create(exec));

    copy->copy_from(lend(lower_trs_solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = copy.get()->get_system_matrix();
    auto copy_b = copy.get()->get_rhs();
    assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()), mtx.get());
    assert_same_matrices(static_cast<const Mtx *>(copy_b.get()), b.get());
}


TEST_F(LowerTrs, CanBeMoved)
{
    auto copy =
        lower_trs_factory->generate(Mtx::create(exec), Mtx::create(exec));

    copy->copy_from(std::move(lower_trs_solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = copy.get()->get_system_matrix();
    auto copy_b = copy.get()->get_rhs();
    assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()), mtx.get());
    assert_same_matrices(static_cast<const Mtx *>(copy_b.get()), b.get());
}


TEST_F(LowerTrs, CanBeCloned)
{
    auto clone = lower_trs_solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = clone.get()->get_system_matrix();
    auto clone_b = clone.get()->get_rhs();
    assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()), mtx.get());
    assert_same_matrices(static_cast<const Mtx *>(clone_b.get()), b.get());
}


TEST_F(LowerTrs, CanBeCleared)
{
    lower_trs_solver->clear();

    ASSERT_EQ(lower_trs_solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx = lower_trs_solver.get()->get_system_matrix();
    auto solver_b = lower_trs_solver.get()->get_rhs();
    ASSERT_EQ(solver_mtx, nullptr);
    ASSERT_EQ(solver_b, nullptr);
}


}  // namespace
