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

#include <ginkgo/core/solver/lower_trs.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <core/test/utils.hpp>


namespace {


template <typename ValueIndexType>
class LowerTrs : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Dense<value_type>;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using Solver = gko::solver::LowerTrs<value_type, index_type>;

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
    std::unique_ptr<typename Solver::Factory> lower_trs_factory;
    std::unique_ptr<Solver> solver;
};


TYPED_TEST_CASE(LowerTrs, gko::test::ValueIndexTypes);


TYPED_TEST(LowerTrs, LowerTrsFactoryCreatesCorrectSolver)
{
    auto sys_mtx = this->solver->get_system_matrix();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_NE(sys_mtx, nullptr);
    GKO_ASSERT_MTX_NEAR(sys_mtx, this->csr_mtx, 0);
}


TYPED_TEST(LowerTrs, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        Solver::build().on(this->exec)->generate(Mtx::create(this->exec));

    copy->copy_from(gko::lend(this->solver));
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx.get(), this->csr_mtx.get(), 0);
}


TYPED_TEST(LowerTrs, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        Solver::build().on(this->exec)->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx.get(), this->csr_mtx.get(), 0);
}


TYPED_TEST(LowerTrs, CanBeCloned)
{
    auto clone = this->solver->clone();

    auto clone_mtx = clone->get_system_matrix();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(clone_mtx.get(), this->csr_mtx.get(), 0);
}


TYPED_TEST(LowerTrs, CanBeCleared)
{
    this->solver->clear();

    auto solver_mtx = this->solver->get_system_matrix();

    ASSERT_EQ(solver_mtx, nullptr);
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
}


}  // namespace
