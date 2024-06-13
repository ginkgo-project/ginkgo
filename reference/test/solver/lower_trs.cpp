// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/test/utils.hpp"


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

TYPED_TEST_SUITE(LowerTrs, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


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

    copy->copy_from(this->solver);
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx, this->csr_mtx, 0);
}


TYPED_TEST(LowerTrs, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        Solver::build().on(this->exec)->generate(Mtx::create(this->exec));

    copy->move_from(this->solver);
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx, this->csr_mtx, 0);
}


TYPED_TEST(LowerTrs, CanBeCloned)
{
    auto clone = this->solver->clone();

    auto clone_mtx = clone->get_system_matrix();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(clone_mtx, this->csr_mtx, 0);
}


TYPED_TEST(LowerTrs, CanBeCleared)
{
    this->solver->clear();

    auto solver_mtx = this->solver->get_system_matrix();

    ASSERT_EQ(solver_mtx, nullptr);
    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
}


}  // namespace
