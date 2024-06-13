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
class UpperTrs : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using CsrMtx = gko::matrix::Csr<value_type, index_type>;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::UpperTrs<value_type, index_type>;

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
    std::unique_ptr<typename Solver::Factory> upper_trs_factory;
    std::unique_ptr<Solver> upper_trs_solver;
};

TYPED_TEST_SUITE(UpperTrs, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(UpperTrs, UpperTrsFactoryCreatesCorrectSolver)
{
    auto sys_mtx = this->upper_trs_solver->get_system_matrix();

    ASSERT_EQ(this->upper_trs_solver->get_size(), gko::dim<2>(3, 3));
    ASSERT_NE(sys_mtx, nullptr);
    GKO_ASSERT_MTX_NEAR(sys_mtx, this->csr_mtx, 0);
}


TYPED_TEST(UpperTrs, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy =
        Solver::build().on(this->exec)->generate(Mtx::create(this->exec));

    copy->copy_from(this->upper_trs_solver);
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx, this->csr_mtx, 0);
}


TYPED_TEST(UpperTrs, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    auto copy = this->upper_trs_factory->generate(Mtx::create(this->exec));

    copy->move_from(this->upper_trs_solver);
    auto copy_mtx = copy->get_system_matrix();

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(copy_mtx, this->csr_mtx, 0);
}


TYPED_TEST(UpperTrs, CanBeCloned)
{
    auto clone = this->upper_trs_solver->clone();

    auto clone_mtx = clone->get_system_matrix();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    GKO_ASSERT_MTX_NEAR(clone_mtx, this->csr_mtx, 0);
}


TYPED_TEST(UpperTrs, CanBeCleared)
{
    this->upper_trs_solver->clear();

    auto solver_mtx = this->upper_trs_solver->get_system_matrix();

    ASSERT_EQ(this->upper_trs_solver->get_size(), gko::dim<2>(0, 0));
    ASSERT_EQ(solver_mtx, nullptr);
}


}  // namespace
