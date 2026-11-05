// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/bicg.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/solver/cg.hpp>
#include <ginkgo/core/solver/cgs.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>
#include <ginkgo/core/solver/direct.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/solver/gcr.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/idr.hpp>
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/solver/minres.hpp>
#include <ginkgo/core/solver/multigrid.hpp>
#include <ginkgo/core/solver/pipe_cg.hpp>
#include <ginkgo/core/solver/triangular.hpp>
#include <ginkgo/core/stop/iteration.hpp>

#include "core/test/utils.hpp"


template <typename T>
class SolverBase : public ::testing::Test {};

using SolverTypes = ::testing::Types<
    gko::solver::Bicg<double>, gko::solver::Bicgstab<double>,
    gko::solver::CbGmres<double>, gko::solver::Cg<double>,
    gko::solver::Cgs<double>, gko::solver::Chebyshev<double>,
    gko::experimental::solver::Direct<double, int>, gko::solver::Fcg<double>,
    gko::solver::Gcr<double>, gko::solver::Gmres<double>,
    gko::solver::Idr<double>, gko::solver::Ir<double>,
    gko::solver::LowerTrs<double, int>, gko::solver::Minres<double>,
    gko::solver::Multigrid, gko::solver::PipeCg<double>,
    gko::solver::UpperTrs<double, int>>;

TYPED_TEST_SUITE(SolverBase, SolverTypes, TypenameNameGenerator);


TYPED_TEST(SolverBase, CannotBeCopied)
{
    ASSERT_FALSE(std::is_copy_constructible<TypeParam>::value);
    ASSERT_FALSE(std::is_copy_assignable<TypeParam>::value);
}


TYPED_TEST(SolverBase, CannotBeMoved)
{
    ASSERT_FALSE(std::is_move_constructible<TypeParam>::value);
    ASSERT_FALSE(std::is_move_assignable<TypeParam>::value);
}


TEST(EnablePreconditionable, CannotBeCopiedOrMoved)
{
    using Mixin = gko::solver::EnablePreconditionable<gko::solver::Cg<double>>;

    ASSERT_FALSE(std::is_copy_constructible<Mixin>::value);
    ASSERT_FALSE(std::is_copy_assignable<Mixin>::value);
    ASSERT_FALSE(std::is_move_constructible<Mixin>::value);
    ASSERT_FALSE(std::is_move_assignable<Mixin>::value);
}


TEST(EnableIterativeBase, CannotBeCopiedOrMoved)
{
    using Mixin = gko::solver::EnableIterativeBase<gko::solver::Ir<double>>;

    ASSERT_FALSE(std::is_copy_constructible<Mixin>::value);
    ASSERT_FALSE(std::is_copy_assignable<Mixin>::value);
    ASSERT_FALSE(std::is_move_constructible<Mixin>::value);
    ASSERT_FALSE(std::is_move_assignable<Mixin>::value);
}


TEST(EnableIterativeBase, ThrowsClearlyForAStopFactoryOnAnotherExecutor)
{
    auto exec = gko::ReferenceExecutor::create();
    auto solver =
        gko::solver::Cg<double>::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec)
            ->generate(
                gko::matrix::Dense<double>::create(exec, gko::dim<2>{1}));
    auto stop_factory = solver->get_stop_criterion_factory();
    auto other_factory =
        gko::share(gko::stop::Iteration::build().with_max_iters(2u).on(
            gko::ReferenceExecutor::create()));

    try {
        solver->set_stop_criterion_factory(other_factory);
        FAIL() << "expected NotSupported";
    } catch (const gko::NotSupported& error) {
        ASSERT_NE(std::string{error.what()}.find(
                      "set_stop_criterion_factory with a factory on another "
                      "executor"),
                  std::string::npos);
    }
    ASSERT_EQ(solver->get_stop_criterion_factory(), stop_factory);
}
