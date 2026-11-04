// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <type_traits>

#include <gtest/gtest.h>

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
