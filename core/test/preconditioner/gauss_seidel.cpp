// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/preconditioner/gauss_seidel.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>

#include "core/test/utils.hpp"


class GaussSeidelFactory : public ::testing::Test {
public:
    using GaussSeidel_type = gko::preconditioner::GaussSeidel<double, int>;
    using l_isai_type = gko::preconditioner::LowerIsai<double, int>;
    using u_isai_type = gko::preconditioner::UpperIsai<double, int>;

    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
};


TEST_F(GaussSeidelFactory, CanDefaultBuild)
{
    auto factory = GaussSeidel_type::build().on(exec);

    auto params = factory->get_parameters();
    ASSERT_EQ(params.skip_sorting, false);
    ASSERT_EQ(params.symmetric, false);
    ASSERT_EQ(params.l_solver, nullptr);
    ASSERT_EQ(params.u_solver, nullptr);
}


TEST_F(GaussSeidelFactory, CanBuildWithParameters)
{
    auto factory = GaussSeidel_type::build()
                       .with_skip_sorting(true)
                       .with_symmetric(true)
                       .with_l_solver(l_isai_type::build())
                       .with_u_solver(u_isai_type::build())
                       .on(exec);

    auto params = factory->get_parameters();
    ASSERT_EQ(params.skip_sorting, true);
    ASSERT_EQ(params.symmetric, true);
    ASSERT_NE(params.l_solver, nullptr);
    GKO_ASSERT_DYNAMIC_TYPE(params.l_solver, l_isai_type::Factory);
    ASSERT_NE(params.u_solver, nullptr);
    GKO_ASSERT_DYNAMIC_TYPE(params.u_solver, u_isai_type::Factory);
}
