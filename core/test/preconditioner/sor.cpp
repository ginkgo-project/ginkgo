// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/preconditioner/isai.hpp>
#include <ginkgo/core/preconditioner/sor.hpp>

#include "core/test/utils.hpp"


class SorFactory : public ::testing::Test {
public:
    using sor_type = gko::preconditioner::Sor<double, int>;
    using l_isai_type = gko::preconditioner::LowerIsai<double, int>;
    using u_isai_type = gko::preconditioner::UpperIsai<double, int>;

    std::shared_ptr<gko::ReferenceExecutor> exec =
        gko::ReferenceExecutor::create();
};


TEST_F(SorFactory, CanDefaultBuild)
{
    auto factory = sor_type::build().on(exec);

    auto params = factory->get_parameters();
    ASSERT_EQ(params.skip_sorting, false);
    ASSERT_EQ(params.relaxation_factor, 1.2);
    ASSERT_EQ(params.symmetric, false);
    ASSERT_EQ(params.l_solver, nullptr);
    ASSERT_EQ(params.u_solver, nullptr);
}


TEST_F(SorFactory, CanBuildWithParameters)
{
    auto factory = sor_type::build()
                       .with_skip_sorting(true)
                       .with_relaxation_factor(0.5)
                       .with_symmetric(true)
                       .with_l_solver(l_isai_type::build())
                       .with_u_solver(u_isai_type::build())
                       .on(exec);

    auto params = factory->get_parameters();
    ASSERT_EQ(params.skip_sorting, true);
    ASSERT_EQ(params.relaxation_factor, 0.5);
    ASSERT_EQ(params.symmetric, true);
    ASSERT_NE(params.l_solver, nullptr);
    GKO_ASSERT_DYNAMIC_TYPE(params.l_solver, l_isai_type::Factory);
    ASSERT_NE(params.u_solver, nullptr);
    GKO_ASSERT_DYNAMIC_TYPE(params.u_solver, u_isai_type::Factory);
}
