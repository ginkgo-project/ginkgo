// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/ilu.hpp>


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>


#include "core/test/utils.hpp"


namespace {


class IluFactory : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using l_solver_type = gko::solver::Bicgstab<value_type>;
    using u_solver_type = gko::solver::Bicgstab<value_type>;
    using ilu_prec_type =
        gko::preconditioner::Ilu<l_solver_type, u_solver_type, false>;
    using ilu_type = gko::factorization::ParIlu<value_type, index_type>;

    IluFactory()
        : exec(gko::ReferenceExecutor::create()),
          l_factory(l_solver_type::build().on(exec)),
          u_factory(u_solver_type::build().on(exec)),
          fact_factory(ilu_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<typename l_solver_type::Factory> l_factory;
    std::shared_ptr<typename u_solver_type::Factory> u_factory;
    std::shared_ptr<typename ilu_type::Factory> fact_factory;
};


TEST_F(IluFactory, KnowsItsExecutor)
{
    auto ilu_factory = ilu_prec_type::build().on(this->exec);

    ASSERT_EQ(ilu_factory->get_executor(), this->exec);
}


TEST_F(IluFactory, CanSetLSolverFactory)
{
    auto ilu_factory =
        ilu_prec_type::build().with_l_solver(this->l_factory).on(this->exec);

    ASSERT_EQ(ilu_factory->get_parameters().l_solver_factory, this->l_factory);
}


TEST_F(IluFactory, CanSetUSolverFactory)
{
    auto ilu_factory =
        ilu_prec_type::build().with_u_solver(this->u_factory).on(this->exec);

    ASSERT_EQ(ilu_factory->get_parameters().u_solver_factory, this->u_factory);
}


TEST_F(IluFactory, CanSetFactorizationFactory)
{
    auto ilu_factory = ilu_prec_type::build()
                           .with_factorization(this->fact_factory)
                           .on(this->exec);

    ASSERT_EQ(ilu_factory->get_parameters().factorization_factory,
              this->fact_factory);
}


TEST_F(IluFactory, DeprecatedFactoryParameter)
{
    auto ilu_factory = ilu_prec_type::build()
                           .with_l_solver_factory(this->l_factory)
                           .with_u_solver_factory(this->u_factory)
                           .with_factorization_factory(this->fact_factory)
                           .on(this->exec);

    ASSERT_EQ(ilu_factory->get_parameters().l_solver_factory, this->l_factory);
    ASSERT_EQ(ilu_factory->get_parameters().u_solver_factory, this->u_factory);
    ASSERT_EQ(ilu_factory->get_parameters().factorization_factory,
              this->fact_factory);
}


TEST_F(IluFactory, DeferredFactoryParameter)
{
    auto ilu_factory = ilu_prec_type::build()
                           .with_l_solver(l_solver_type::build())
                           .with_u_solver(u_solver_type::build())
                           .with_factorization(ilu_type::build())
                           .on(this->exec);

    GKO_ASSERT_DYNAMIC_TYPE(ilu_factory->get_parameters().l_solver_factory,
                            l_solver_type::Factory);
    GKO_ASSERT_DYNAMIC_TYPE(ilu_factory->get_parameters().u_solver_factory,
                            u_solver_type::Factory);
    GKO_ASSERT_DYNAMIC_TYPE(ilu_factory->get_parameters().factorization_factory,
                            ilu_type::Factory);
}


}  // namespace


GKO_END_DISABLE_DEPRECATION_WARNINGS
