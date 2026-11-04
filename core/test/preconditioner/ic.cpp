// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/preconditioner/ic.hpp>


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


#include <memory>
#include <type_traits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>

#include "core/test/utils.hpp"


namespace {


class IcFactory : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using solver_type = gko::solver::Bicgstab<value_type>;
    using ic_prec_type = gko::preconditioner::Ic<value_type>;
    using ic_type = gko::factorization::ParIc<value_type, index_type>;

    IcFactory()
        : exec(gko::ReferenceExecutor::create()),
          l_factory(solver_type::build().on(exec)),
          fact_factory(ic_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<typename solver_type::Factory> l_factory;
    std::shared_ptr<typename ic_type::Factory> fact_factory;
};


TEST_F(IcFactory, KnowsItsExecutor)
{
    auto ic_factory = ic_prec_type::build().on(this->exec);

    ASSERT_EQ(ic_factory->get_executor(), this->exec);
}


TEST_F(IcFactory, CanSetLSolverFactory)
{
    auto ic_factory =
        ic_prec_type::build().with_l_solver(this->l_factory).on(this->exec);

    ASSERT_EQ(ic_factory->get_parameters().l_solver_factory, this->l_factory);
}


TEST_F(IcFactory, CanSetFactorizationFactory)
{
    auto ic_factory = ic_prec_type::build()
                          .with_factorization(this->fact_factory)
                          .on(this->exec);

    ASSERT_EQ(ic_factory->get_parameters().factorization_factory,
              this->fact_factory);
}


TEST_F(IcFactory, DeferredFactoryParameter)
{
    auto ic_factory = ic_prec_type::build()
                          .with_l_solver(solver_type::build())
                          .with_factorization(ic_type::build())
                          .on(this->exec);

    GKO_ASSERT_DYNAMIC_TYPE(ic_factory->get_parameters().l_solver_factory,
                            solver_type::Factory);
    GKO_ASSERT_DYNAMIC_TYPE(ic_factory->get_parameters().factorization_factory,
                            ic_type::Factory);
}


TEST_F(IcFactory, IcCannotBeCopied)
{
    ASSERT_FALSE(std::is_copy_constructible<ic_prec_type>::value);
    ASSERT_FALSE(std::is_copy_assignable<ic_prec_type>::value);
}


TEST_F(IcFactory, IcCannotBeMoved)
{
    ASSERT_FALSE(std::is_move_constructible<ic_prec_type>::value);
    ASSERT_FALSE(std::is_move_assignable<ic_prec_type>::value);
}

}  // namespace


GKO_END_DISABLE_DEPRECATION_WARNINGS
