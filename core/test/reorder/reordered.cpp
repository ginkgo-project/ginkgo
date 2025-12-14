// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/reorder/reordered.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/reorder/rcm.hpp>
#include <ginkgo/core/solver/bicgstab.hpp>

#include "core/test/utils/assertions.hpp"


class Reordered : public ::testing::Test {
protected:
    using value_type = double;
    using index_type = gko::int32;
    using solver_type = gko::solver::Bicgstab<value_type>;
    using reorder_type = gko::experimental::reorder::Rcm<index_type>;
    using reordered_type =
        gko::experimental::reorder::Reordered<value_type, index_type>;

    Reordered()
        : exec(gko::ReferenceExecutor::create()),
          reordered_factory(reordered_type::build().on(exec)),
          reordering_factory(reorder_type::build().on(exec)),
          solver_factory(solver_type::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<typename reordered_type::Factory> reordered_factory;
    std::shared_ptr<reorder_type> reordering_factory;
    std::shared_ptr<typename solver_type::Factory> solver_factory;
};


TEST_F(Reordered, KnowsItsExecutor)
{
    auto reordered_factory = reordered_type::build().on(this->exec);

    ASSERT_EQ(reordered_factory->get_executor(), this->exec);
}


TEST_F(Reordered, CanSetReorderingFactory)
{
    auto reordered_factory = reordered_type::build()
                                 .with_reordering(this->reordering_factory)
                                 .on(this->exec);

    ASSERT_EQ(reordered_factory->get_parameters().reordering,
              this->reordering_factory);
}


TEST_F(Reordered, CanSetReorderingFactoryDeferred)
{
    auto reordered_factory = reordered_type::build()
                                 .with_reordering(reorder_type::build())
                                 .on(this->exec);

    GKO_ASSERT_DYNAMIC_TYPE_EQ(reordered_factory->get_parameters().reordering,
                               this->reordering_factory);
    ASSERT_EQ(reordered_factory->get_parameters().reordering->get_executor(),
              this->exec);
}


TEST_F(Reordered, CanSetInnerOperatorFactory)
{
    auto reordered_factory = reordered_type::build()
                                 .with_inner_operator(this->solver_factory)
                                 .on(this->exec);

    ASSERT_EQ(reordered_factory->get_parameters().inner_operator,
              this->solver_factory);
}


TEST_F(Reordered, CanSetInnerOperatorFactoryDeferred)
{
    auto reordered_factory = reordered_type::build()
                                 .with_inner_operator(solver_type::build())
                                 .on(this->exec);

    GKO_ASSERT_DYNAMIC_TYPE_EQ(
        reordered_factory->get_parameters().inner_operator,
        this->solver_factory);
    ASSERT_EQ(
        reordered_factory->get_parameters().inner_operator->get_executor(),
        this->exec);
}
