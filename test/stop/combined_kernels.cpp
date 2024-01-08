// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>


#include "test/utils/executor.hpp"


constexpr gko::size_type test_iterations = 10;


class Combined : public CommonTestFixture {
protected:
    Combined()
    {
        // Actually use an iteration stopping criterion because Criterion is an
        // abstract class
        factory = gko::stop::Combined::build()
                      .with_criteria(gko::stop::Iteration::build().on(ref),
                                     gko::stop::Iteration::build().on(ref),
                                     gko::stop::Iteration::build().on(ref))
                      .on(ref);
    }

    std::unique_ptr<gko::stop::Combined::Factory> factory;
};


TEST_F(Combined, CopyPropagatesExecutor)
{
    auto dev_factory = gko::clone(exec, factory.get());

    for (const auto& c : dev_factory->get_parameters().criteria) {
        ASSERT_TRUE(c->get_executor());
        ASSERT_EQ(exec.get(), c->get_executor().get());
    }
}


TEST_F(Combined, MovePropagatesExecutor)
{
    auto dev_factory = factory->create_default(exec);

    dev_factory->move_from(factory);

    for (const auto& c : dev_factory->get_parameters().criteria) {
        ASSERT_TRUE(c->get_executor());
        ASSERT_EQ(exec.get(), c->get_executor().get());
    }
}
