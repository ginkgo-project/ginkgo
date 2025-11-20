// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/stop/iteration.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/stop/iteration.hpp>


namespace {


constexpr gko::size_type test_iterations = 10;


class Iteration : public ::testing::Test {
protected:
    Iteration() : exec_{gko::ReferenceExecutor::create()}
    {
        factory_ = gko::stop::Iteration::build()
                       .with_max_iters(test_iterations)
                       .on(exec_);
    }

    std::unique_ptr<gko::stop::Iteration::Factory> factory_;
    std::shared_ptr<const gko::Executor> exec_;
};


TEST_F(Iteration, CanCreateFactory)
{
    ASSERT_NE(factory_, nullptr);
    ASSERT_EQ(factory_->get_parameters().max_iters, test_iterations);
}


TEST_F(Iteration, CanCreateCriterion)
{
    auto criterion = factory_->generate(nullptr, nullptr, nullptr);
    ASSERT_NE(criterion, nullptr);
}


TEST_F(Iteration, CanCreateMinIterationWithInnerCriterion)
{
    auto factory = gko::as<gko::stop::MinIterationWrapper::Factory>(
        gko::stop::min_iters(10, gko::stop::max_iters(100),
                             gko::stop::max_iters(1000))
            .on(exec_));

    auto inner = gko::as<gko::stop::Combined::Factory>(
        factory->get_parameters().inner_criterion);
    ASSERT_EQ(factory->get_parameters().min_iters, 10);
    ASSERT_EQ(inner->get_parameters().criteria.size(), 2);
    auto inner1 = gko::as<gko::stop::Iteration::Factory>(
        inner->get_parameters().criteria.at(0));
    auto inner2 = gko::as<gko::stop::Iteration::Factory>(
        inner->get_parameters().criteria.at(1));
    ASSERT_EQ(inner1->get_parameters().max_iters, 100);
    ASSERT_EQ(inner2->get_parameters().max_iters, 1000);
}


}  // namespace
