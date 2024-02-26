// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/iteration.hpp>


#include <gtest/gtest.h>


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


}  // namespace
