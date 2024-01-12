// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/uniform_coarsening.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class UniformCoarseningFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::UniformCoarsening<value_type, index_type>;
    UniformCoarseningFactory()
        : exec(gko::ReferenceExecutor::create()),
          uniform_coarsening1_factory(
              MgLevel::build().with_num_jumps(4u).with_skip_sorting(true).on(
                  exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename MgLevel::Factory> uniform_coarsening1_factory;
};

TYPED_TEST_SUITE(UniformCoarseningFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(UniformCoarseningFactory, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->uniform_coarsening1_factory->get_executor(), this->exec);
}


TYPED_TEST(UniformCoarseningFactory, DefaultSetting)
{
    using MgLevel = typename TestFixture::MgLevel;
    auto factory = MgLevel::build().on(this->exec);

    ASSERT_EQ(factory->get_parameters().num_jumps, 2u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
}


TYPED_TEST(UniformCoarseningFactory, SetNumJumps)
{
    ASSERT_EQ(this->uniform_coarsening1_factory->get_parameters().num_jumps,
              4u);
}


TYPED_TEST(UniformCoarseningFactory, SetSkipSorting)
{
    ASSERT_EQ(this->uniform_coarsening1_factory->get_parameters().skip_sorting,
              true);
}


}  // namespace
