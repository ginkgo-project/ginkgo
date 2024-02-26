// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/pgm.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class PgmFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::Pgm<value_type, index_type>;
    PgmFactory()
        : exec(gko::ReferenceExecutor::create()),
          pgm_factory(MgLevel::build()
                          .with_max_iterations(2u)
                          .with_max_unassigned_ratio(0.1)
                          .with_deterministic(true)
                          .with_skip_sorting(true)
                          .on(exec))

    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename MgLevel::Factory> pgm_factory;
};

TYPED_TEST_SUITE(PgmFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(PgmFactory, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->pgm_factory->get_executor(), this->exec);
}


TYPED_TEST(PgmFactory, DefaultSetting)
{
    using MgLevel = typename TestFixture::MgLevel;
    auto factory = MgLevel::build().on(this->exec);

    ASSERT_EQ(factory->get_parameters().max_iterations, 15u);
    ASSERT_EQ(factory->get_parameters().max_unassigned_ratio, 0.05);
    ASSERT_EQ(factory->get_parameters().deterministic, false);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
}


TYPED_TEST(PgmFactory, SetMaxIterations)
{
    ASSERT_EQ(this->pgm_factory->get_parameters().max_iterations, 2u);
}


TYPED_TEST(PgmFactory, SetMaxUnassignedPercentage)
{
    ASSERT_EQ(this->pgm_factory->get_parameters().max_unassigned_ratio, 0.1);
}


TYPED_TEST(PgmFactory, SetDeterministic)
{
    ASSERT_EQ(this->pgm_factory->get_parameters().deterministic, true);
}


TYPED_TEST(PgmFactory, SetSkipSorting)
{
    ASSERT_EQ(this->pgm_factory->get_parameters().skip_sorting, true);
}


}  // namespace
