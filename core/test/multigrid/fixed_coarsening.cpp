// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/multigrid/fixed_coarsening.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class FixedCoarseningFactory : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    using Vec = gko::matrix::Dense<value_type>;
    using MgLevel = gko::multigrid::FixedCoarsening<value_type, index_type>;
    FixedCoarseningFactory()
        : exec(gko::ReferenceExecutor::create()),
          fixed_coarsening_factory(
              MgLevel::build()
                  .with_coarse_rows(gko::array<index_type>(exec, {2, 3}))
                  .with_skip_sorting(true)
                  .on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename MgLevel::Factory> fixed_coarsening_factory;
};

TYPED_TEST_SUITE(FixedCoarseningFactory, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(FixedCoarseningFactory, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->fixed_coarsening_factory->get_executor(), this->exec);
}


TYPED_TEST(FixedCoarseningFactory, DefaultSetting)
{
    using MgLevel = typename TestFixture::MgLevel;
    auto factory = MgLevel::build().on(this->exec);

    ASSERT_EQ(factory->get_parameters().coarse_rows.get_const_data(), nullptr);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
}


TYPED_TEST(FixedCoarseningFactory, SetCoarseRows)
{
    using T = typename TestFixture::index_type;
    GKO_ASSERT_ARRAY_EQ(
        this->fixed_coarsening_factory->get_parameters().coarse_rows,
        gko::array<T>(this->exec, {2, 3}));
}


TYPED_TEST(FixedCoarseningFactory, SetSkipSorting)
{
    ASSERT_EQ(this->fixed_coarsening_factory->get_parameters().skip_sorting,
              true);
}


}  // namespace
