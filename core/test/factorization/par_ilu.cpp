// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/par_ilu.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class ParIlu : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using ilu_factory_type = gko::factorization::ParIlu<value_type, index_type>;
    using strategy_type = typename ilu_factory_type::matrix_type::classical;

protected:
    ParIlu() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(ParIlu, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(ParIlu, SetIterations)
{
    auto factory =
        TestFixture::ilu_factory_type::build().with_iterations(5u).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 5u);
}


TYPED_TEST(ParIlu, SetSkip)
{
    auto factory =
        TestFixture::ilu_factory_type::build().with_skip_sorting(true).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
}


TYPED_TEST(ParIlu, SetLStrategy)
{
    auto strategy = std::make_shared<typename TestFixture::strategy_type>();

    auto factory =
        TestFixture::ilu_factory_type::build().with_l_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


TYPED_TEST(ParIlu, SetUStrategy)
{
    auto strategy = std::make_shared<typename TestFixture::strategy_type>();

    auto factory =
        TestFixture::ilu_factory_type::build().with_u_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().u_strategy, strategy);
}


TYPED_TEST(ParIlu, SetDefaults)
{
    auto factory = TestFixture::ilu_factory_type::build().on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 0u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, nullptr);
    ASSERT_EQ(factory->get_parameters().u_strategy, nullptr);
}


TYPED_TEST(ParIlu, SetEverything)
{
    auto strategy = std::make_shared<typename TestFixture::strategy_type>();
    auto strategy2 = std::make_shared<typename TestFixture::strategy_type>();

    auto factory = TestFixture::ilu_factory_type::build()
                       .with_iterations(7u)
                       .with_skip_sorting(false)
                       .with_l_strategy(strategy)
                       .with_u_strategy(strategy2)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 7u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
    ASSERT_EQ(factory->get_parameters().u_strategy, strategy2);
}


}  // namespace
