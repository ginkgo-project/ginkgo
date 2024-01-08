// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/par_ic.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class ParIc : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using ic_factory_type = gko::factorization::ParIc<value_type, index_type>;
    using strategy_type = typename ic_factory_type::matrix_type::classical;

protected:
    ParIc() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(ParIc, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(ParIc, SetIterations)
{
    auto factory =
        TestFixture::ic_factory_type::build().with_iterations(5u).on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 5u);
}


TYPED_TEST(ParIc, SetSkip)
{
    auto factory =
        TestFixture::ic_factory_type::build().with_skip_sorting(true).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
}


TYPED_TEST(ParIc, SetLStrategy)
{
    auto strategy = std::make_shared<typename TestFixture::strategy_type>();

    auto factory =
        TestFixture::ic_factory_type::build().with_l_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


TYPED_TEST(ParIc, SetBothFactors)
{
    auto factory =
        TestFixture::ic_factory_type::build().with_both_factors(false).on(
            this->ref);

    ASSERT_FALSE(factory->get_parameters().both_factors);
}


TYPED_TEST(ParIc, SetDefaults)
{
    auto factory = TestFixture::ic_factory_type::build().on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 0u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, nullptr);
    ASSERT_TRUE(factory->get_parameters().both_factors);
}


TYPED_TEST(ParIc, SetEverything)
{
    auto strategy = std::make_shared<typename TestFixture::strategy_type>();

    auto factory = TestFixture::ic_factory_type::build()
                       .with_iterations(7u)
                       .with_skip_sorting(false)
                       .with_l_strategy(strategy)
                       .with_both_factors(false)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 7u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
    ASSERT_FALSE(factory->get_parameters().both_factors);
}


}  // namespace
