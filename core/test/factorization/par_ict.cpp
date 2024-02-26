// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/factorization/par_ict.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class ParIct : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using ict_factory_type = gko::factorization::ParIct<value_type, index_type>;
    using strategy_type = typename ict_factory_type::matrix_type::classical;

protected:
    ParIct() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(ParIct, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(ParIct, SetIterations)
{
    auto factory =
        TestFixture::ict_factory_type::build().with_iterations(6u).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 6u);
}


TYPED_TEST(ParIct, SetSkip)
{
    auto factory =
        TestFixture::ict_factory_type::build().with_skip_sorting(true).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
}


TYPED_TEST(ParIct, SetApprox)
{
    auto factory = TestFixture::ict_factory_type::build()
                       .with_approximate_select(false)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().approximate_select, false);
}


TYPED_TEST(ParIct, SetDeterministic)
{
    auto factory = TestFixture::ict_factory_type::build()
                       .with_deterministic_sample(true)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().deterministic_sample, true);
}


TYPED_TEST(ParIct, SetFillIn)
{
    auto factory =
        TestFixture::ict_factory_type::build().with_fill_in_limit(1.2).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().fill_in_limit, 1.2);
}


TYPED_TEST(ParIct, SetLStrategy)
{
    auto strategy = std::make_shared<typename TestFixture::strategy_type>();

    auto factory =
        TestFixture::ict_factory_type::build().with_l_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


TYPED_TEST(ParIct, SetDefaults)
{
    auto factory = TestFixture::ict_factory_type::build().on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 5u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().approximate_select, true);
    ASSERT_EQ(factory->get_parameters().deterministic_sample, false);
    ASSERT_EQ(factory->get_parameters().fill_in_limit, 2.0);
    ASSERT_EQ(factory->get_parameters().l_strategy, nullptr);
}


TYPED_TEST(ParIct, SetEverything)
{
    auto strategy = std::make_shared<typename TestFixture::strategy_type>();

    auto factory = TestFixture::ict_factory_type::build()
                       .with_iterations(7u)
                       .with_skip_sorting(true)
                       .with_approximate_select(false)
                       .with_deterministic_sample(true)
                       .with_fill_in_limit(1.2)
                       .with_l_strategy(strategy)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 7u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
    ASSERT_EQ(factory->get_parameters().approximate_select, false);
    ASSERT_EQ(factory->get_parameters().deterministic_sample, true);
    ASSERT_EQ(factory->get_parameters().fill_in_limit, 1.2);
    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


}  // namespace
