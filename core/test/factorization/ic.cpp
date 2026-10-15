// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ic.hpp>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class Ic : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using ic_factory_type = gko::factorization::Ic<value_type, index_type>;

protected:
    Ic() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Ic, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Ic, SetSkip)
{
    auto factory =
        TestFixture::ic_factory_type::build().with_skip_sorting(true).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
}


TYPED_TEST(Ic, SetLStrategy)
{
    auto strategy = gko::matrix::csr::spmv_strategy::load_balance;

    auto factory =
        TestFixture::ic_factory_type::build().with_l_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(Ic, SetLStrategyDeprecated)
{
    auto strategy = std::make_shared<
        typename TestFixture::ic_factory_type::matrix_type::load_balance>(
        this->ref);

    auto factory =
        TestFixture::ic_factory_type::build().with_l_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy,
              gko::matrix::csr::spmv_strategy::load_balance);
}


GKO_END_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(Ic, SetBothFactors)
{
    auto factory =
        TestFixture::ic_factory_type::build().with_both_factors(false).on(
            this->ref);

    ASSERT_FALSE(factory->get_parameters().both_factors);
}


TYPED_TEST(Ic, SetAlgorithm)
{
    auto factory =
        TestFixture::ic_factory_type::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->ref);

    ASSERT_EQ(factory->get_parameters().algorithm,
              gko::factorization::incomplete_algorithm::syncfree);
}


TYPED_TEST(Ic, SetDefaults)
{
    auto factory = TestFixture::ic_factory_type::build().on(this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy,
              gko::matrix::csr::spmv_strategy::classical);
    ASSERT_TRUE(factory->get_parameters().both_factors);
    ASSERT_EQ(factory->get_parameters().algorithm,
              gko::factorization::incomplete_algorithm::sparselib);
}


TYPED_TEST(Ic, SetEverything)
{
    auto strategy = gko::matrix::csr::spmv_strategy::classical;

    auto factory =
        TestFixture::ic_factory_type::build()
            .with_skip_sorting(false)
            .with_l_strategy(strategy)
            .with_both_factors(false)
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
    ASSERT_FALSE(factory->get_parameters().both_factors);
    ASSERT_EQ(factory->get_parameters().algorithm,
              gko::factorization::incomplete_algorithm::syncfree);
}
