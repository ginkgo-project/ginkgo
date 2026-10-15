// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/ilu.hpp>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class Ilu : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using ilu_factory_type = gko::factorization::Ilu<value_type, index_type>;

protected:
    Ilu() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Ilu, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Ilu, SetSkip)
{
    auto factory =
        TestFixture::ilu_factory_type::build().with_skip_sorting(true).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, true);
}


TYPED_TEST(Ilu, SetLStrategy)
{
    auto strategy = gko::matrix::csr::spmv_strategy::load_balance;

    auto factory =
        TestFixture::ilu_factory_type::build().with_l_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


TYPED_TEST(Ilu, SetUStrategy)
{
    auto strategy = gko::matrix::csr::spmv_strategy::load_balance;

    auto factory =
        TestFixture::ilu_factory_type::build().with_u_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().u_strategy, strategy);
}


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(Ilu, SetStrategyDeprecated)
{
    using matrix_type = typename TestFixture::ilu_factory_type::matrix_type;
    auto l_strategy =
        std::make_shared<typename matrix_type::load_balance>(this->ref);
    auto u_strategy = std::make_shared<typename matrix_type::sparselib>();

    auto factory = TestFixture::ilu_factory_type::build()
                       .with_l_strategy(l_strategy)
                       .with_u_strategy(u_strategy)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy,
              gko::matrix::csr::spmv_strategy::load_balance);
    ASSERT_EQ(factory->get_parameters().u_strategy,
              gko::matrix::csr::spmv_strategy::sparselib);
}


GKO_END_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(Ilu, SetAlgorithm)
{
    auto factory =
        TestFixture::ilu_factory_type::build()
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->ref);

    ASSERT_EQ(factory->get_parameters().algorithm,
              gko::factorization::incomplete_algorithm::syncfree);
}


TYPED_TEST(Ilu, SetDefaults)
{
    auto factory = TestFixture::ilu_factory_type::build().on(this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy,
              gko::matrix::csr::spmv_strategy::classical);
    ASSERT_EQ(factory->get_parameters().u_strategy,
              gko::matrix::csr::spmv_strategy::classical);
    ASSERT_EQ(factory->get_parameters().algorithm,
              gko::factorization::incomplete_algorithm::sparselib);
}


TYPED_TEST(Ilu, SetEverything)
{
    auto l_strategy = gko::matrix::csr::spmv_strategy::load_balance;
    auto u_strategy = gko::matrix::csr::spmv_strategy::sparselib;

    auto factory =
        TestFixture::ilu_factory_type::build()
            .with_skip_sorting(false)
            .with_l_strategy(l_strategy)
            .with_u_strategy(u_strategy)
            .with_algorithm(gko::factorization::incomplete_algorithm::syncfree)
            .on(this->ref);

    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(factory->get_parameters().u_strategy, u_strategy);
    ASSERT_EQ(factory->get_parameters().algorithm,
              gko::factorization::incomplete_algorithm::syncfree);
}
