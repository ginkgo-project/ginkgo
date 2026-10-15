// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/par_ilu.hpp>

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
    auto strategy = gko::matrix::csr::spmv_strategy::load_balance;

    auto factory =
        TestFixture::ilu_factory_type::build().with_l_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().l_strategy, strategy);
}


TYPED_TEST(ParIlu, SetUStrategy)
{
    auto strategy = gko::matrix::csr::spmv_strategy::load_balance;

    auto factory =
        TestFixture::ilu_factory_type::build().with_u_strategy(strategy).on(
            this->ref);

    ASSERT_EQ(factory->get_parameters().u_strategy, strategy);
}


GKO_BEGIN_DISABLE_DEPRECATION_WARNINGS


TYPED_TEST(ParIlu, SetStrategyDeprecated)
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


TYPED_TEST(ParIlu, SetDefaults)
{
    auto factory = TestFixture::ilu_factory_type::build().on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 0u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy,
              gko::matrix::csr::spmv_strategy::classical);
    ASSERT_EQ(factory->get_parameters().u_strategy,
              gko::matrix::csr::spmv_strategy::classical);
}


TYPED_TEST(ParIlu, SetEverything)
{
    auto l_strategy = gko::matrix::csr::spmv_strategy::load_balance;
    auto u_strategy = gko::matrix::csr::spmv_strategy::sparselib;

    auto factory = TestFixture::ilu_factory_type::build()
                       .with_iterations(7u)
                       .with_skip_sorting(false)
                       .with_l_strategy(l_strategy)
                       .with_u_strategy(u_strategy)
                       .on(this->ref);

    ASSERT_EQ(factory->get_parameters().iterations, 7u);
    ASSERT_EQ(factory->get_parameters().skip_sorting, false);
    ASSERT_EQ(factory->get_parameters().l_strategy, l_strategy);
    ASSERT_EQ(factory->get_parameters().u_strategy, u_strategy);
}


}  // namespace
