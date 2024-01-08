// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/direct.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/lu.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class Direct : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Solver = gko::experimental::solver::Direct<value_type, index_type>;
    using Lu = gko::experimental::factorization::Lu<value_type, index_type>;

    Direct()
        : exec(gko::ReferenceExecutor::create()),
          factory(Solver::build().with_factorization(Lu::build()).on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename Solver::Factory> factory;
};

TYPED_TEST_SUITE(Direct, gko::test::ValueIndexTypes, PairTypenameNameGenerator);


TYPED_TEST(Direct, FactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->factory->get_executor(), this->exec);
}


TYPED_TEST(Direct, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = gko::matrix::Csr<typename TestFixture::value_type,
                                 typename TestFixture::index_type>;
    std::shared_ptr<Mtx> rectangular_matrix =
        Mtx::create(this->exec, gko::dim<2>{1, 2}, 0);

    ASSERT_THROW(this->factory->generate(rectangular_matrix),
                 gko::DimensionMismatch);
}


TYPED_TEST(Direct, PassExplicitFactory)
{
    using Solver = typename TestFixture::Solver;
    using Lu = typename TestFixture::Lu;
    auto lu_factory = gko::share(Lu::build().on(this->exec));

    auto factory =
        Solver::build().with_factorization(lu_factory).on(this->exec);

    ASSERT_EQ(factory->get_parameters().factorization, lu_factory);
}


}  // namespace
