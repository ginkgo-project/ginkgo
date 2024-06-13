// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class UpperTrs : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Solver = gko::solver::UpperTrs<value_type, index_type>;

    UpperTrs()
        : exec(gko::ReferenceExecutor::create()),
          upper_trs_factory(Solver::build().on(exec))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<typename Solver::Factory> upper_trs_factory;
};

TYPED_TEST_SUITE(UpperTrs, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(UpperTrs, UpperTrsFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->upper_trs_factory->get_executor(), this->exec);
}


TYPED_TEST(UpperTrs, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = gko::matrix::Dense<typename TestFixture::value_type>;
    std::shared_ptr<Mtx> rectangular_matrix =
        Mtx::create(this->exec, gko::dim<2>{1, 2});

    ASSERT_THROW(this->upper_trs_factory->generate(rectangular_matrix),
                 gko::DimensionMismatch);
}


}  // namespace
