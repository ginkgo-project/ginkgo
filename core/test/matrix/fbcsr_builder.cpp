// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fbcsr_builder.hpp"


#include <memory>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class FbcsrBuilder : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Fbcsr<value_type, index_type>;

protected:
    FbcsrBuilder()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{4, 6}, 8, 2))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(FbcsrBuilder, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(FbcsrBuilder, ReturnsCorrectArrays)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::matrix::FbcsrBuilder<value_type, index_type> builder{this->mtx};

    auto builder_col_idxs = builder.get_col_idx_array().get_data();
    auto builder_values = builder.get_value_array().get_data();
    auto ref_col_idxs = this->mtx->get_col_idxs();
    auto ref_values = this->mtx->get_values();

    ASSERT_EQ(builder_col_idxs, ref_col_idxs);
    ASSERT_EQ(builder_values, ref_values);
}


}  // namespace
