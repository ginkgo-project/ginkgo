// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/coo_builder.hpp"


#include <memory>


#include <gtest/gtest.h>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class CooBuilder : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Coo<value_type, index_type>;

    CooBuilder()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(CooBuilder, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(CooBuilder, ReturnsCorrectArrays)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::matrix::CooBuilder<value_type, index_type> builder{this->mtx};

    auto builder_row_idxs = builder.get_row_idx_array().get_data();
    auto builder_col_idxs = builder.get_col_idx_array().get_data();
    auto builder_values = builder.get_value_array().get_data();
    auto ref_row_idxs = this->mtx->get_row_idxs();
    auto ref_col_idxs = this->mtx->get_col_idxs();
    auto ref_values = this->mtx->get_values();

    ASSERT_EQ(builder_row_idxs, ref_row_idxs);
    ASSERT_EQ(builder_col_idxs, ref_col_idxs);
    ASSERT_EQ(builder_values, ref_values);
}


}  // namespace
