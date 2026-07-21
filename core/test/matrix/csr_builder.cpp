// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/csr_builder.hpp"

#include <memory>

#include <gtest/gtest.h>

#include "core/test/utils.hpp"


template <typename ValueIndexType>
class CsrBuilder : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;

protected:
    CsrBuilder()
        : exec(gko::ReferenceExecutor::create()),
          mtx(Mtx::create(exec, gko::dim<2>{2, 3}, 4))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::unique_ptr<Mtx> mtx;
};

TYPED_TEST_SUITE(CsrBuilder, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(CsrBuilder, ReturnsCorrectArrays)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    gko::matrix::CsrBuilder<value_type, index_type> builder{this->mtx};

    auto builder_col_idxs = builder.get_col_idx_array().get_data();
    auto builder_values = builder.get_value_array().get_data();
    auto ref_col_idxs = this->mtx->get_col_idxs();
    auto ref_values = this->mtx->get_values();
    auto builder_mtx = builder.get_matrix();

    ASSERT_EQ(builder_col_idxs, ref_col_idxs);
    ASSERT_EQ(builder_values, ref_values);
    ASSERT_EQ(builder_mtx, this->mtx.get());
}


TYPED_TEST(CsrBuilder, HelperFunctionOnUniquePtrReturnCorrect)
{
    auto ref_col_idxs = this->mtx->get_col_idxs();
    auto ref_values = this->mtx->get_values();

    auto builder = gko::matrix::make_builder_unique_ptr(this->mtx);

    ASSERT_EQ(builder->get_col_idx_array().get_data(), ref_col_idxs);
    ASSERT_EQ(builder->get_value_array().get_data(), ref_values);
    ASSERT_EQ(builder->get_matrix(), this->mtx.get());
}


TYPED_TEST(CsrBuilder, HelperFunctionOnSharedPtrReturnCorrect)
{
    auto ref_mtx = gko::share(this->mtx->clone());
    auto ref_col_idxs = ref_mtx->get_col_idxs();
    auto ref_values = ref_mtx->get_values();

    auto builder = gko::matrix::make_builder_unique_ptr(ref_mtx);

    ASSERT_EQ(builder->get_col_idx_array().get_data(), ref_col_idxs);
    ASSERT_EQ(builder->get_value_array().get_data(), ref_values);
    ASSERT_EQ(builder->get_matrix(), ref_mtx.get());
}


TYPED_TEST(CsrBuilder, HelperFunctionOnPlainPtrReturnCorrect)
{
    auto ref_col_idxs = this->mtx->get_col_idxs();
    auto ref_values = this->mtx->get_values();
    auto ref_mtx = this->mtx.get();

    auto builder = gko::matrix::make_builder_unique_ptr(ref_mtx);

    ASSERT_EQ(builder->get_col_idx_array().get_data(), ref_col_idxs);
    ASSERT_EQ(builder->get_value_array().get_data(), ref_values);
    ASSERT_EQ(builder->get_matrix(), ref_mtx);
}
