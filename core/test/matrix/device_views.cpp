// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <ginkgo/core/matrix/device_views.hpp>

#include "core/test/utils.hpp"


template <typename T>
class DenseView : public ::testing::Test {};

TYPED_TEST_SUITE(DenseView, gko::test::ValueTypes, TypenameNameGenerator);


TYPED_TEST(DenseView, AccessWorks)
{
    std::vector<TypeParam> values(10);
    gko::matrix::view::dense<TypeParam> view{gko::dim<2>{2, 2}, 3,
                                             values.data()};
    auto const_view = view.as_const();

    ASSERT_EQ(view.size, gko::dim<2>(2, 2));
    ASSERT_EQ(view.stride, 3);
    ASSERT_EQ(view.values, values.data());
    ASSERT_EQ(&view(0, 0), &values[0]);
    ASSERT_EQ(&view(1, 0), &values[3]);
    ASSERT_EQ(&view(1, 1), &values[4]);
    ASSERT_EQ(const_view.size, view.size);
    ASSERT_EQ(const_view.stride, view.stride);
    ASSERT_EQ(const_view.values, view.values);
}


TYPED_TEST(DenseView, AssertTriggersOnOutOfBoundsDeathTest)
{
#ifdef NDEBUG
    GTEST_SKIP() << "Assertion is only enabled in debug mode";
#endif

    std::vector<TypeParam> values(10);
    gko::matrix::view::dense<TypeParam> view{gko::dim<2>{2, 2}, 3,
                                             values.data()};

    EXPECT_EXIT((void)(view(3, 0)), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(view(0, 3)), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(view(3, 3)), check_assertion_exit_code, "");
}


template <typename ValueIndexType>
class EllView : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(EllView, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(EllView, AccessWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::vector<value_type> values(12);
    std::vector<index_type> col_idxs(12);
    gko::matrix::view::ell<value_type, index_type> view{
        gko::dim<2>{2, 5}, 3, 4, values.data(), col_idxs.data()};
    auto const_view = view.as_const();

    ASSERT_EQ(view.size, gko::dim<2>(2, 5));
    ASSERT_EQ(view.stride, 4);
    ASSERT_EQ(view.num_stored_elements_per_row, 3);
    ASSERT_EQ(view.values, values.data());
    ASSERT_EQ(view.col_idxs, col_idxs.data());
    ASSERT_EQ(&view.val_at(0, 0), &values[0]);
    ASSERT_EQ(&view.val_at(1, 0), &values[1]);
    ASSERT_EQ(&view.val_at(1, 1), &values[5]);
    ASSERT_EQ(&view.col_at(0, 0), &col_idxs[0]);
    ASSERT_EQ(&view.col_at(1, 0), &col_idxs[1]);
    ASSERT_EQ(&view.col_at(1, 1), &col_idxs[5]);
    ASSERT_EQ(const_view.size, view.size);
    ASSERT_EQ(const_view.stride, view.stride);
    ASSERT_EQ(const_view.num_stored_elements_per_row,
              view.num_stored_elements_per_row);
    ASSERT_EQ(const_view.values, view.values);
    ASSERT_EQ(const_view.col_idxs, view.col_idxs);
}
