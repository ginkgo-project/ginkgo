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
