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


TYPED_TEST(DenseView, AssertTriggersInConstructorDeathTest)
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


TYPED_TEST(DenseView, AssertTriggersOnOutOfBoundsDeathTest)
{
#ifdef NDEBUG
    GTEST_SKIP() << "Assertion is only enabled in debug mode";
#endif

    EXPECT_EXIT((void)(gko::matrix::view::dense<TypeParam>{gko::dim<2>{2, 2}, 1,
                                                           nullptr}),
                check_assertion_exit_code, "");
}


template <typename ValueIndexType>
class CooView : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(CooView, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(CooView, AccessWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::vector<value_type> values{1, 2, 3};
    std::vector<index_type> row_idxs{0, 1, 2};
    std::vector<index_type> col_idxs{1, 0, 2};
    gko::matrix::view::coo<value_type, index_type> view{
        gko::dim<2>{3, 3}, 3, values.data(), row_idxs.data(), col_idxs.data()};
    auto const_view = view.as_const();

    ASSERT_EQ(view.size, gko::dim<2>(3, 3));
    ASSERT_EQ(view.num_stored_elements, 3);
    ASSERT_EQ(view.values, values.data());
    ASSERT_EQ(view.row_idxs, row_idxs.data());
    ASSERT_EQ(view.col_idxs, col_idxs.data());
    ASSERT_EQ(const_view.size, view.size);
    ASSERT_EQ(const_view.num_stored_elements, view.num_stored_elements);
    ASSERT_EQ(const_view.values, view.values);
    ASSERT_EQ(const_view.row_idxs, view.row_idxs);
    ASSERT_EQ(const_view.col_idxs, view.col_idxs);
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


TYPED_TEST(EllView, AssertTriggersInConstructorDeathTest)
{
#ifdef NDEBUG
    GTEST_SKIP() << "Assertion is only enabled in debug mode";
#endif

    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    // stride is smaller than dim[0]
    EXPECT_EXIT((void)(gko::matrix::view::ell<value_type, index_type>{
                    gko::dim<2>{2, 5}, 3, 1, nullptr, nullptr}),
                check_assertion_exit_code, "");
}


TYPED_TEST(EllView, AssertTriggersOnOutOfBoundsDeathTest)
{
#ifdef NDEBUG
    GTEST_SKIP() << "Assertion is only enabled in debug mode";
#endif

    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::vector<value_type> values(12);
    std::vector<index_type> col_idxs(12);
    gko::matrix::view::ell<value_type, index_type> view{
        gko::dim<2>{2, 5}, 3, 4, values.data(), col_idxs.data()};

    // access exceed nonzero per row
    EXPECT_EXIT((void)(view.val_at(1, 3)), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(view.col_at(1, 3)), check_assertion_exit_code, "");
    // access exceed the dimension
    EXPECT_EXIT((void)(view.val_at(2, 0)), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(view.col_at(2, 0)), check_assertion_exit_code, "");
    // access exceed the stride
    EXPECT_EXIT((void)(view.val_at(4, 0)), check_assertion_exit_code, "");
    EXPECT_EXIT((void)(view.col_at(4, 0)), check_assertion_exit_code, "");
}


template <typename ValueIndexType>
class SellpView : public ::testing::Test {
public:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
};

TYPED_TEST_SUITE(SellpView, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(SellpView, AccessWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    std::vector<value_type> values(21);
    std::vector<index_type> col_idxs(21);
    std::vector<gko::size_type> slice_lengths{3, 4};
    std::vector<gko::size_type> slice_sets{0, 3, 7};
    gko::matrix::view::sellp<value_type, index_type> view{gko::dim<2>{3, 5},
                                                          2,
                                                          3,
                                                          7,
                                                          values.data(),
                                                          col_idxs.data(),
                                                          slice_lengths.data(),
                                                          slice_sets.data()};
    auto const_view = view.as_const();

    ASSERT_EQ(view.size, gko::dim<2>(3, 5));
    ASSERT_EQ(view.slice_size, 2);
    ASSERT_EQ(view.stride_factor, 3);
    ASSERT_EQ(view.total_cols, 7);
    ASSERT_EQ(view.values, values.data());
    ASSERT_EQ(view.col_idxs, col_idxs.data());
    ASSERT_EQ(view.slice_lengths, slice_lengths.data());
    ASSERT_EQ(view.slice_sets, slice_sets.data());
    ASSERT_EQ(&view.val_at(0, slice_sets.at(0), 0), &values[0]);
    ASSERT_EQ(&view.val_at(1, slice_sets.at(0), 0), &values[1]);
    ASSERT_EQ(&view.val_at(1, slice_sets.at(0), 1), &values[3]);
    ASSERT_EQ(&view.val_at(0, slice_sets.at(1), 0), &values[6]);
    ASSERT_EQ(&view.val_at(1, slice_sets.at(1), 0), &values[7]);
    ASSERT_EQ(&view.val_at(1, slice_sets.at(1), 1), &values[9]);
    ASSERT_EQ(&view.col_at(0, slice_sets.at(0), 0), &col_idxs[0]);
    ASSERT_EQ(&view.col_at(1, slice_sets.at(0), 0), &col_idxs[1]);
    ASSERT_EQ(&view.col_at(1, slice_sets.at(0), 1), &col_idxs[3]);
    ASSERT_EQ(&view.col_at(0, slice_sets.at(1), 0), &col_idxs[6]);
    ASSERT_EQ(&view.col_at(1, slice_sets.at(1), 0), &col_idxs[7]);
    ASSERT_EQ(&view.col_at(1, slice_sets.at(1), 1), &col_idxs[9]);
    ASSERT_EQ(const_view.size, view.size);
    ASSERT_EQ(const_view.slice_size, view.slice_size);
    ASSERT_EQ(const_view.stride_factor, view.stride_factor);
    ASSERT_EQ(const_view.total_cols, view.total_cols);
    ASSERT_EQ(const_view.values, view.values);
    ASSERT_EQ(const_view.col_idxs, view.col_idxs);
    ASSERT_EQ(const_view.slice_lengths, view.slice_lengths);
    ASSERT_EQ(const_view.slice_sets, view.slice_sets);
}
