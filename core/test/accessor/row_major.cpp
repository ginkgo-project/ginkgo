// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <tuple>
#include <type_traits>


#include <gtest/gtest.h>


#include "accessor/index_span.hpp"
#include "accessor/range.hpp"
#include "accessor/row_major.hpp"


namespace {


class RowMajorAccessor : public ::testing::Test {
protected:
    using span = gko::acc::index_span;
    using dim_type = std::array<gko::acc::size_type, 2>;
    using stride_type = std::array<gko::acc::size_type, 1>;

    using row_major_int_range = gko::acc::range<gko::acc::row_major<int, 2>>;

    // clang-format off
    int data[9]{
        1, 2, -1,
        3, 4, -2,
        5, 6, -3
    };
    // clang-format on
    row_major_int_range r{dim_type{{3u, 2u}}, data, stride_type{{3u}}};
};


TEST_F(RowMajorAccessor, CanCreateDefaultStride)
{
    row_major_int_range r2{dim_type{{3, 3}}, data};

    EXPECT_EQ(r2(0, 0), 1);
    EXPECT_EQ(r2(0, 1), 2);
    EXPECT_EQ(r2(0, 2), -1);
    EXPECT_EQ(r2(1, 0), 3);
    EXPECT_EQ(r2(1, 1), 4);
    EXPECT_EQ(r2(1, 2), -2);
    EXPECT_EQ(r2(2, 0), 5);
    EXPECT_EQ(r2(2, 1), 6);
    EXPECT_EQ(r2(2, 2), -3);
}


TEST_F(RowMajorAccessor, CanAccessData)
{
    EXPECT_EQ(r(0, 0), 1);
    EXPECT_EQ(r(0, 1), 2);
    EXPECT_EQ(r(1, 0), 3);
    EXPECT_EQ(r(1, 1), 4);
    EXPECT_EQ(r(2, 0), 5);
    EXPECT_EQ(r(2, 1), 6);
}


TEST_F(RowMajorAccessor, CanWriteData)
{
    r(0, 0) = 4;

    EXPECT_EQ(r(0, 0), 4);
}


TEST_F(RowMajorAccessor, CanCreateSubrange)
{
    auto subr = r(span{1u, 3u}, span{0u, 2u});

    EXPECT_EQ(subr(0, 0), 3);
    EXPECT_EQ(subr(0, 1), 4);
    EXPECT_EQ(subr(1, 0), 5);
    EXPECT_EQ(subr(1, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateRowVector)
{
    auto subr = r(2u, span{0u, 2u});

    EXPECT_EQ(subr(0, 0), 5);
    EXPECT_EQ(subr(0, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateColumnVector)
{
    auto subr = r(span{0u, 3u}, 0u);

    EXPECT_EQ(subr(0, 0), 1);
    EXPECT_EQ(subr(1, 0), 3);
    EXPECT_EQ(subr(2, 0), 5);
}


TEST_F(RowMajorAccessor, CanAssignValues)
{
    r(1, 1) = r(0, 0);

    EXPECT_EQ(data[4], 1);
}


class RowMajorAccessor3d : public ::testing::Test {
protected:
    using span = gko::acc::index_span;
    static constexpr gko::acc::size_type dimensionality{3};

    using row_major_int_range =
        gko::acc::range<gko::acc::row_major<int, dimensionality>>;

    // clang-format off
    int data[2 * 3 * 4]{
        1, 2, -1, 11,
        3, 4, -2, 12,
        5, 6, -3, 13,

        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32
    };
    // clang-format on
    const std::array<gko::acc::size_type, dimensionality> dim1{{2, 3, 4}};
    const std::array<gko::acc::size_type, dimensionality> dim2{{2, 2, 3}};
    row_major_int_range default_r{dim1, data};
    row_major_int_range custom_r{
        dim2, data,
        std::array<gko::acc::size_type, dimensionality - 1>{{12, 4}}};
};


TEST_F(RowMajorAccessor3d, CanAccessData)
{
    EXPECT_EQ(default_r(0, 0, 0), 1);
    EXPECT_EQ(custom_r(0, 0, 0), 1);
    EXPECT_EQ(default_r(0, 1, 0), 3);
    EXPECT_EQ(custom_r(0, 1, 0), 3);
    EXPECT_EQ(default_r(0, 1, 3), 12);
    EXPECT_EQ(default_r(0, 2, 2), -3);
    EXPECT_EQ(default_r(1, 2, 1), 30);
    EXPECT_EQ(default_r(1, 2, 2), 31);
    EXPECT_EQ(default_r(1, 2, 3), 32);
}


TEST_F(RowMajorAccessor3d, CanWriteData)
{
    default_r(0, 0, 0) = 4;
    custom_r(1, 1, 1) = 100;

    EXPECT_EQ(default_r(0, 0, 0), 4);
    EXPECT_EQ(custom_r(0, 0, 0), 4);
    EXPECT_EQ(default_r(1, 1, 1), 100);
    EXPECT_EQ(custom_r(1, 1, 1), 100);
}


TEST_F(RowMajorAccessor3d, CanCreateSubrange)
{
    auto subr = custom_r(span{0u, 2u}, span{1u, 2u}, span{1u, 3u});

    EXPECT_EQ(subr(0, 0, 0), 4);
    EXPECT_EQ(subr(0, 0, 1), -2);
    EXPECT_EQ(subr(1, 0, 0), 26);
    EXPECT_EQ(subr(1, 0, 1), 27);
}


TEST_F(RowMajorAccessor3d, CanCreateRowVector)
{
    auto subr = default_r(1u, 2u, span{0u, 2u});

    EXPECT_EQ(subr(0, 0, 0), 29);
    EXPECT_EQ(subr(0, 0, 1), 30);
}


TEST_F(RowMajorAccessor3d, CanCreateColumnVector)
{
    auto subr = default_r(span{0u, 2u}, 1u, 3u);

    EXPECT_EQ(subr(0, 0, 0), 12);
    EXPECT_EQ(subr(1, 0, 0), 28);
}


TEST_F(RowMajorAccessor3d, CanAssignValues)
{
    default_r(1, 1, 1) = default_r(0, 0, 0);

    EXPECT_EQ(data[17], 1);
}


}  // namespace
