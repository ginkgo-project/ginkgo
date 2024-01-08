// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <tuple>
#include <type_traits>


#include <gtest/gtest.h>


#include "accessor/block_col_major.hpp"
#include "accessor/index_span.hpp"
#include "accessor/range.hpp"


namespace {


class BlockColMajorAccessor3d : public ::testing::Test {
protected:
    using span = gko::acc::index_span;
    static constexpr gko::acc::size_type dimensionality{3};

    using blk_col_major_range =
        gko::acc::range<gko::acc::block_col_major<int, dimensionality>>;

    // clang-format off
    int data[2 * 3 * 4]{
         1, 3, 5,
         2, 4, 6,
        -1,-2,-3,
        11,12,13,

        21,25,29,
        22,26,30,
        23,27,31,
        24,28,32

        /* This matrix actually looks like
        1, 2, -1, 11,
        3, 4, -2, 12,
        5, 6, -3, 13,

        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32
        */
    };
    // clang-format on
    const std::array<gko::acc::size_type, dimensionality> dim1{{2, 3, 4}};
    const std::array<gko::acc::size_type, dimensionality> dim2{{2, 2, 3}};
    blk_col_major_range default_r{dim1, data};
    blk_col_major_range custom_r{
        dim2, data,
        std::array<gko::acc::size_type, dimensionality - 1>{{12, 3}}};
};


TEST_F(BlockColMajorAccessor3d, ComputesCorrectStride)
{
    auto range_stride = default_r.get_accessor().stride;
    auto check_stride = std::array<gko::acc::size_type, 2>{{12, 3}};

    ASSERT_EQ(range_stride, check_stride);
}


TEST_F(BlockColMajorAccessor3d, CanAccessData)
{
    EXPECT_EQ(default_r(0, 0, 0), 1);
    EXPECT_EQ(custom_r(0, 0, 0), 1);
    EXPECT_EQ(default_r(0, 1, 0), 3);
    EXPECT_EQ(custom_r(0, 1, 0), 3);
    EXPECT_EQ(default_r(0, 1, 1), 4);
    EXPECT_EQ(default_r(0, 1, 3), 12);
    EXPECT_EQ(default_r(0, 2, 2), -3);
    EXPECT_EQ(default_r(1, 2, 1), 30);
    EXPECT_EQ(default_r(1, 2, 2), 31);
    EXPECT_EQ(default_r(1, 2, 3), 32);
}


TEST_F(BlockColMajorAccessor3d, CanWriteData)
{
    default_r(0, 0, 0) = 4;
    custom_r(1, 1, 1) = 100;

    EXPECT_EQ(default_r(0, 0, 0), 4);
    EXPECT_EQ(custom_r(0, 0, 0), 4);
    EXPECT_EQ(default_r(1, 1, 1), 100);
    EXPECT_EQ(custom_r(1, 1, 1), 100);
}


TEST_F(BlockColMajorAccessor3d, CanCreateSubrange)
{
    auto subr = custom_r(span{0u, 2u}, span{1u, 2u}, span{1u, 3u});

    EXPECT_EQ(subr(0, 0, 0), 4);
    EXPECT_EQ(subr(0, 0, 1), -2);
    EXPECT_EQ(subr(1, 0, 0), 26);
    EXPECT_EQ(subr(1, 0, 1), 27);
}


TEST_F(BlockColMajorAccessor3d, CanCreateRowVector)
{
    auto subr = default_r(1u, 2u, span{0u, 2u});

    EXPECT_EQ(subr(0, 0, 0), 29);
    EXPECT_EQ(subr(0, 0, 1), 30);
}


TEST_F(BlockColMajorAccessor3d, CanCreateColumnVector)
{
    auto subr = default_r(span{0u, 2u}, 1u, 3u);

    EXPECT_EQ(subr(0, 0, 0), 12);
    EXPECT_EQ(subr(1, 0, 0), 28);
}


}  // namespace
