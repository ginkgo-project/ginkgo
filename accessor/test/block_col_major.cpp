// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "accessor/block_col_major.hpp"

#include <array>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include "accessor/index_span.hpp"
#include "accessor/range.hpp"
#include "index_types.hpp"


namespace {


template <typename IndexType>
class BlockColMajorAccessor3d : public ::testing::Test {
protected:
    using span = acc::index_span;
    static constexpr acc::size_type dimensionality{3};

    using blk_col_major_range =
        acc::range<acc::block_col_major<int, dimensionality, IndexType>>;

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
    const std::array<acc::size_type, dimensionality> dim1{{2, 3, 4}};
    const std::array<acc::size_type, dimensionality> dim2{{2, 2, 3}};
    blk_col_major_range default_r{dim1, data};
    blk_col_major_range custom_r{
        dim2, data, std::array<acc::size_type, dimensionality - 1>{{12, 3}}};
};


TYPED_TEST_SUITE(BlockColMajorAccessor3d, acc::test::AltIndexTypes);


TYPED_TEST(BlockColMajorAccessor3d, ComputesCorrectStride)
{
    auto range_stride = this->default_r.get_accessor().stride;
    auto check_stride = std::array<acc::size_type, 2>{{12, 3}};

    ASSERT_EQ(range_stride, check_stride);
}


TYPED_TEST(BlockColMajorAccessor3d, CanAccessData)
{
    EXPECT_EQ(this->default_r(0, 0, 0), 1);
    EXPECT_EQ(this->custom_r(0, 0, 0), 1);
    EXPECT_EQ(this->default_r(0, 1, 0), 3);
    EXPECT_EQ(this->custom_r(0, 1, 0), 3);
    EXPECT_EQ(this->default_r(0, 1, 1), 4);
    EXPECT_EQ(this->default_r(0, 1, 3), 12);
    EXPECT_EQ(this->default_r(0, 2, 2), -3);
    EXPECT_EQ(this->default_r(1, 2, 1), 30);
    EXPECT_EQ(this->default_r(1, 2, 2), 31);
    EXPECT_EQ(this->default_r(1, 2, 3), 32);
}


TYPED_TEST(BlockColMajorAccessor3d, CanWriteData)
{
    this->default_r(0, 0, 0) = 4;
    this->custom_r(1, 1, 1) = 100;

    EXPECT_EQ(this->default_r(0, 0, 0), 4);
    EXPECT_EQ(this->custom_r(0, 0, 0), 4);
    EXPECT_EQ(this->default_r(1, 1, 1), 100);
    EXPECT_EQ(this->custom_r(1, 1, 1), 100);
}


TYPED_TEST(BlockColMajorAccessor3d, CanCreateSubrange)
{
    using span = typename TestFixture::span;
    auto subr = this->custom_r(span{0u, 2u}, span{1u, 2u}, span{1u, 3u});

    EXPECT_EQ(subr(0, 0, 0), 4);
    EXPECT_EQ(subr(0, 0, 1), -2);
    EXPECT_EQ(subr(1, 0, 0), 26);
    EXPECT_EQ(subr(1, 0, 1), 27);
}


TYPED_TEST(BlockColMajorAccessor3d, CanCreateRowVector)
{
    using span = typename TestFixture::span;
    auto subr = this->default_r(1u, 2u, span{0u, 2u});

    EXPECT_EQ(subr(0, 0, 0), 29);
    EXPECT_EQ(subr(0, 0, 1), 30);
}


TYPED_TEST(BlockColMajorAccessor3d, CanCreateColumnVector)
{
    using span = typename TestFixture::span;
    auto subr = this->default_r(span{0u, 2u}, 1u, 3u);

    EXPECT_EQ(subr(0, 0, 0), 12);
    EXPECT_EQ(subr(1, 0, 0), 28);
}


TYPED_TEST(BlockColMajorAccessor3d, ComputeIndexReturnsIndexType)
{
    using size_array = std::array<acc::size_type, 3>;
    using stride_array = std::array<acc::size_type, 2>;
    using result_type =
        decltype(acc::helper::blk_col_major::compute_index<TypeParam>(
            std::declval<size_array>(), std::declval<stride_array>(), 0, 0, 0));

    static_assert(std::is_same<result_type, TypeParam>::value,
                  "block_col_major index computation must return IndexType");
    SUCCEED();
}


}  // namespace
