/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/base/range_accessors.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/types.hpp>


#include "core/test/utils.hpp"


namespace {


class RowMajorAccessor : public ::testing::Test {
protected:
    using span = gko::span;

    using row_major_int_range = gko::range<gko::accessor::row_major<int, 2>>;

    // clang-format off
    int data[9]{
        1, 2, -1,
        3, 4, -2,
        5, 6, -3
    };
    // clang-format on
    row_major_int_range r{data, 3u, 2u, 3u};
};


TEST_F(RowMajorAccessor, CanCreateWithDim)
{
    row_major_int_range r2{data, gko::dim<2>{3, 2}, 3u};

    EXPECT_EQ(r2(0, 0), 1);
    EXPECT_EQ(r2(0, 1), 2);
    EXPECT_EQ(r2(1, 0), 3);
    EXPECT_EQ(r2(1, 1), 4);
    EXPECT_EQ(r2(2, 0), 5);
    EXPECT_EQ(r2(2, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateDefaultStride)
{
    row_major_int_range r2{data, gko::dim<2>{3, 3}};

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
    auto subr = r(span{1, 3}, span{0, 2});

    EXPECT_EQ(subr(0, 0), 3);
    EXPECT_EQ(subr(0, 1), 4);
    EXPECT_EQ(subr(1, 0), 5);
    EXPECT_EQ(subr(1, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateRowVector)
{
    auto subr = r(2, span{0, 2});

    EXPECT_EQ(subr(0, 0), 5);
    EXPECT_EQ(subr(0, 1), 6);
}


TEST_F(RowMajorAccessor, CanCreateColumnVector)
{
    auto subr = r(span{0, 3}, 0);

    EXPECT_EQ(subr(0, 0), 1);
    EXPECT_EQ(subr(1, 0), 3);
    EXPECT_EQ(subr(2, 0), 5);
}


TEST_F(RowMajorAccessor, CanAssignValues)
{
    r(1, 1) = r(0, 0);

    EXPECT_EQ(data[4], 1);
}


TEST_F(RowMajorAccessor, CanAssignSubranges)
{
    r(0, span{0, 2}) = r(1, span{0, 2});

    EXPECT_EQ(data[0], 3);
    EXPECT_EQ(data[1], 4);
    EXPECT_EQ(data[2], -1);
    EXPECT_EQ(data[3], 3);
    EXPECT_EQ(data[4], 4);
    EXPECT_EQ(data[5], -2);
    EXPECT_EQ(data[6], 5);
    EXPECT_EQ(data[7], 6);
    EXPECT_EQ(data[8], -3);
}


class RowMajorAccessor3d : public ::testing::Test {
protected:
    using span = gko::span;
    static constexpr gko::size_type dimensionality{3};

    using row_major_int_range =
        gko::range<gko::accessor::row_major<int, dimensionality>>;

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
    const gko::dim<dimensionality> dim1{2, 3, 4};
    const gko::dim<dimensionality> dim2{2, 2, 3};
    row_major_int_range default_r{data, dim1};
    row_major_int_range custom_r{
        data, dim2,
        std::array<const gko::size_type, dimensionality - 1>{12, 4}};
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
    auto subr = custom_r(span{0, 2}, span{1, 2}, span{1, 3});

    EXPECT_EQ(subr(0, 0, 0), 4);
    EXPECT_EQ(subr(0, 0, 1), -2);
    EXPECT_EQ(subr(1, 0, 0), 26);
    EXPECT_EQ(subr(1, 0, 1), 27);
}


TEST_F(RowMajorAccessor3d, CanCreateRowVector)
{
    auto subr = default_r(1u, 2u, span{0, 2});

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


TEST_F(RowMajorAccessor3d, CanAssignSubranges)
{
    default_r(1u, span{0, 2}, span{0, 3}) =
        custom_r(0u, span{0, 2}, span{0, 3});

    EXPECT_EQ(data[12], 1);
    EXPECT_EQ(data[13], 2);
    EXPECT_EQ(data[14], -1);
    EXPECT_EQ(data[15], 24);
    EXPECT_EQ(data[16], 3);
    EXPECT_EQ(data[17], 4);
    EXPECT_EQ(data[18], -2);
    EXPECT_EQ(data[19], 28);
    EXPECT_EQ(data[20], 29);
    EXPECT_EQ(data[21], 30);
    EXPECT_EQ(data[22], 31);
    EXPECT_EQ(data[23], 32);
}


class BlockColMajorAccessor3d : public ::testing::Test {
protected:
    using span = gko::span;
    static constexpr gko::size_type dimensionality{3};

    using blk_col_major_range =
        gko::range<gko::accessor::block_col_major<int, dimensionality>>;

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
    const gko::dim<dimensionality> dim1{2, 3, 4};
    const gko::dim<dimensionality> dim2{2, 2, 3};
    blk_col_major_range default_r{data, dim1};
    blk_col_major_range custom_r{
        data, dim2,
        std::array<const gko::size_type, dimensionality - 1>{12, 3}};
};


TEST_F(BlockColMajorAccessor3d, ComputesCorrectStride)
{
    auto range_stride = default_r.get_accessor().stride;
    auto check_stride = std::array<const gko::size_type, 2>{12, 3};

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
    auto subr = custom_r(span{0, 2}, span{1, 2}, span{1, 3});

    EXPECT_EQ(subr(0, 0, 0), 4);
    EXPECT_EQ(subr(0, 0, 1), -2);
    EXPECT_EQ(subr(1, 0, 0), 26);
    EXPECT_EQ(subr(1, 0, 1), 27);
}


TEST_F(BlockColMajorAccessor3d, CanCreateRowVector)
{
    auto subr = default_r(1u, 2u, span{0, 2});

    EXPECT_EQ(subr(0, 0, 0), 29);
    EXPECT_EQ(subr(0, 0, 1), 30);
}


TEST_F(BlockColMajorAccessor3d, CanCreateColumnVector)
{
    auto subr = default_r(span{0u, 2u}, 1u, 3u);

    EXPECT_EQ(subr(0, 0, 0), 12);
    EXPECT_EQ(subr(1, 0, 0), 28);
}


TEST_F(BlockColMajorAccessor3d, CanAssignValues)
{
    default_r(1, 1, 1) = default_r(0, 0, 0);

    EXPECT_EQ(data[16], 1);
}


TEST_F(BlockColMajorAccessor3d, CanAssignSubranges)
{
    default_r(1u, span{0, 2}, span{0, 3}) =
        custom_r(0u, span{0, 2}, span{0, 3});

    EXPECT_EQ(data[12], 1);
    EXPECT_EQ(data[15], 2);
    EXPECT_EQ(data[18], -1);
    EXPECT_EQ(data[21], 24);
    EXPECT_EQ(data[13], 3);
    EXPECT_EQ(data[16], 4);
    EXPECT_EQ(data[19], -2);
    EXPECT_EQ(data[22], 28);
    EXPECT_EQ(data[14], 29);
    EXPECT_EQ(data[17], 30);
    EXPECT_EQ(data[20], 31);
    EXPECT_EQ(data[23], 32);
}


}  // namespace
