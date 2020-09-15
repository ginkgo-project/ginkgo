/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <iostream>


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
    //clang-format on
    row_major_int_range r{data, 3u, 2u, 3u};
};


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


TEST_F(RowMajorAccessor, CanWriteDataToConst)
{
    const row_major_int_range cr = r;
    cr(0, 0) = 4;

    EXPECT_EQ(cr(0, 0), 4);
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


class ReducedStorage3d : public ::testing::Test {
protected:
    using span = gko::span;
    using ar_type = double;
    using st_type = double;

    using accessor = gko::accessor::ReducedStorage3d<ar_type, st_type>;
    using const_accessor = gko::accessor::ReducedStorage3d<ar_type, const st_type>;

    using reduced_storage = gko::range<accessor>;
    using const_reduced_storage = gko::range<const_accessor>;

    // clang-format off
    st_type data[8]{
        1.0, 2.1,
        -1.2, 3.3,
        4.4, -2.5,
        5.6, 6.7
    };
    //clang-format on
    reduced_storage r{data, gko::dim<3>{2u, 2u, 2u}};
    const_reduced_storage cr{data, gko::dim<3>{2u, 2u, 2u}};
};


TEST_F(ReducedStorage3d, CanUseConst)
{
    EXPECT_EQ(cr(0, 0, 0), 1.0);
    EXPECT_EQ(cr(0, 0, 1), 2.1);
    EXPECT_EQ(cr(0, 1, 0), -1.2);
    EXPECT_EQ(cr(0, 1, 1), 3.3);
    EXPECT_EQ(cr(1, 0, 0), 4.4);
    EXPECT_EQ(cr(1, 0, 1), -2.5);
    EXPECT_EQ(cr(1, 1, 0), 5.6);
    EXPECT_EQ(cr(1, 1, 1), 6.7);

    r(0, 1, 0) = cr(0, 0, 0);
    EXPECT_EQ(r(0, 1, 0), 1.0);

    auto subr = cr(span{0, 2}, 0, 0);
    //cr(0, 0, 0) = 2.0;

    EXPECT_EQ(subr(0, 0, 0), 1.0);
    EXPECT_EQ(subr(1, 0, 0), 4.4);
}


TEST_F(ReducedStorage3d, CanReadData)
{
    EXPECT_EQ(r(0, 0, 0), 1.0);
    EXPECT_EQ(r(0, 0, 1), 2.1);
    EXPECT_EQ(r(0, 1, 0), -1.2);
    EXPECT_EQ(r(0, 1, 1), 3.3);
    EXPECT_EQ(r(1, 0, 0), 4.4);
    EXPECT_EQ(r(1, 0, 1), -2.5);
    EXPECT_EQ(r(1, 1, 0), 5.6);
    EXPECT_EQ(r(1, 1, 1), 6.7);
}


TEST_F(ReducedStorage3d, CanWriteData)
{
    r(0, 0, 0) = 2.0;

    EXPECT_EQ(r(0, 0, 0), 2.0);
}


TEST_F(ReducedStorage3d, CanCreateSubrange)
{
    auto subr = r(span{1, 2}, span{0, 2}, span{0, 1});

    EXPECT_EQ(subr(0, 0, 0), 4.4);
    EXPECT_EQ(subr(0, 1, 0), 5.6);
}


TEST_F(ReducedStorage3d, CanCreateRowVector)
{
    auto subr = r(0, 0, span{0, 2});

    EXPECT_EQ(subr(0, 0, 0), 1.0);
    EXPECT_EQ(subr(0, 0, 1), 2.1);
}


TEST_F(ReducedStorage3d, CanCreateColumnVector)
{
    auto subr = r(span{0, 2}, 0, 0);

    EXPECT_EQ(subr(0, 0, 0), 1.0);
    EXPECT_EQ(subr(1, 0, 0), 4.4);
}


class ScaledReducedStorage3d : public ::testing::Test {
protected:
    using span = gko::span;
    using ar_type = double;
    using st_type = gko::int32;

    using accessor = gko::accessor::ScaledReducedStorage3d<ar_type, st_type>;
    using const_accessor = gko::accessor::ScaledReducedStorage3d<ar_type, const st_type>;

    using reduced_storage = gko::range<accessor>;
    using const_reduced_storage = gko::range<const_accessor>;

    // clang-format off
    st_type data[8]{
        1, 2,
        -3, 4,
        55, 6,
        -777, 8
    };
    ar_type scale[4]{
        1., 1.,
        1., 1.
    };
    //clang-format on
    reduced_storage r{data, gko::dim<3>{2u, 2u, 2u}, scale};
    const_reduced_storage cr{data, gko::dim<3>{2u, 2u, 2u}, scale};
};


TEST_F(ScaledReducedStorage3d, CanUseConst)
{
    EXPECT_EQ(cr(0, 0, 0), 1.);
    EXPECT_EQ(cr(0, 0, 1), 2.);
    EXPECT_EQ(cr(0, 1, 0), -3.);
    EXPECT_EQ(cr(0, 1, 1), 4.);
    EXPECT_EQ(cr(1, 0, 0), 55.);
    EXPECT_EQ(cr(1, 0, 1), 6.);
    EXPECT_EQ(cr(1, 1, 0), -777.);
    EXPECT_EQ(cr(1, 1, 1), 8.);

    auto subr = cr(span{0, 2}, 0, 0);

    EXPECT_EQ(subr(0, 0, 0), 1.0);
    EXPECT_EQ(subr(1, 0, 0), 55.);

    //cr(0, 0, 0) = 2.0;
    r->set_scale(0, 0, 2.);
    EXPECT_EQ(r(0, 0, 0), 2.);
}


}  // namespace
