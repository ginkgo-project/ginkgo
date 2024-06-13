// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <array>
#include <cstddef>


#include <gtest/gtest.h>


#include "accessor/range.hpp"


namespace {


// 0-memory constant accessor, which "stores" x*i + y*j + k at location
// (i, j, k)
struct dummy_accessor {
    static constexpr std::size_t dimensionality = 3;

    dummy_accessor(std::size_t size, int x, int y)
        : sizes{size, size, size}, x{x}, y{y}
    {}

    dummy_accessor(std::size_t size_x, std::size_t size_y, std::size_t size_z,
                   int x, int y)
        : sizes{size_x, size_y, size_z}, x{x}, y{y}
    {}

    int operator()(int a, int b, int c) const { return x * a + y * b + c; }

    gko::acc::size_type length(std::size_t dim) const { return sizes[dim]; }

    std::array<std::size_t, 3> sizes;
    mutable int x;
    mutable int y;
};


using dummy_range = gko::acc::range<dummy_accessor>;


TEST(Range, CreatesRange)
{
    dummy_range r{5u, 2, 3};

    EXPECT_EQ(r->x, 2);
    ASSERT_EQ(r->y, 3);
}


TEST(Range, ForwardsCallsToAccessor)
{
    dummy_range r{5u, 2, 3};

    EXPECT_EQ(r(1, 2, 3), 2 * 1 + 3 * 2 + 3);
    ASSERT_EQ(r(4, 2, 5), 2 * 4 + 3 * 2 + 5);
}


TEST(Range, ForwardsCopyToAccessor)
{
    dummy_range r{5u, 2, 3};
    r = dummy_range{5u, 2, 5};

    EXPECT_EQ(r->x, 2);
    ASSERT_EQ(r->y, 5);
}


TEST(Range, ForwardsLength)
{
    dummy_range r{5u, 2, 3};

    EXPECT_EQ(r->length(0), 5);
    EXPECT_EQ(r->length(1), 5);
    ASSERT_EQ(r->length(2), 5);
}


}  // namespace
