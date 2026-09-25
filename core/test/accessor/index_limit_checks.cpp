// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "accessor/index_limit_checks.hpp"

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include "accessor/block_col_major.hpp"
#include "accessor/row_major.hpp"


namespace {


TEST(IntegerRangeChecks, SumChecksLimitsWithoutOverflow)
{
    const auto max = std::numeric_limits<std::int32_t>::max();
    const auto too_large = std::uint64_t{max} + 1;
    const auto wide_max = std::numeric_limits<std::uint64_t>::max();

    EXPECT_TRUE(gko::acc::sum_fits<std::int32_t>(max - 1, 1));
    EXPECT_FALSE(gko::acc::sum_fits<std::int32_t>(max, 1));
    EXPECT_FALSE(gko::acc::sum_fits<std::int32_t>(too_large, 0));
    EXPECT_FALSE(gko::acc::sum_fits<std::int32_t>(0, too_large));
    EXPECT_TRUE(gko::acc::sum_fits<std::uint64_t>(wide_max, 0));
    EXPECT_FALSE(gko::acc::sum_fits<std::uint64_t>(wide_max, 1));
}


TEST(IntegerRangeChecks, ProductHandlesBoundaryAndZero)
{
    const auto wide_max = std::numeric_limits<std::uint64_t>::max();

    EXPECT_TRUE(gko::acc::product_fits<std::int32_t>(32767, 65536));
    EXPECT_FALSE(gko::acc::product_fits<std::int32_t>(32768, 65536));
    EXPECT_TRUE(gko::acc::product_fits<std::int32_t>(0, wide_max));
    EXPECT_TRUE(gko::acc::product_fits<std::int32_t>(wide_max, 0));
    EXPECT_FALSE(gko::acc::product_fits<std::uint64_t>(wide_max, 2));
}


TEST(IntegerRangeChecks, DenseAccessExcludesTrailingPadding)
{
    const auto max = std::numeric_limits<std::int32_t>::max();

    EXPECT_TRUE(gko::acc::dense_access_fits<std::int32_t>(2, 2, max - 1));
    EXPECT_FALSE(gko::acc::dense_access_fits<std::int32_t>(2, 3, max - 1));
}


TEST(IntegerRangeChecks, AccessChecksHandleEmptyViewsAndOverflow)
{
    const auto max = std::numeric_limits<std::uint64_t>::max();

    EXPECT_TRUE(gko::acc::dense_access_fits<std::int32_t>(0, max, max));
    EXPECT_TRUE(gko::acc::dense_access_fits<std::int32_t>(max, 0, max));
    EXPECT_FALSE(gko::acc::dense_access_fits<std::uint64_t>(3, 1, max));
    EXPECT_FALSE(gko::acc::dense_access_fits<std::uint64_t>(2, 2, max));
    EXPECT_TRUE(gko::acc::block_access_fits<std::int32_t>(0, max));
    EXPECT_TRUE(gko::acc::block_access_fits<std::int32_t>(max, 0));
    EXPECT_FALSE(gko::acc::block_access_fits<std::uint64_t>(1, max));
}


TEST(IntegerRangeChecks, DenseAccessorUsesSeparateSizeAndIndexTypes)
{
    using wide_index =
        gko::acc::row_major<double, 2, std::int64_t, std::int32_t>;
    using wide_size =
        gko::acc::row_major<double, 2, std::int32_t, std::int64_t>;

    EXPECT_TRUE(gko::acc::dense_accessor_fits<wide_index>(50000, 50000, 50000));
    EXPECT_FALSE(gko::acc::dense_accessor_fits<wide_size>(50000, 50000, 50000));
}


TEST(IntegerRangeChecks, BlockAccessorChecksFinalOffset)
{
    using accessor = gko::acc::block_col_major<double, 3, std::int32_t>;
    const auto blocks = std::uint64_t{1} << 29;

    EXPECT_TRUE(gko::acc::block_accessor_fits<accessor>(blocks, 2));
    EXPECT_FALSE(gko::acc::block_accessor_fits<accessor>(blocks + 1, 2));
}


}  // namespace
