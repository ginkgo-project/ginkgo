// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/batch_dim.hpp>


#include <memory>


#include <gtest/gtest.h>


TEST(BatchDim, ConstructsCorrectUniformObject)
{
    gko::batch_dim<2> d{4, gko::dim<2>(5)};

    ASSERT_EQ(d.get_num_batch_items(), 4);
    ASSERT_EQ(d.get_common_size(), gko::dim<2>(5));
}


TEST(BatchDim, ConstructsNullObject)
{
    gko::batch_dim<2> d{};

    ASSERT_EQ(d.get_num_batch_items(), 0);
    ASSERT_EQ(d.get_common_size(), gko::dim<2>{});
}


TEST(BatchDim, EqualityReturnsTrueWhenEqual)
{
    ASSERT_TRUE(gko::batch_dim<2>(2, gko::dim<2>{3}) ==
                gko::batch_dim<2>(2, gko::dim<2>{3}));
}


TEST(BatchDim, EqualityReturnsFalseWhenDifferentNumBatches)
{
    ASSERT_FALSE(gko::batch_dim<2>(3, gko::dim<2>{3}) ==
                 gko::batch_dim<2>(2, gko::dim<2>{3}));
}


TEST(BatchDim, EqualityReturnsFalseWhenDifferentBatchSizes)
{
    ASSERT_FALSE(gko::batch_dim<2>(3, gko::dim<2>{3}) ==
                 gko::batch_dim<2>(3, gko::dim<2>{4}));
}


TEST(BatchDim, NotEqualWorks)
{
    ASSERT_TRUE(gko::batch_dim<2>(3, gko::dim<2>{3}) !=
                gko::batch_dim<2>(3, gko::dim<2>{4}));
}


TEST(BatchDim, TransposesBatchDimensions)
{
    ASSERT_EQ(gko::transpose(gko::batch_dim<2>(2, gko::dim<2>{4, 2})),
              gko::batch_dim<2>(2, gko::dim<2>{2, 4}));
}
