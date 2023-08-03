/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


TEST(BatchDim, CanGetCumulativeOffsets)
{
    auto d = gko::batch_dim<2>(3, gko::dim<2>(4, 2));

    ASSERT_EQ(d.get_cumulative_offset(0), 0);
    ASSERT_EQ(d.get_cumulative_offset(1), 8);
    ASSERT_EQ(d.get_cumulative_offset(2), 16);
}


TEST(BatchDim, TransposesBatchDimensions)
{
    ASSERT_EQ(gko::transpose(gko::batch_dim<2>(2, gko::dim<2>{4, 2})),
              gko::batch_dim<2>(2, gko::dim<2>{2, 4}));
}
