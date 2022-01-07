/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/dim.hpp>


namespace {


TEST(BatchDim, ConstructsCorrectUniformObject)
{
    gko::batch_dim<2> d{4, gko::dim<2>(5)};

    ASSERT_EQ(d.stores_equal_sizes(), true);
    ASSERT_EQ(d.get_num_batch_entries(), 4);
    ASSERT_EQ(d.get_batch_sizes()[0], gko::dim<2>(5));
    ASSERT_EQ(d.get_batch_sizes()[3], gko::dim<2>(5));
}


TEST(BatchDim, ConstructsCorrectNonUniformObject)
{
    gko::batch_dim<3> d{std::vector<gko::dim<3>>{gko::dim<3>(1), gko::dim<3>(5),
                                                 gko::dim<3>(2)}};

    ASSERT_EQ(d.stores_equal_sizes(), false);
    ASSERT_EQ(d.get_num_batch_entries(), 3);
    ASSERT_EQ(d.get_batch_sizes()[0], gko::dim<3>(1));
    ASSERT_EQ(d.get_batch_sizes()[2], gko::dim<3>(2));
}


TEST(BatchDim, ConstructsNullObject)
{
    gko::batch_dim<2> d{};

    ASSERT_EQ(d.get_num_batch_entries(), 0);
    ASSERT_EQ(d.get_batch_sizes()[0], gko::dim<2>{});
}


class batch_dim_manager {
public:
    using batch_dim = gko::batch_dim<3>;
    const batch_dim& get_size() const { return size_; }

    static std::unique_ptr<batch_dim_manager> create(const batch_dim& size)
    {
        return std::unique_ptr<batch_dim_manager>{new batch_dim_manager{size}};
    }

private:
    batch_dim_manager(const batch_dim& size) : size_{size} {}
    batch_dim size_;
};


TEST(BatchDim, CopiesProperlyOnHeap)
{
    auto manager =
        batch_dim_manager::create(gko::batch_dim<3>{std::vector<gko::dim<3>>{
            gko::dim<3>(1), gko::dim<3>(5), gko::dim<3>(2)}}

        );

    const gko::batch_dim<3> copy = manager->get_size();

    ASSERT_EQ(copy.stores_equal_sizes(), false);
    ASSERT_EQ(copy.get_num_batch_entries(), 3);
    ASSERT_EQ(copy.get_batch_sizes()[0], gko::dim<3>(1));
    ASSERT_EQ(copy.get_batch_sizes()[2], gko::dim<3>(2));
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


TEST(BatchDim, EqualityReturnsFalseWhenOneBatchSizeIsDifferent)
{
    ASSERT_FALSE(gko::batch_dim<2>(std::vector<gko::dim<2>>{gko::dim<2>{4},
                                                            gko::dim<2>{3}}) ==
                 gko::batch_dim<2>(
                     std::vector<gko::dim<2>>{gko::dim<2>{4}, gko::dim<2>{2}}));
}


TEST(BatchDim, CanDetectUniformBatchesWhenUsingNonUniformBatchConstructor)
{
    auto b1 = gko::batch_dim<2>(std::vector<gko::dim<2>>{
        gko::dim<2>{3}, gko::dim<2>{3}, gko::dim<2>{3}});
    ASSERT_TRUE(b1.stores_equal_sizes());
}


TEST(BatchDim, NotEqualWorks)
{
    ASSERT_TRUE(gko::batch_dim<2>(3, gko::dim<2>{3}) !=
                gko::batch_dim<2>(3, gko::dim<2>{4}));
}


TEST(BatchDim, TransposesNonUniformBatchDimensions)
{
    ASSERT_EQ(gko::transpose(gko::batch_dim<2>(
                  std::vector<gko::dim<2>>{gko::dim<2>{4, 2}, gko::dim<2>{3}})),
              gko::batch_dim<2>(
                  std::vector<gko::dim<2>>{gko::dim<2>{2, 4}, gko::dim<2>{3}}));
}


TEST(BatchDim, TransposesUniformBatchDimensions)
{
    ASSERT_EQ(gko::transpose(gko::batch_dim<2>(3, gko::dim<2>{4, 2})),
              gko::batch_dim<2>(3, gko::dim<2>{2, 4}));
}


}  // namespace
