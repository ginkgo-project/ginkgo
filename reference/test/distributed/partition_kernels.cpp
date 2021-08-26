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

#include <ginkgo/core/distributed/partition.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/distributed/partition_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


using global_index_type = gko::distributed::global_index_type;
using comm_index_type = gko::distributed::comm_index_type;


template <typename LocalIndexType>
class Partition : public ::testing::Test {
protected:
    using local_index_type = LocalIndexType;
    Partition() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Partition, gko::test::IndexTypes);


TYPED_TEST(Partition, BuildsFromMapping)
{
    using local_index_type = typename TestFixture::local_index_type;
    gko::Array<comm_index_type> mapping{
        this->ref, {2, 2, 0, 1, 1, 2, 0, 0, 1, 0, 1, 1, 1, 2, 2, 0}};
    comm_index_type num_parts = 3;
    gko::size_type num_ranges = 10;

    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, mapping, num_parts);

    EXPECT_EQ(partition->get_size(), mapping.get_num_elems());
    EXPECT_EQ(partition->get_num_ranges(), num_ranges);
    EXPECT_EQ(partition->get_num_parts(), num_parts);
    EXPECT_EQ(partition->get_const_range_bounds(),
              partition->get_range_bounds());
    EXPECT_EQ(partition->get_const_part_ids(), partition->get_part_ids());
    EXPECT_EQ(partition->get_const_range_bounds()[0], 0);
    EXPECT_EQ(partition->get_const_range_bounds()[1], 2);
    EXPECT_EQ(partition->get_const_range_bounds()[2], 3);
    EXPECT_EQ(partition->get_const_range_bounds()[3], 5);
    EXPECT_EQ(partition->get_const_range_bounds()[4], 6);
    EXPECT_EQ(partition->get_const_range_bounds()[5], 8);
    EXPECT_EQ(partition->get_const_range_bounds()[6], 9);
    EXPECT_EQ(partition->get_const_range_bounds()[7], 10);
    EXPECT_EQ(partition->get_const_range_bounds()[8], 13);
    EXPECT_EQ(partition->get_const_range_bounds()[9], 15);
    EXPECT_EQ(partition->get_const_range_bounds()[10], 16);
    EXPECT_EQ(partition->get_part_ids()[0], 2);
    EXPECT_EQ(partition->get_part_ids()[1], 0);
    EXPECT_EQ(partition->get_part_ids()[2], 1);
    EXPECT_EQ(partition->get_part_ids()[3], 2);
    EXPECT_EQ(partition->get_part_ids()[4], 0);
    EXPECT_EQ(partition->get_part_ids()[5], 1);
    EXPECT_EQ(partition->get_part_ids()[6], 0);
    EXPECT_EQ(partition->get_part_ids()[7], 1);
    EXPECT_EQ(partition->get_part_ids()[8], 2);
    EXPECT_EQ(partition->get_part_ids()[9], 0);
    EXPECT_EQ(partition->get_range_ranks()[0], 0);
    EXPECT_EQ(partition->get_range_ranks()[1], 0);
    EXPECT_EQ(partition->get_range_ranks()[2], 0);
    EXPECT_EQ(partition->get_range_ranks()[3], 2);
    EXPECT_EQ(partition->get_range_ranks()[4], 1);
    EXPECT_EQ(partition->get_range_ranks()[5], 2);
    EXPECT_EQ(partition->get_range_ranks()[6], 3);
    EXPECT_EQ(partition->get_range_ranks()[7], 3);
    EXPECT_EQ(partition->get_range_ranks()[8], 3);
    EXPECT_EQ(partition->get_range_ranks()[9], 4);
    EXPECT_EQ(partition->get_part_sizes()[0], 5);
    EXPECT_EQ(partition->get_part_sizes()[1], 6);
    EXPECT_EQ(partition->get_part_sizes()[2], 5);
}


TYPED_TEST(Partition, BuildsFromRanges)
{
    using local_index_type = typename TestFixture::local_index_type;
    gko::Array<global_index_type> ranges{this->ref, {0, 5, 5, 7, 9, 10}};

    auto partition =
        gko::distributed::Partition<local_index_type>::build_from_contiguous(
            this->ref, ranges);

    EXPECT_EQ(partition->get_size(),
              ranges.get_const_data()[ranges.get_num_elems() - 1]);
    EXPECT_EQ(partition->get_num_ranges(), ranges.get_num_elems() - 1);
    EXPECT_EQ(partition->get_num_parts(), ranges.get_num_elems() - 1);
    EXPECT_EQ(partition->get_const_range_bounds(),
              partition->get_range_bounds());
    EXPECT_EQ(partition->get_const_part_ids(), partition->get_part_ids());
    EXPECT_EQ(partition->get_const_range_bounds()[0], 0);
    EXPECT_EQ(partition->get_const_range_bounds()[1], 5);
    EXPECT_EQ(partition->get_const_range_bounds()[2], 5);
    EXPECT_EQ(partition->get_const_range_bounds()[3], 7);
    EXPECT_EQ(partition->get_const_range_bounds()[4], 9);
    EXPECT_EQ(partition->get_const_range_bounds()[5], 10);
    EXPECT_EQ(partition->get_part_ids()[0], 0);
    EXPECT_EQ(partition->get_part_ids()[1], 1);
    EXPECT_EQ(partition->get_part_ids()[2], 2);
    EXPECT_EQ(partition->get_part_ids()[3], 3);
    EXPECT_EQ(partition->get_part_ids()[4], 4);
    EXPECT_EQ(partition->get_range_ranks()[0], 0);
    EXPECT_EQ(partition->get_range_ranks()[1], 0);
    EXPECT_EQ(partition->get_range_ranks()[2], 0);
    EXPECT_EQ(partition->get_range_ranks()[3], 0);
    EXPECT_EQ(partition->get_range_ranks()[4], 0);
    EXPECT_EQ(partition->get_part_sizes()[0], 5);
    EXPECT_EQ(partition->get_part_sizes()[1], 0);
    EXPECT_EQ(partition->get_part_sizes()[2], 2);
    EXPECT_EQ(partition->get_part_sizes()[3], 2);
    EXPECT_EQ(partition->get_part_sizes()[4], 1);
}


TYPED_TEST(Partition, IsConnected)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {0, 0, 1, 1, 2}},
            3));

    ASSERT_TRUE(gko::distributed::is_connected(part.get()));
}


TYPED_TEST(Partition, IsConnectedUnordered)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {1, 1, 0, 0, 2}},
            3));

    ASSERT_TRUE(gko::distributed::is_connected(part.get()));
}


TYPED_TEST(Partition, IsConnectedFail)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {0, 1, 2, 0, 1}},
            3));

    ASSERT_FALSE(gko::distributed::is_connected(part.get()));
}


TYPED_TEST(Partition, IsOrdered)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {1, 1, 0, 0, 2}},
            3));

    ASSERT_FALSE(gko::distributed::is_ordered(part.get()));
}


TYPED_TEST(Partition, IsOrderedFail)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {0, 1, 1, 2, 2}},
            3));

    ASSERT_TRUE(gko::distributed::is_ordered(part.get()));
}


TYPED_TEST(Partition, BuildsRowPermuteIdentity)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {0, 0, 1, 1, 2}},
            3));
    gko::Array<local_index_type> permute{this->ref, part->get_size()};
    gko::Array<local_index_type> result{this->ref, {0, 1, 2, 3, 4}};

    gko::kernels::reference::partition::build_block_gathered_permute(
        this->ref, part.get(), permute);

    GKO_ASSERT_ARRAY_EQ(result, permute);
}

TYPED_TEST(Partition, BuildsRowPermuteReversed)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {2, 2, 1, 1, 0}},
            3));
    gko::Array<local_index_type> permute{this->ref, part->get_size()};
    gko::Array<local_index_type> result{this->ref, {3, 4, 1, 2, 0}};

    gko::kernels::reference::partition::build_block_gathered_permute(
        this->ref, part.get(), permute);

    GKO_ASSERT_ARRAY_EQ(result, permute);
}

TYPED_TEST(Partition, BuildsRowPermuteScattered)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto part = gko::share(
        gko::distributed::Partition<local_index_type>::build_from_mapping(
            this->ref, gko::Array<comm_index_type>{this->ref, {0, 1, 2, 0, 1}},
            3));
    gko::Array<local_index_type> permute{this->ref, part->get_size()};
    gko::Array<local_index_type> result{this->ref, {0, 2, 4, 1, 3}};

    gko::kernels::reference::partition::build_block_gathered_permute(
        this->ref, part.get(), permute);

    GKO_ASSERT_ARRAY_EQ(result, permute);
}


}  // namespace
