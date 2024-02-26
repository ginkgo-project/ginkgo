// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/distributed/partition_helpers_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename GlobalIndexType>
class PartitionHelpers : public ::testing::Test {
protected:
    using global_index_type = GlobalIndexType;

    PartitionHelpers() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::array<global_index_type> default_range_start_ends{
        this->ref, {0, 4, 4, 7, 7, 9, 9, 11}};
    gko::array<comm_index_type> default_part_ids{this->ref, {0, 1, 2, 3}};
};

TYPED_TEST_SUITE(PartitionHelpers, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(PartitionHelpers, CanSortByRangeStartIdentity)
{
    using itype = typename TestFixture::global_index_type;
    auto range_start_ends = this->default_range_start_ends;
    auto part_ids = this->default_part_ids;

    gko::kernels::reference::partition_helpers::sort_by_range_start(
        this->ref, range_start_ends, part_ids);

    GKO_ASSERT_ARRAY_EQ(range_start_ends, this->default_range_start_ends);
    GKO_ASSERT_ARRAY_EQ(part_ids, this->default_part_ids);
}


TYPED_TEST(PartitionHelpers, CanSortByRangeStart)
{
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<global_index_type> range_start_ends{this->ref,
                                                   {7, 9, 4, 7, 0, 4, 9, 11}};
    gko::array<comm_index_type> result_part_ids{this->ref, {2, 1, 0, 3}};
    auto part_ids = this->default_part_ids;

    gko::kernels::reference::partition_helpers::sort_by_range_start(
        this->ref, range_start_ends, part_ids);

    GKO_ASSERT_ARRAY_EQ(range_start_ends, this->default_range_start_ends);
    GKO_ASSERT_ARRAY_EQ(part_ids, result_part_ids);
}


TYPED_TEST(PartitionHelpers, CanCheckConsecutiveRanges)
{
    using global_index_type = typename TestFixture::global_index_type;
    auto range_start_ends = this->default_range_start_ends;
    bool result = false;

    gko::kernels::reference::partition_helpers::check_consecutive_ranges(
        this->ref, range_start_ends, result);

    ASSERT_TRUE(result);
}


TYPED_TEST(PartitionHelpers, CanCheckNonConsecutiveRanges)
{
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<global_index_type> range_start_ends{this->ref,
                                                   {7, 9, 4, 7, 0, 4, 9, 11}};
    bool result = true;

    gko::kernels::reference::partition_helpers::check_consecutive_ranges(
        this->ref, range_start_ends, result);

    ASSERT_FALSE(result);
}


TYPED_TEST(PartitionHelpers, CanCompressRanges)
{
    using itype = typename TestFixture::global_index_type;
    auto range_start_ends = this->default_range_start_ends;
    gko::array<itype> range_offsets{this->ref,
                                    range_start_ends.get_size() / 2 + 1};
    gko::array<itype> expected_range_offsets{this->ref, {0, 4, 7, 9, 11}};

    gko::kernels::reference::partition_helpers::compress_ranges(
        this->ref, range_start_ends, range_offsets);

    GKO_ASSERT_ARRAY_EQ(range_offsets, expected_range_offsets);
}


}  // namespace
