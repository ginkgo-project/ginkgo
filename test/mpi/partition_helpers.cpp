// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/partition.hpp>
#include <ginkgo/core/distributed/partition_helpers.hpp>


#include "core/test/utils.hpp"
#include "test/utils/mpi/executor.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename IndexType>
class PartitionHelpers : public CommonMpiTestFixture {
protected:
    using index_type = IndexType;
};

TYPED_TEST_SUITE(PartitionHelpers, gko::test::IndexTypes,
                 TypenameNameGenerator);


TYPED_TEST(PartitionHelpers, CanBuildFromLocalRanges)
{
    using itype = typename TestFixture::index_type;
    gko::span local_range[] = {{0u, 4u}, {4u, 9u}, {9u, 11u}};
    gko::array<itype> expects_ranges{this->exec, {0, 4, 9, 11}};
    gko::array<comm_index_type> expects_pid{this->exec, {0, 1, 2}};

    auto part =
        gko::experimental::distributed::build_partition_from_local_range<
            gko::int32, itype>(this->exec, this->comm,
                               local_range[this->comm.rank()]);

    GKO_ASSERT_ARRAY_EQ(
        expects_ranges,
        gko::make_const_array_view(this->exec, expects_ranges.get_size(),
                                   part->get_range_bounds()));
    GKO_ASSERT_ARRAY_EQ(expects_pid, gko::make_const_array_view(
                                         this->exec, expects_pid.get_size(),
                                         part->get_part_ids()));
}


TYPED_TEST(PartitionHelpers, CanBuildFromLocalRangesUnsorted)
{
    using itype = typename TestFixture::index_type;
    gko::span local_range[] = {{4u, 9u}, {9u, 11u}, {0u, 4u}};
    gko::array<itype> expects_ranges{this->exec, {0, 4, 9, 11}};
    gko::array<comm_index_type> expects_pid{this->exec, {2, 0, 1}};

    auto part =
        gko::experimental::distributed::build_partition_from_local_range<
            gko::int32, itype>(this->exec, this->comm,
                               local_range[this->comm.rank()]);

    GKO_ASSERT_ARRAY_EQ(
        expects_ranges,
        gko::make_const_array_view(this->exec, expects_ranges.get_size(),
                                   part->get_range_bounds()));
    GKO_ASSERT_ARRAY_EQ(expects_pid, gko::make_const_array_view(
                                         this->exec, expects_pid.get_size(),
                                         part->get_part_ids()));
}


TYPED_TEST(PartitionHelpers, CanBuildFromLocalRangesThrowsOnGap)
{
    using itype = typename TestFixture::index_type;
    gko::span local_range[] = {{4u, 6u}, {9u, 11u}, {0u, 4u}};
    // Hack because of multiple template arguments in macro
    auto build_from_local_ranges = [](auto... args) {
        return gko::experimental::distributed::build_partition_from_local_range<
            gko::int32, itype>(args...);
    };

    ASSERT_THROW(build_from_local_ranges(this->exec, this->comm,
                                         local_range[this->comm.rank()]),
                 gko::InvalidStateError);
}


TYPED_TEST(PartitionHelpers, CanBuildFromLocalSize)
{
    using itype = typename TestFixture::index_type;
    gko::size_type local_range[] = {4, 5, 3};
    gko::array<itype> expects_ranges{this->exec, {0, 4, 9, 12}};
    gko::array<comm_index_type> expects_pid{this->exec, {0, 1, 2}};

    auto part = gko::experimental::distributed::build_partition_from_local_size<
        gko::int32, itype>(this->exec, this->comm,
                           local_range[this->comm.rank()]);

    GKO_ASSERT_ARRAY_EQ(
        expects_ranges,
        gko::make_const_array_view(this->exec, expects_ranges.get_size(),
                                   part->get_range_bounds()));
    GKO_ASSERT_ARRAY_EQ(expects_pid, gko::make_const_array_view(
                                         this->exec, expects_pid.get_size(),
                                         part->get_part_ids()));
}
