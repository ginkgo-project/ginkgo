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

TYPED_TEST_SUITE(PartitionHelpers, gko::test::IndexTypes);


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
        gko::make_const_array_view(this->exec, expects_ranges.get_num_elems(),
                                   part->get_range_bounds()));
    GKO_ASSERT_ARRAY_EQ(
        expects_pid,
        gko::make_const_array_view(this->exec, expects_pid.get_num_elems(),
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
        gko::make_const_array_view(this->exec, expects_ranges.get_num_elems(),
                                   part->get_range_bounds()));
    GKO_ASSERT_ARRAY_EQ(
        expects_pid,
        gko::make_const_array_view(this->exec, expects_pid.get_num_elems(),
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
                 gko::Error);
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
        gko::make_const_array_view(this->exec, expects_ranges.get_num_elems(),
                                   part->get_range_bounds()));
    GKO_ASSERT_ARRAY_EQ(
        expects_pid,
        gko::make_const_array_view(this->exec, expects_pid.get_num_elems(),
                                   part->get_part_ids()));
}
