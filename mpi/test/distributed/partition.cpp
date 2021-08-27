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

#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"


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


TYPED_TEST(Partition, BuildsFromLocalRanges)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto comm = gko::mpi::communicator::create();
    local_index_type ranges[4][2] = {{0, 10}, {10, 30}, {30, 60}, {60, 100}};
    auto rank = comm->rank();

    auto part =
        gko::distributed::Partition<local_index_type>::build_from_local_range(
            this->ref, ranges[rank][0], ranges[rank][1], comm);

    ASSERT_EQ(part->get_num_ranges(), part->get_num_parts());
    for (int i = 0; i < comm->size(); ++i) {
        ASSERT_EQ(part->get_part_size(i), 10 * (i + 1));
        // ASSERT_EQ(part->get_range_ranks()[i], i);
    }
    ASSERT_EQ(part->get_size(), 100);
}

TYPED_TEST(Partition, ThrowsBuildFromUnsortedLocalRanges)
{
    using local_index_type = typename TestFixture::local_index_type;
    auto comm = gko::mpi::communicator::create();
    local_index_type ranges[4][2] = {{0, 10}, {30, 60}, {10, 30}, {60, 100}};
    auto rank = comm->rank();

    ASSERT_THROW(
        gko::distributed::Partition<local_index_type>::build_from_local_range(
            this->ref, ranges[rank][0], ranges[rank][1], comm),
        gko::ValueMismatch);
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
