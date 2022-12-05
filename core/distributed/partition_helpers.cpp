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


namespace gko {
namespace experimental {
namespace distributed {


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_range(std::shared_ptr<const Executor> exec,
                                 LocalIndexType local_start,
                                 LocalIndexType local_end,
                                 mpi::communicator comm)
{
    GlobalIndexType range[2] = {static_cast<GlobalIndexType>(local_start),
                                static_cast<GlobalIndexType>(local_end)};

    // make all range_ends available on each rank
    Array<GlobalIndexType> ranges_start_end(exec->get_master(),
                                            comm.size() * 2);
    ranges_start_end.fill(0);
    comm.all_gather(exec->get_master(), range, 2, ranges_start_end.get_data(),
                    2);

    // remove duplicates
    Array<GlobalIndexType> ranges(exec->get_master(), comm.size() + 1);
    auto ranges_se_data = ranges_start_end.get_const_data();
    ranges.get_data()[0] = ranges_se_data[0];
    for (int i = 1; i < ranges_start_end.get_num_elems() - 1; i += 2) {
        GKO_ASSERT_EQ(ranges_se_data[i], ranges_se_data[i + 1]);
        ranges.get_data()[i / 2 + 1] = ranges_se_data[i];
    }
    ranges.get_data()[ranges.get_num_elems() - 1] =
        ranges_se_data[ranges_start_end.get_num_elems() - 1];

    // move data to correct executor
    ranges.set_executor(exec);

    return Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, ranges);
}

#define GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE(_local_type,      \
                                                     _global_type)     \
    std::unique_ptr<Partition<_local_type, _global_type>>              \
    build_partition_from_local_range(                                  \
        std::shared_ptr<const Executor> exec, _local_type local_start, \
        _local_type local_end, mpi::communicator comm)
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
