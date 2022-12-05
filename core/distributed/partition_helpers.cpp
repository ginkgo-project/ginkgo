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


#include "core/distributed/partition_helpers_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace partition_helpers {
namespace {


GKO_REGISTER_OPERATION(compress_start_ends,
                       partition_helpers::compress_start_ends);


}
}  // namespace partition_helpers


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
build_partition_from_local_range(std::shared_ptr<const Executor> exec,
                                 span local_range, mpi::communicator comm)
{
    GlobalIndexType range[2] = {static_cast<GlobalIndexType>(local_range.begin),
                                static_cast<GlobalIndexType>(local_range.end)};

    // make all range_start_ends available on each rank
    auto mpi_exec = (exec == exec->get_master() || mpi::is_gpu_aware())
                        ? exec
                        : exec->get_master();
    array<GlobalIndexType> ranges_start_end(mpi_exec, comm.size() * 2);
    ranges_start_end.fill(0);
    comm.all_gather(mpi_exec, range, 2, ranges_start_end.get_data(), 2);
    ranges_start_end.set_executor(exec);

    // remove duplicates
    array<GlobalIndexType> ranges(exec, comm.size() + 1);
    exec->run(
        partition_helpers::make_compress_start_ends(ranges_start_end, ranges));

    return Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
        exec, ranges);
}

#define GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE(_local_type,          \
                                                     _global_type)         \
    std::unique_ptr<Partition<_local_type, _global_type>>                  \
    build_partition_from_local_range(std::shared_ptr<const Executor> exec, \
                                     span local_range, mpi::communicator comm)
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_BUILD_PARTITION_FROM_LOCAL_RANGE);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
