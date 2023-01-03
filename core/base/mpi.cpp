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

#include <ginkgo/core/base/mpi.hpp>


#if GINKGO_BUILD_MPI


#include <string>


#include <mpi.h>


namespace gko {
namespace experimental {
namespace mpi {


int map_rank_to_device_id(MPI_Comm comm, const int num_devices)
{
    GKO_ASSERT(num_devices > 0);
    if (num_devices == 1) {
        return 0;
    } else {
        auto mpi_node_local_rank = [](MPI_Comm comm_) {
            int local_rank;
            MPI_Comm local_comm;
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split_type(
                comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm));
            GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(local_comm, &local_rank));
            MPI_Comm_free(&local_comm);
            return local_rank;
        };

        // When we are using MPI_COMM_WORLD, there might be already an
        // environment variable describing the node local rank, so we
        // prioritize it. If no suitable environment variable is found
        // we determine the node-local rank with MPI calls.
        int local_rank;
        int compare_result;
        GKO_ASSERT_NO_MPI_ERRORS(
            MPI_Comm_compare(comm, MPI_COMM_WORLD, &compare_result));
        if (compare_result != MPI_IDENT && compare_result != MPI_CONGRUENT) {
            local_rank = mpi_node_local_rank(comm);
        } else {
            if (auto str = std::getenv("MV2_COMM_WORLD_LOCAL_RANK")) {
                local_rank = std::stoi(str);
            } else if (auto str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) {
                local_rank = std::stoi(str);
            } else if (auto str = std::getenv("MPI_LOCALRANKID")) {
                local_rank = std::stoi(str);
            } else if (auto str = std::getenv("SLURM_LOCALID")) {
                local_rank = std::stoi(str);
            } else {
                local_rank = mpi_node_local_rank(comm);
            }
        }
        return local_rank % num_devices;
    }
}


}  // namespace mpi
}  // namespace experimental
}  // namespace gko


#endif  // GKO_HAVE_MPI
