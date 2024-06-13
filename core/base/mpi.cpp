// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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


bool requires_host_buffer(const std::shared_ptr<const Executor>& exec,
                          const communicator& comm)
{
    return exec != exec->get_master() &&
           (comm.force_host_buffer() || !is_gpu_aware());
}


}  // namespace mpi
}  // namespace experimental
}  // namespace gko


#endif  // GKO_HAVE_MPI
