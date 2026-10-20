// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_MPI_BASE_NODE_LOCAL_RANK_HPP_
#define GKO_CORE_TEST_MPI_BASE_NODE_LOCAL_RANK_HPP_


#include <cstring>
#include <vector>

#include <mpi.h>


/**
 * Returns the rank among the ranks of comm on the same node, determined from
 * the processor names instead of MPI_Comm_split_type.
 */
inline int node_local_rank_from_names(MPI_Comm comm)
{
    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    std::vector<char> names(size * MPI_MAX_PROCESSOR_NAME);
    auto name = [&](int i) {
        return names.data() + i * MPI_MAX_PROCESSOR_NAME;
    };
    int length;
    MPI_Get_processor_name(name(rank), &length);
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, names.data(),
                  MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);
    int local_rank = 0;
    for (int i = 0; i < rank; ++i) {
        if (std::strcmp(name(i), name(rank)) == 0) {
            ++local_rank;
        }
    }
    return local_rank;
}


#endif  // GKO_CORE_TEST_MPI_BASE_NODE_LOCAL_RANK_HPP_
