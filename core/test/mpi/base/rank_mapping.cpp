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

#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/mpi.hpp>


#include "core/test/utils.hpp"


class MapRankToDevice : public ::testing::Test {
protected:
    MapRankToDevice()
        : comm(MPI_COMM_WORLD),
          rank(gko::experimental::mpi::communicator(comm).rank()),
          size(gko::experimental::mpi::communicator(comm).size()),
          env({{"MV2_COMM_WORLD_LOCAL_RANK", ""},
               {"OMPI_COMM_WORLD_LOCAL_RANK", ""},
               {"MPI_LOCALRANKID", ""},
               {"SLURM_LOCALID", ""}})
    {}

    void SetUp() override
    {
        for (auto& it : env) {
            const auto& env_name = it.first;
            if (auto v = std::getenv(env_name.c_str())) {
                env[env_name] = std::string(v);
            }
            unsetenv(env_name.c_str());
        }
    }

    void TearDown() override
    {
        for (auto& it : env) {
            const auto& env_name = it.first;
            const auto& env_value = it.second;
            setenv(env_name.c_str(), env_value.c_str(), 1);
        }
    }

    MPI_Comm comm;
    int rank;
    int size;
    std::map<std::string, std::string> env;
};


TEST_F(MapRankToDevice, OneDevice)
{
    ASSERT_EQ(gko::experimental::mpi::map_rank_to_device_id(comm, 1), 0);
}


TEST_F(MapRankToDevice, EqualDevicesAndRanks)
{
    auto id = gko::experimental::mpi::map_rank_to_device_id(comm, size);

    ASSERT_EQ(id, rank);
}


TEST_F(MapRankToDevice, LessDevicesThanRanks)
{
    int target_id[] = {0, 1, 2, 0};

    auto id = gko::experimental::mpi::map_rank_to_device_id(comm, 3);

    ASSERT_EQ(id, target_id[rank]);
}


TEST_F(MapRankToDevice, UsesRankFromEnvironment)
{
    int reordered_rank[] = {2, 3, 1, 0};
    for (const auto& it : env) {
        SCOPED_TRACE("Using environment variable " + it.first);
        setenv(it.first.c_str(), std::to_string(reordered_rank[rank]).c_str(),
               1);

        auto id = gko::experimental::mpi::map_rank_to_device_id(comm, size);

        ASSERT_EQ(id, reordered_rank[rank]);
        unsetenv(it.first.c_str());
    }
}


TEST_F(MapRankToDevice, NonCommWorld)
{
    MPI_Comm split;
    MPI_Comm_split(comm, static_cast<int>(rank < 3), rank, &split);
    int target_id[] = {0, 1, 0, 0};

    auto id = gko::experimental::mpi::map_rank_to_device_id(split, 2);

    ASSERT_EQ(id, target_id[rank]);
}
