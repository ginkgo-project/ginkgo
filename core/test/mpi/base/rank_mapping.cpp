// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/mpi.hpp>

#include "core/test/utils.hpp"


class MapRankToDevice : public ::testing::Test {
protected:
    MapRankToDevice()
        : comm(MPI_COMM_WORLD),
          rank(comm.rank()),
          size(comm.size()),
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

    gko::experimental::mpi::communicator comm;
    int rank;
    int size;
    std::map<std::string, std::string> env;
};


TEST_F(MapRankToDevice, OneDevice)
{
    ASSERT_EQ(gko::experimental::mpi::map_rank_to_device_id(comm.get(), 1), 0);
}


TEST_F(MapRankToDevice, EqualDevicesAndRanks)
{
    auto id = gko::experimental::mpi::map_rank_to_device_id(comm.get(), size);

    ASSERT_EQ(id, comm.node_local_rank());
}


TEST_F(MapRankToDevice, LessDevicesThanRanks)
{
    auto id = gko::experimental::mpi::map_rank_to_device_id(comm.get(), 3);

    ASSERT_EQ(id, comm.node_local_rank() % 3);
}


TEST_F(MapRankToDevice, UsesRankFromEnvironment)
{
    int reordered_rank[] = {2, 3, 1, 0};
    for (const auto& it : env) {
        SCOPED_TRACE("Using environment variable " + it.first);
        setenv(it.first.c_str(), std::to_string(reordered_rank[rank]).c_str(),
               1);

        auto id =
            gko::experimental::mpi::map_rank_to_device_id(comm.get(), size);

        ASSERT_EQ(id, reordered_rank[rank]);
        unsetenv(it.first.c_str());
    }
}


TEST_F(MapRankToDevice, NonCommWorld)
{
    MPI_Comm split;
    MPI_Comm_split(comm.get(), static_cast<int>(rank < 3), rank, &split);
    auto id = gko::experimental::mpi::map_rank_to_device_id(split, 2);

    ASSERT_EQ(
        id, gko::experimental::mpi::communicator(split).node_local_rank() % 2);
}
