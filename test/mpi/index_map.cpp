// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/index_map.hpp>
#include <ginkgo/core/distributed/partition.hpp>


#include "core/test/utils.hpp"
#include "test/utils/mpi/executor.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


class IndexMap : public CommonMpiTestFixture {
protected:
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;


    std::shared_ptr<const part_type> part =
        part_type::build_from_mapping(exec, {exec, {0, 0, 1, 1, 2, 2}}, 3);
    std::array<gko::array<global_index_type>, 3> remote_idxs = {
        {{exec, {2, 3, 5, 3, 5}}, {exec, {0, 0, 1, 4}}, {exec, {3, 0, 1}}}};
};


TEST_F(IndexMap, CanBuildMapping)
{
    auto rank = comm.rank();

    auto imap = map_type(exec, comm, part, remote_idxs[rank]);

    std::array expected_global = {
        gko::array<global_index_type>{exec, {2, 3, 5}},
        gko::array<global_index_type>{exec, {0, 1, 4}},
        gko::array<global_index_type>{exec, {0, 1, 3}}};
    std::array expected_local = {gko::array<local_index_type>{exec, {0, 1, 1}},
                                 gko::array<local_index_type>{exec, {0, 1, 0}},
                                 gko::array<local_index_type>{exec, {0, 1, 1}}};
    std::array expected_ids = {gko::array<comm_index_type>{exec, {1, 2}},
                               gko::array<comm_index_type>{exec, {0, 2}},
                               gko::array<comm_index_type>{exec, {0, 1}}};
    std::array expected_shared_local = {
        gko::array<local_index_type>{exec, {0, 1, 0, 1}},
        gko::array<local_index_type>{exec, {0, 1, 1}},
        gko::array<local_index_type>{exec, {1, 0}},
    };
    GKO_ASSERT_ARRAY_EQ(imap.get_recv_target_ids(), expected_ids[rank]);
    GKO_ASSERT_ARRAY_EQ(imap.get_send_target_ids(), expected_ids[rank]);
    GKO_ASSERT_ARRAY_EQ(imap.get_remote_global_idxs().get_flat(),
                        expected_global[rank]);
    GKO_ASSERT_ARRAY_EQ(imap.get_remote_local_idxs().get_flat(),
                        expected_local[rank]);
    GKO_ASSERT_ARRAY_EQ(imap.get_local_shared_idxs().get_flat(),
                        expected_shared_local[rank]);
}


TEST_F(IndexMap, CanGetLocal)
{
    auto rank = comm.rank();
    auto imap = map_type(exec, comm, part, remote_idxs[rank]);
    std::array query = {
        gko::array<global_index_type>{exec, {0, 1, 0}},
        gko::array<global_index_type>{exec, {2, 3, 2}},
        gko::array<global_index_type>{exec, {5, 5, 4, 4}},
    };

    auto result = imap.get_local(
        query[rank], gko::experimental::distributed::index_space::local);

    std::array expected = {gko::array<local_index_type>{exec, {0, 1, 0}},
                           gko::array<local_index_type>{exec, {0, 1, 0}},
                           gko::array<local_index_type>{exec, {1, 1, 0, 0}}};
    GKO_ASSERT_ARRAY_EQ(result, expected[rank]);
}


TEST_F(IndexMap, CanGetLocalWithInvalidIndex)
{
    auto rank = comm.rank();
    auto imap = map_type(exec, comm, part, remote_idxs[rank]);
    std::array query = {
        gko::array<global_index_type>{exec, {0, 1, 0, 2, 3}},
        gko::array<global_index_type>{exec, {0, 1, 2, 3, 2}},
        gko::array<global_index_type>{exec, {5, 5, 1, 2, 4, 4}},
    };

    auto result = imap.get_local(
        query[rank], gko::experimental::distributed::index_space::local);

    std::array expected = {
        gko::array<local_index_type>{exec, {0, 1, 0, -1, -1}},
        gko::array<local_index_type>{exec, {-1, -1, 0, 1, 0}},
        gko::array<local_index_type>{exec, {1, 1, -1, -1, 0, 0}}};
    GKO_ASSERT_ARRAY_EQ(result, expected[rank]);
}


TEST_F(IndexMap, CanGetNonLocal)
{
    auto rank = comm.rank();
    auto imap = map_type(exec, comm, part, remote_idxs[rank]);
    std::array query = {
        gko::array<global_index_type>{exec, {3, 3, 2, 5, 3}},
        gko::array<global_index_type>{exec, {0, 1, 0, 1, 4}},
        gko::array<global_index_type>{exec, {3, 0, 3, 1, 1}},
    };

    auto result = imap.get_local(
        query[rank], gko::experimental::distributed::index_space::non_local);

    std::array expected = {gko::array<local_index_type>{exec, {1, 1, 0, 2, 1}},
                           gko::array<local_index_type>{exec, {0, 1, 0, 1, 2}},
                           gko::array<local_index_type>{exec, {2, 0, 2, 1, 1}}};
    GKO_ASSERT_ARRAY_EQ(result, expected[rank]);
}


TEST_F(IndexMap, CanGetNonLocalWithInvalidIndex)
{
    auto rank = comm.rank();
    auto imap = map_type(exec, comm, part, remote_idxs[rank]);
    std::array query = {
        gko::array<global_index_type>{exec, {0, 1, 3, 3, 2, 5, 3}},
        gko::array<global_index_type>{exec, {0, 1, 0, 1, 4, 2, 5}},
        gko::array<global_index_type>{exec, {3, 0, 3, 2, 2, 1, 1}},
    };

    auto result = imap.get_local(
        query[rank], gko::experimental::distributed::index_space::non_local);

    std::array expected = {
        gko::array<local_index_type>{exec, {-1, -1, 1, 1, 0, 2, 1}},
        gko::array<local_index_type>{exec, {0, 1, 0, 1, 2, -1, -1}},
        gko::array<local_index_type>{exec, {2, 0, 2, -1, -1, 1, 1}}};
    GKO_ASSERT_ARRAY_EQ(result, expected[rank]);
}


TEST_F(IndexMap, CanGetCombined)
{
    auto rank = comm.rank();
    auto imap = map_type(exec, comm, part, remote_idxs[rank]);
    std::array query = {
        gko::array<global_index_type>{exec, {3, 3, 1, 2, 5, 0, 3}},
        gko::array<global_index_type>{exec, {0, 1, 0, 1, 4, 2, 2}},
        gko::array<global_index_type>{exec, {5, 4, 3, 0, 3, 1, 1}},
    };

    auto result = imap.get_local(
        query[rank], gko::experimental::distributed::index_space::combined);

    std::array expected = {
        gko::array<local_index_type>{exec, {3, 3, 1, 2, 4, 0, 3}},
        gko::array<local_index_type>{exec, {2, 3, 2, 3, 4, 0, 0}},
        gko::array<local_index_type>{exec, {1, 0, 4, 2, 4, 3, 3}}};
    GKO_ASSERT_ARRAY_EQ(result, expected[rank]);
}


TEST_F(IndexMap, CanGetCombinedWithInvalidIndex)
{
    auto rank = comm.rank();
    auto imap = map_type(exec, comm, part, remote_idxs[rank]);
    std::array query = {
        gko::array<global_index_type>{exec, {3, 3, 1, 2, 5, 0, 3, 4}},
        gko::array<global_index_type>{exec, {5, 0, 1, 0, 1, 4, 2, 2}},
        gko::array<global_index_type>{exec, {5, 4, 3, 2, 0, 3, 1, 1}},
    };

    auto result = imap.get_local(
        query[rank], gko::experimental::distributed::index_space::combined);

    std::array expected = {
        gko::array<local_index_type>{exec, {3, 3, 1, 2, 4, 0, 3, -1}},
        gko::array<local_index_type>{exec, {-1, 2, 3, 2, 3, 4, 0, 0}},
        gko::array<local_index_type>{exec, {1, 0, 4, -1, 2, 4, 3, 3}}};
    GKO_ASSERT_ARRAY_EQ(result, expected[rank]);
}
