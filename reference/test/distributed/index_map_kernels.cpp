// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/distributed/index_map.hpp>

#include "core/test/utils.hpp"


using comm_index_type = gko::experimental::distributed::comm_index_type;


class IndexMap : public ::testing::Test {
protected:
    using local_index_type = gko::int32;
    using global_index_type = gko::int64;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;

    IndexMap() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    std::shared_ptr<const part_type> part =
        part_type::build_from_mapping(ref, {ref, {0, 0, 1, 1, 2, 2}}, 3);
};


TEST_F(IndexMap, CanBuildMapping)
{
    gko::array<comm_index_type> target_ids(ref);
    gko::array<local_index_type> remote_local_idxs(ref);
    gko::array<global_index_type> remote_global_idxs(ref);
    gko::array<gko::int64> remote_sizes(ref);

    gko::kernels::reference::index_map::build_mapping(
        ref, part.get(), {ref, {2, 3, 3, 5, 5}}, target_ids, remote_local_idxs,
        remote_global_idxs, remote_sizes);

    auto expected_global = gko::array<global_index_type>{ref, {2, 3, 5}};
    auto expected_local = gko::array<local_index_type>{ref, {0, 1, 1}};
    auto expected_ids = gko::array<comm_index_type>{ref, {1, 2}};
    auto expected_sizes = gko::array<gko::int64>{ref, {2, 1}};
    GKO_ASSERT_ARRAY_EQ(remote_sizes, expected_sizes);
    GKO_ASSERT_ARRAY_EQ(target_ids, expected_ids);
    GKO_ASSERT_ARRAY_EQ(remote_global_idxs, expected_global);
    GKO_ASSERT_ARRAY_EQ(remote_local_idxs, expected_local);
}


TEST_F(IndexMap, CanBuildMappingWithoutRecvConnections)
{
    gko::array<comm_index_type> target_ids(ref);
    gko::array<local_index_type> remote_local_idxs(ref);
    gko::array<global_index_type> remote_global_idxs(ref);
    gko::array<gko::int64> remote_sizes(ref);

    gko::kernels::reference::index_map::build_mapping(
        ref, part.get(), {ref, 0}, target_ids, remote_local_idxs,
        remote_global_idxs, remote_sizes);

    auto expected_global = gko::array<global_index_type>{ref, 0};
    auto expected_local = gko::array<local_index_type>{ref, 0};
    auto expected_ids = gko::array<comm_index_type>{ref, 0};
    auto expected_sizes = gko::array<gko::int64>{ref};
    GKO_ASSERT_ARRAY_EQ(remote_sizes, expected_sizes);
    GKO_ASSERT_ARRAY_EQ(target_ids, expected_ids);
    GKO_ASSERT_ARRAY_EQ(remote_global_idxs, expected_global);
    GKO_ASSERT_ARRAY_EQ(remote_local_idxs, expected_local);
}


TEST_F(IndexMap, CanGetLocalWithNonLocalIS)
{
    gko::array<global_index_type> global_ids(ref, {1, 1, 4, 0, 4});
    gko::array<local_index_type> local_ids(ref);
    auto remote_global_idxs =
        gko::segmented_array<global_index_type>::create_from_sizes(
            {ref, {0, 1, 4}}, {ref, {2, 1}});
    gko::array<comm_index_type> remote_target_ids(ref, {0, 2});

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), remote_target_ids, to_device_const(remote_global_idxs),
        1, global_ids, gko::experimental::distributed::index_space::non_local,
        local_ids);

    gko::array<local_index_type> expected(ref, {1, 1, 2, 0, 2});
    GKO_ASSERT_ARRAY_EQ(local_ids, expected);
}


TEST_F(IndexMap, CanGetLocalWithNonLocalISWithInvalid)
{
    gko::array<global_index_type> global_ids(ref, {1, 1, 4, 3, 0, 4});
    gko::array<local_index_type> local_ids(ref);
    auto remote_global_idxs =
        gko::segmented_array<global_index_type>::create_from_sizes(
            {ref, {0, 1, 4}}, {ref, {2, 1}});
    gko::array<comm_index_type> remote_target_ids(ref, {0, 2});

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), remote_target_ids, to_device_const(remote_global_idxs),
        1, global_ids, gko::experimental::distributed::index_space::non_local,
        local_ids);

    gko::array<local_index_type> expected(ref, {1, 1, 2, -1, 0, 2});
    GKO_ASSERT_ARRAY_EQ(local_ids, expected);
}


TEST_F(IndexMap, CanGetLocalWithLocalIS)
{
    gko::array<global_index_type> global_ids(ref, {2, 3, 3, 2});
    gko::array<local_index_type> local_ids(ref);
    auto remote_global_idxs =
        gko::segmented_array<global_index_type>::create_from_sizes(
            {ref, {0, 1, 4}}, {ref, {2, 1}});
    gko::array<comm_index_type> remote_target_ids(ref, {0, 2});

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), remote_target_ids, to_device_const(remote_global_idxs),
        1, global_ids, gko::experimental::distributed::index_space::local,
        local_ids);

    gko::array<local_index_type> expected(ref, {0, 1, 1, 0});
    GKO_ASSERT_ARRAY_EQ(local_ids, expected);
}


TEST_F(IndexMap, CanGetLocalWithLocalISWithInvalid)
{
    gko::array<global_index_type> global_ids(ref, {2, 4, 5, 3, 3, 2});
    gko::array<local_index_type> local_ids(ref);
    auto remote_global_idxs =
        gko::segmented_array<global_index_type>::create_from_sizes(
            {ref, {0, 1, 4}}, {ref, {2, 1}});
    gko::array<comm_index_type> remote_target_ids(ref, {0, 2});

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), remote_target_ids, to_device_const(remote_global_idxs),
        1, global_ids, gko::experimental::distributed::index_space::local,
        local_ids);

    gko::array<local_index_type> expected(ref, {0, -1, -1, 1, 1, 0});
    GKO_ASSERT_ARRAY_EQ(local_ids, expected);
}


TEST_F(IndexMap, CanGetLocalWithCombinedIS)
{
    gko::array<global_index_type> global_ids(ref, {0, 1, 2, 3, 0, 4, 3});
    gko::array<local_index_type> local_ids(ref);
    auto remote_global_idxs =
        gko::segmented_array<global_index_type>::create_from_sizes(
            {ref, {0, 1, 4}}, {ref, {2, 1}});
    gko::array<comm_index_type> remote_target_ids(ref, {0, 2});

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), remote_target_ids, to_device_const(remote_global_idxs),
        1, global_ids, gko::experimental::distributed::index_space::combined,
        local_ids);

    gko::array<local_index_type> expected(ref, {2, 3, 0, 1, 2, 4, 1});
    GKO_ASSERT_ARRAY_EQ(local_ids, expected);
}


TEST_F(IndexMap, CanGetLocalWithCombinedISWithInvalid)
{
    gko::array<global_index_type> global_ids(ref, {0, 1, 2, 3, 0, 4, 5, 3});
    gko::array<local_index_type> local_ids(ref);
    auto remote_global_idxs =
        gko::segmented_array<global_index_type>::create_from_sizes(
            {ref, {0, 1, 4}}, {ref, {2, 1}});
    gko::array<comm_index_type> remote_target_ids(ref, {0, 2});

    gko::kernels::reference::index_map::map_to_local(
        ref, part.get(), remote_target_ids, to_device_const(remote_global_idxs),
        1, global_ids, gko::experimental::distributed::index_space::combined,
        local_ids);

    gko::array<local_index_type> expected(ref, {2, 3, 0, 1, 2, 4, -1, 1});
    GKO_ASSERT_ARRAY_EQ(local_ids, expected);
}
