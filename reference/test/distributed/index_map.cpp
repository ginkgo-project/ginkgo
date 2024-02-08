// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/index_map.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/distributed/index_map_kernels.hpp"
#include "core/test/utils.hpp"

namespace {


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
    gko::collection::array<local_index_type> remote_local_idxs(ref);
    gko::collection::array<global_index_type> remote_global_idxs(ref);

    gko::kernels::reference::index_map::build_mapping(
        ref, part.get(), {ref, {2, 3, 3, 5, 5}}, target_ids, remote_local_idxs,
        remote_global_idxs);

    auto expected_global = gko::array<global_index_type>{ref, {2, 3, 5}};
    auto expected_local = gko::array<local_index_type>{ref, {0, 1, 1}};
    auto expected_ids = gko::array<comm_index_type>{ref, {1, 2}};
    GKO_ASSERT_ARRAY_EQ(target_ids, expected_ids);
    GKO_ASSERT_ARRAY_EQ(remote_global_idxs.get_flat(), expected_global);
    GKO_ASSERT_ARRAY_EQ(remote_local_idxs.get_flat(), expected_local);
}

// TEST_F(LocalizedPartition, CanMapSingleId)
// {
//     auto part = part_type::build_from_blocked_recv(
//         ref, 2,
//         {
//             std::make_pair(gko::array<int>(ref, {0, 1}), 1),
//             std::make_pair(gko::array<int>(ref, {0, 1}), 2),
//         },
//         {ref, {1, 2}}, {2, 1});
//     auto map = map_type(ref, part,
//                         gko::collection::array<gko::int32>{
//                             {ref, {0, 1, 1}}, std::vector<int>{2, 1}});
//
//     auto r0 = map.get_local(1, 0);
//     auto r1 = map.get_local(0, 1);
//     auto r2 = map.get_local(2, 1);
//
//     ASSERT_EQ(r0, 0);
//     ASSERT_EQ(r1, 1);
//     ASSERT_EQ(r2, 2);
// }
//
//
// TEST_F(LocalizedPartition, CanMapSingleArrayOfIds)
// {
//     auto part = part_type::build_from_blocked_recv(
//         ref, 2,
//         {
//             std::make_pair(gko::array<int>(ref, {0, 1}), 1),
//             std::make_pair(gko::array<int>(ref, {0, 1}), 2),
//         },
//         {ref, {1, 2}}, {2, 1});
//     auto map = map_type(ref, part,
//                         gko::collection::array<gko::int32>{
//                             {ref, {0, 1, 1}}, std::vector<int>{2, 1}});
//
//     {
//         auto semi_gid = gko::array<int>{ref, {0, 1, 1, 0}};
//
//         auto result = map.get_local(1, semi_gid);
//
//         auto expected = gko::array<int>{ref, {0, 1, 1, 0}};
//         GKO_ASSERT_ARRAY_EQ(result, expected);
//     }
//     {
//         auto semi_gid = gko::array<int>{ref, {1, 1}};
//
//         auto result = map.get_local(2, semi_gid);
//
//         auto expected = gko::array<int>{ref, {2, 2}};
//         GKO_ASSERT_ARRAY_EQ(result, expected);
//     }
// }
//
//
// TEST_F(LocalizedPartition, CanMapMultipleArrayOfIds)
// {
//     auto part = part_type::build_from_blocked_recv(
//         ref, 2,
//         {
//             std::make_pair(gko::array<int>(ref, {0, 1}), 1),
//             std::make_pair(gko::array<int>(ref, {0, 1}), 2),
//         },
//         {ref, {1, 2}}, {2, 1});
//     auto map = map_type(ref, part,
//                         gko::collection::array<gko::int32>{
//                             {ref, {0, 1, 1}}, std::vector<int>{2, 1}});
//
//     auto result = map.get_local(
//         {ref, {1, 2}}, gko::collection::array<int>({ref, {0, 1, 1, 0, 1, 1}},
//                                                    std::vector<int>{4, 2}));
//
//     auto expected = gko::array<int>{ref, {0, 1, 1, 0, 2, 2}};
//     GKO_ASSERT_ARRAY_EQ(result, expected);
// }


}  // namespace
