// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/partition.hpp>


#include <algorithm>
#include <memory>
#include <vector>


#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/distributed/localized_partition.hpp>


#include "core/distributed/index_map_kernels.hpp"
#include "core/distributed/partition_kernels.hpp"
#include "core/test/utils.hpp"

namespace {


using comm_index_type = gko::experimental::distributed::comm_index_type;


template <typename T, typename U>
void assert_equal_data(const T* data, std::initializer_list<U> reference_data)
{
    std::vector<U> ref(std::move(reference_data));
    for (auto i = 0; i < ref.size(); ++i) {
        EXPECT_EQ(data[i], ref[i]);
    }
}


template <typename LocalGlobalIndexType>
class Partition : public ::testing::Test {
protected:
    using local_index_type =
        typename std::tuple_element<0, decltype(LocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<1, decltype(LocalGlobalIndexType())>::type;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;

    Partition() : ref(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(Partition, gko::test::LocalGlobalIndexTypes,
                 PairTypenameNameGenerator);


TYPED_TEST(Partition, BuildsFromMapping)
{
    using part_type = typename TestFixture::part_type;
    gko::array<comm_index_type> mapping{
        this->ref, {2, 2, 0, 1, 1, 2, 0, 0, 1, 0, 1, 1, 1, 2, 2, 0}};
    comm_index_type num_parts = 3;
    gko::size_type num_ranges = 10;

    auto partition =
        part_type::build_from_mapping(this->ref, mapping, num_parts);

    EXPECT_EQ(partition->get_size(), mapping.get_size());
    EXPECT_EQ(partition->get_num_ranges(), num_ranges);
    EXPECT_EQ(partition->get_num_parts(), num_parts);
    EXPECT_EQ(partition->get_num_empty_parts(), 0);
    assert_equal_data(partition->get_range_bounds(),
                      {0, 2, 3, 5, 6, 8, 9, 10, 13, 15, 16});
    assert_equal_data(partition->get_part_ids(),
                      {2, 0, 1, 2, 0, 1, 0, 1, 2, 0});
    assert_equal_data(partition->get_range_starting_indices(),
                      {0, 0, 0, 2, 1, 2, 3, 3, 3, 4});
    assert_equal_data(partition->get_part_sizes(), {5, 6, 5});
}


TYPED_TEST(Partition, BuildsFromMappingWithEmptyParts)
{
    using part_type = typename TestFixture::part_type;
    gko::array<comm_index_type> mapping{
        this->ref, {3, 3, 0, 1, 1, 3, 0, 0, 1, 0, 1, 1, 1, 3, 3, 0}};
    comm_index_type num_parts = 5;
    gko::size_type num_ranges = 10;

    auto partition =
        part_type::build_from_mapping(this->ref, mapping, num_parts);

    EXPECT_EQ(partition->get_size(), mapping.get_size());
    EXPECT_EQ(partition->get_num_ranges(), num_ranges);
    EXPECT_EQ(partition->get_num_parts(), num_parts);
    EXPECT_EQ(partition->get_num_empty_parts(), 2);
    assert_equal_data(partition->get_range_bounds(),
                      {0, 2, 3, 5, 6, 8, 9, 10, 13, 15, 16});
    assert_equal_data(partition->get_part_ids(),
                      {3, 0, 1, 3, 0, 1, 0, 1, 3, 0});
    assert_equal_data(partition->get_range_starting_indices(),
                      {0, 0, 0, 2, 1, 2, 3, 3, 3, 4});
    assert_equal_data(partition->get_part_sizes(), {5, 6, 0, 5, 0});
}


TYPED_TEST(Partition, BuildsFromRanges)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref, {0, 5, 5, 7, 9, 10}};

    auto partition = part_type::build_from_contiguous(this->ref, ranges);

    EXPECT_EQ(partition->get_size(), ranges.get_data()[ranges.get_size() - 1]);
    EXPECT_EQ(partition->get_num_ranges(), ranges.get_size() - 1);
    EXPECT_EQ(partition->get_num_parts(), ranges.get_size() - 1);
    EXPECT_EQ(partition->get_num_empty_parts(), 1);
    assert_equal_data(partition->get_range_bounds(), {0, 5, 5, 7, 9, 10});
    assert_equal_data(partition->get_part_ids(), {0, 1, 2, 3, 4});
    assert_equal_data(partition->get_range_starting_indices(), {0, 0, 0, 0, 0});
    assert_equal_data(partition->get_part_sizes(), {5, 0, 2, 2, 1});
}


TYPED_TEST(Partition, BuildsFromRangeWithSingleElement)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref, {0}};

    auto partition = part_type::build_from_contiguous(this->ref, ranges);

    EXPECT_EQ(partition->get_size(), 0);
    EXPECT_EQ(partition->get_num_ranges(), 0);
    EXPECT_EQ(partition->get_num_parts(), 0);
    EXPECT_EQ(partition->get_num_empty_parts(), 0);
    assert_equal_data(partition->get_range_bounds(), {0});
}


TYPED_TEST(Partition, BuildsFromRangesWithPartIds)
{
    using global_index_type = typename TestFixture::global_index_type;
    using part_type = typename TestFixture::part_type;
    gko::array<global_index_type> ranges{this->ref, {0, 5, 5, 7, 9, 10}};
    gko::array<comm_index_type> part_id{this->ref, {0, 4, 3, 1, 2}};

    auto partition =
        part_type::build_from_contiguous(this->ref, ranges, part_id);

    EXPECT_EQ(partition->get_size(), ranges.get_data()[ranges.get_size() - 1]);
    EXPECT_EQ(partition->get_num_ranges(), ranges.get_size() - 1);
    EXPECT_EQ(partition->get_num_parts(), ranges.get_size() - 1);
    EXPECT_EQ(partition->get_num_empty_parts(), 1);
    assert_equal_data(partition->get_range_bounds(), {0, 5, 5, 7, 9, 10});
    assert_equal_data(partition->get_part_ids(), {0, 4, 3, 1, 2});
    assert_equal_data(partition->get_range_starting_indices(), {0, 0, 0, 0, 0});
    assert_equal_data(partition->get_part_sizes(), {5, 2, 1, 2, 0});
}


TYPED_TEST(Partition, BuildsFromGlobalSize)
{
    using part_type = typename TestFixture::part_type;

    auto partition =
        part_type::build_from_global_size_uniform(this->ref, 5, 13);

    EXPECT_EQ(partition->get_size(), 13);
    EXPECT_EQ(partition->get_num_ranges(), 5);
    EXPECT_EQ(partition->get_num_parts(), 5);
    EXPECT_EQ(partition->get_num_empty_parts(), 0);
    assert_equal_data(partition->get_range_bounds(), {0, 3, 6, 9, 11, 13});
    assert_equal_data(partition->get_part_ids(), {0, 1, 2, 3, 4});
    assert_equal_data(partition->get_range_starting_indices(), {0, 0, 0, 0, 0});
    assert_equal_data(partition->get_part_sizes(), {3, 3, 3, 2, 2});
}


TYPED_TEST(Partition, BuildsFromGlobalSizeEmptySize)
{
    using part_type = typename TestFixture::part_type;

    auto partition = part_type::build_from_global_size_uniform(this->ref, 5, 0);

    EXPECT_EQ(partition->get_size(), 0);
    EXPECT_EQ(partition->get_num_ranges(), 5);
    EXPECT_EQ(partition->get_num_parts(), 5);
    EXPECT_EQ(partition->get_num_empty_parts(), 5);
    assert_equal_data(partition->get_range_bounds(), {0, 0, 0, 0, 0, 0});
    assert_equal_data(partition->get_part_ids(), {0, 1, 2, 3, 4});
    assert_equal_data(partition->get_range_starting_indices(), {0, 0, 0, 0, 0});
    assert_equal_data(partition->get_part_sizes(), {0, 0, 0, 0, 0});
}


TYPED_TEST(Partition, BuildsFromGlobalSizeWithEmptyParts)
{
    using part_type = typename TestFixture::part_type;

    auto partition = part_type::build_from_global_size_uniform(this->ref, 5, 3);

    EXPECT_EQ(partition->get_size(), 3);
    EXPECT_EQ(partition->get_num_ranges(), 5);
    EXPECT_EQ(partition->get_num_parts(), 5);
    EXPECT_EQ(partition->get_num_empty_parts(), 2);
    assert_equal_data(partition->get_range_bounds(), {0, 1, 2, 3, 3, 3});
    assert_equal_data(partition->get_part_ids(), {0, 1, 2, 3, 4});
    assert_equal_data(partition->get_range_starting_indices(), {0, 0, 0, 0, 0});
    assert_equal_data(partition->get_part_sizes(), {1, 1, 1, 0, 0});
}


TYPED_TEST(Partition, IsConnected)
{
    using part_type = typename TestFixture::part_type;
    auto part = part_type::build_from_mapping(
        this->ref, gko::array<comm_index_type>{this->ref, {0, 0, 1, 1, 2}}, 3);

    ASSERT_TRUE(part->has_connected_parts());
}


TYPED_TEST(Partition, IsConnectedWithEmptyParts)
{
    using part_type = typename TestFixture::part_type;
    auto part = part_type::build_from_mapping(
        this->ref, gko::array<comm_index_type>{this->ref, {0, 0, 2, 2, 5}}, 6);

    ASSERT_TRUE(part->has_connected_parts());
}


TYPED_TEST(Partition, IsConnectedUnordered)
{
    using part_type = typename TestFixture::part_type;
    auto part = part_type::build_from_mapping(
        this->ref, gko::array<comm_index_type>{this->ref, {1, 1, 0, 0, 2}}, 3);

    ASSERT_TRUE(part->has_connected_parts());
    ASSERT_FALSE(part->has_ordered_parts());
}


TYPED_TEST(Partition, IsConnectedFail)
{
    using part_type = typename TestFixture::part_type;
    auto part = part_type::build_from_mapping(
        this->ref, gko::array<comm_index_type>{this->ref, {0, 1, 2, 0, 1}}, 3);

    ASSERT_FALSE(part->has_connected_parts());
}


TYPED_TEST(Partition, IsOrdered)
{
    using part_type = typename TestFixture::part_type;
    auto part = part_type::build_from_mapping(
        this->ref, gko::array<comm_index_type>{this->ref, {0, 1, 1, 2, 2}}, 3);

    ASSERT_TRUE(part->has_ordered_parts());
}


TYPED_TEST(Partition, IsOrderedWithEmptyParts)
{
    using part_type = typename TestFixture::part_type;
    auto part = part_type::build_from_mapping(
        this->ref, gko::array<comm_index_type>{this->ref, {0, 2, 2, 5, 5}}, 6);

    ASSERT_TRUE(part->has_ordered_parts());
}


TYPED_TEST(Partition, IsOrderedFail)
{
    using part_type = typename TestFixture::part_type;
    auto part = part_type::build_from_mapping(
        this->ref, gko::array<comm_index_type>{this->ref, {1, 1, 0, 0, 2}}, 3);

    ASSERT_FALSE(part->has_ordered_parts());
}


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


TEST_F(IndexMap, CanConstruct)
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
