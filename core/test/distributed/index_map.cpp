// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/distributed/index_map.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


using gko::experimental::distributed::comm_index_type;


template <typename LocalGlobalIndexType>
class IndexMap : public ::testing::Test {
public:
    using local_index_type =
        typename std::tuple_element<0, decltype(LocalGlobalIndexType())>::type;
    using global_index_type =
        typename std::tuple_element<1, decltype(LocalGlobalIndexType())>::type;
    using part_type =
        gko::experimental::distributed::Partition<local_index_type,
                                                  global_index_type>;
    using map_type =
        gko::experimental::distributed::index_map<local_index_type,
                                                  global_index_type>;

    std::shared_ptr<const gko::Executor> exec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<part_type> part =
        part_type::build_from_global_size_uniform(exec, 3, 6);
};

TYPED_TEST_SUITE(IndexMap, gko::test::LocalGlobalIndexTypes,
                 PairTypenameNameGenerator);


template <typename T>
void assert_collection_eq(const gko::segmented_array<T>& a,
                          const gko::segmented_array<T>& b)
{
    ASSERT_EQ(a.size(), b.size());
    GKO_ASSERT_ARRAY_EQ(a.get_flat(), b.get_flat());
    GKO_ASSERT_ARRAY_EQ(a.get_offsets(), b.get_offsets());
}


TYPED_TEST(IndexMap, CanDefaultConstruct)
{
    using map_type = typename TestFixture::map_type;

    auto imap = map_type(this->exec);

    ASSERT_EQ(imap.get_local_size(), 0);
    ASSERT_EQ(imap.get_non_local_size(), 0);
    ASSERT_EQ(imap.get_remote_target_ids().get_size(), 0);
    ASSERT_EQ(imap.get_remote_global_idxs().get_flat().get_size(), 0);
    ASSERT_EQ(imap.get_remote_local_idxs().get_flat().get_size(), 0);
}


TYPED_TEST(IndexMap, CanCopyConstruct)
{
    using map_type = typename TestFixture::map_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<global_index_type> connections(this->exec, {4, 3, 3, 4, 2});
    auto imap = map_type(this->exec, this->part, 0, connections);

    auto copy = imap;

    GKO_ASSERT_ARRAY_EQ(copy.get_remote_target_ids(),
                        imap.get_remote_target_ids());
    assert_collection_eq(copy.get_remote_local_idxs(),
                         imap.get_remote_local_idxs());
    assert_collection_eq(copy.get_remote_global_idxs(),
                         imap.get_remote_global_idxs());
}

TYPED_TEST(IndexMap, CanMoveConstruct)
{
    using map_type = typename TestFixture::map_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<global_index_type> connections(this->exec, {4, 3, 3, 4, 2});
    auto imap = map_type(this->exec, this->part, 0, connections);
    auto copy = imap;
    auto imap_remote_global_it =
        imap.get_remote_global_idxs().get_flat().get_const_data();
    auto imap_remote_local_it =
        imap.get_remote_local_idxs().get_flat().get_const_data();
    auto imap_remote_target_it = imap.get_remote_target_ids().get_const_data();

    auto move = std::move(imap);

    GKO_ASSERT_ARRAY_EQ(move.get_remote_target_ids(),
                        copy.get_remote_target_ids());
    assert_collection_eq(move.get_remote_local_idxs(),
                         copy.get_remote_local_idxs());
    assert_collection_eq(move.get_remote_global_idxs(),
                         copy.get_remote_global_idxs());
    ASSERT_EQ(move.get_remote_target_ids().get_const_data(),
              imap_remote_target_it);
    ASSERT_EQ(move.get_remote_local_idxs().get_flat().get_const_data(),
              imap_remote_local_it);
    ASSERT_EQ(move.get_remote_global_idxs().get_flat().get_const_data(),
              imap_remote_global_it);
}


TYPED_TEST(IndexMap, CanCopyAssign)
{
    using map_type = typename TestFixture::map_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<global_index_type> connections(this->exec, {4, 3, 3, 4, 2});
    auto imap = map_type(this->exec, this->part, 0, connections);
    auto copy = map_type(this->exec);

    copy = imap;

    GKO_ASSERT_ARRAY_EQ(copy.get_remote_target_ids(),
                        imap.get_remote_target_ids());
    assert_collection_eq(copy.get_remote_local_idxs(),
                         imap.get_remote_local_idxs());
    assert_collection_eq(copy.get_remote_global_idxs(),
                         imap.get_remote_global_idxs());
}


TYPED_TEST(IndexMap, CanMoveAssign)
{
    using map_type = typename TestFixture::map_type;
    using global_index_type = typename TestFixture::global_index_type;
    gko::array<global_index_type> connections(this->exec, {4, 3, 3, 4, 2});
    auto imap = map_type(this->exec, this->part, 0, connections);
    auto copy = imap;
    auto imap_remote_global_it =
        imap.get_remote_global_idxs().get_flat().get_const_data();
    auto imap_remote_local_it =
        imap.get_remote_local_idxs().get_flat().get_const_data();
    auto imap_remote_target_it = imap.get_remote_target_ids().get_const_data();
    auto move = map_type(this->exec);

    move = std::move(imap);

    GKO_ASSERT_ARRAY_EQ(move.get_remote_target_ids(),
                        copy.get_remote_target_ids());
    assert_collection_eq(move.get_remote_local_idxs(),
                         copy.get_remote_local_idxs());
    assert_collection_eq(move.get_remote_global_idxs(),
                         copy.get_remote_global_idxs());
    ASSERT_EQ(move.get_remote_target_ids().get_const_data(),
              imap_remote_target_it);
    ASSERT_EQ(move.get_remote_local_idxs().get_flat().get_const_data(),
              imap_remote_local_it);
    ASSERT_EQ(move.get_remote_global_idxs().get_flat().get_const_data(),
              imap_remote_global_it);
}
