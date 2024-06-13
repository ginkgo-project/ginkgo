// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/disjoint_sets.hpp"


#include <algorithm>
#include <bitset>
#include <type_traits>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class DisjointSets : public ::testing::Test {
protected:
    DisjointSets() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(DisjointSets, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(DisjointSets, InitializesCorrectly)
{
    const gko::disjoint_sets<TypeParam> sets{this->exec, 10};

    ASSERT_EQ(sets.get_size(), 10);
    for (int i = 0; i < 10; i++) {
        ASSERT_EQ(sets.const_find(i), i);
        ASSERT_EQ(sets.get_set_size(i), 1);
        ASSERT_EQ(sets.get_path_length(i), 0);
    }
}


TYPED_TEST(DisjointSets, SequentialMergesIntoSingleSet)
{
    gko::disjoint_sets<TypeParam> sets{this->exec, 10};

    // merge pairs (0,i), which makes 0 rep to all of them
    // path compression makes 0 direct parent to all
    for (int i = 1; i < 10; i++) {
        // before join: 0 represents [0, i)
        ASSERT_TRUE(sets.is_representative(0));
        ASSERT_EQ(sets.const_find(0), 0);
        ASSERT_EQ(sets.get_path_length(0), 0);
        ASSERT_EQ(sets.get_set_size(0), i);
        for (int j = 1; j < i; j++) {
            ASSERT_FALSE(sets.is_representative(j));
            ASSERT_EQ(sets.const_find(j), 0);
            ASSERT_EQ(sets.get_path_length(j), 1);
            ASSERT_EQ(sets.get_set_size(j), i);
        }
        // everything else is a singleton
        for (int j = i; j < 10; j++) {
            ASSERT_TRUE(sets.is_representative(j));
            ASSERT_EQ(sets.const_find(j), j);
            ASSERT_EQ(sets.get_path_length(j), 0);
            ASSERT_EQ(sets.get_set_size(j), 1);
        }

        // after join: 0 remains representative, i direct child of 0
        ASSERT_EQ(sets.join(0, i), 0);

        ASSERT_TRUE(sets.is_representative(0));
        ASSERT_EQ(sets.const_find(0), 0);
        ASSERT_EQ(sets.get_path_length(0), 0);
        ASSERT_EQ(sets.get_set_size(0), i + 1);
        for (int j = 1; j <= i; j++) {
            ASSERT_FALSE(sets.is_representative(j));
            ASSERT_EQ(sets.const_find(j), 0);
            ASSERT_EQ(sets.get_path_length(j), 1);
            ASSERT_EQ(sets.get_set_size(j), i + 1);
        }
        // everything else is a singleton
        for (int j = i + 1; j < 10; j++) {
            ASSERT_TRUE(sets.is_representative(j));
            ASSERT_EQ(sets.const_find(j), j);
            ASSERT_EQ(sets.get_path_length(j), 0);
            ASSERT_EQ(sets.get_set_size(j), 1);
        }
    }
}


int popcount(int i)
{
    return std::bitset<32>(static_cast<unsigned>(i)).count();
}


TYPED_TEST(DisjointSets, BinaryTreeMergesIntoSingleSet)
{
    gko::disjoint_sets<TypeParam> sets{this->exec, 16};

    int d = 1;
    while (d < 16) {
        d *= 2;
        // merge pairs (i, i + d / 2) for i % d == 0
        for (int i = 0; i < 16; i += d) {
            ASSERT_EQ(sets.join(i, i + d / 2), i);
        }
        for (int i = 0; i < 16; i++) {
            // afterwards, every i where i % d == 0 is a rep
            ASSERT_EQ(sets.is_representative(i), i % d == 0);
            // every other i has its rep at the beginning of its block of size d
            ASSERT_EQ(sets.const_find(i), i / d * d);
            // every set has size d
            ASSERT_EQ(sets.get_set_size(i), d);
            // and every i walks to i with the last bit cleared,
            // because we only merged representatives from the previous level
            ASSERT_EQ(sets.get_path_length(i), popcount(i & (d - 1)));
        }
    }

    // check that path compression works by compressing some long paths
    ASSERT_EQ(sets.find(15), 0);
    ASSERT_EQ(sets.get_path_length(15), 1);
    ASSERT_EQ(sets.get_path_length(14), 1);
    ASSERT_EQ(sets.get_path_length(12), 1);
    ASSERT_EQ(sets.get_path_length(8), 1);
    ASSERT_EQ(sets.get_path_length(13), 2);
    ASSERT_EQ(sets.find(13), 0);
    ASSERT_EQ(sets.get_path_length(13), 1);
    ASSERT_EQ(sets.get_path_length(6), 2);
    ASSERT_EQ(sets.find(6), 0);
    ASSERT_EQ(sets.get_path_length(7), 2);
    ASSERT_EQ(sets.get_path_length(6), 1);
    ASSERT_EQ(sets.get_path_length(4), 1);
}


TYPED_TEST(DisjointSets, WorksForGeneralSetting)
{
    gko::disjoint_sets<TypeParam> sets{this->exec, 6};

    // merge 4 and 3
    ASSERT_EQ(sets.join(4, 3), 4);
    ASSERT_TRUE(sets.is_representative(4));
    ASSERT_FALSE(sets.is_representative(3));
    ASSERT_EQ(sets.const_find(4), 4);
    ASSERT_EQ(sets.const_find(3), 4);
    ASSERT_EQ(sets.get_set_size(4), 2);
    ASSERT_EQ(sets.get_set_size(3), 2);
    ASSERT_EQ(sets.get_path_length(4), 0);
    ASSERT_EQ(sets.get_path_length(3), 1);
    // merge 1 and 2
    ASSERT_EQ(sets.join(2, 1), 2);
    ASSERT_TRUE(sets.is_representative(2));
    ASSERT_FALSE(sets.is_representative(1));
    ASSERT_EQ(sets.const_find(2), 2);
    ASSERT_EQ(sets.const_find(1), 2);
    ASSERT_EQ(sets.get_set_size(2), 2);
    ASSERT_EQ(sets.get_set_size(1), 2);
    ASSERT_EQ(sets.get_path_length(2), 0);
    ASSERT_EQ(sets.get_path_length(1), 1);
    // merge 3 and 5
    ASSERT_EQ(sets.join(3, 5), 4);
    ASSERT_TRUE(sets.is_representative(4));
    ASSERT_FALSE(sets.is_representative(3));
    ASSERT_FALSE(sets.is_representative(5));
    ASSERT_EQ(sets.const_find(3), 4);
    ASSERT_EQ(sets.const_find(4), 4);
    ASSERT_EQ(sets.const_find(5), 4);
    ASSERT_EQ(sets.get_set_size(3), 3);
    ASSERT_EQ(sets.get_set_size(4), 3);
    ASSERT_EQ(sets.get_set_size(5), 3);
    ASSERT_EQ(sets.get_path_length(3), 1);
    ASSERT_EQ(sets.get_path_length(4), 0);
    ASSERT_EQ(sets.get_path_length(5), 1);
    // merge 2 and 3
    ASSERT_EQ(sets.join(2, 3), 4);
    ASSERT_TRUE(sets.is_representative(4));
    ASSERT_FALSE(sets.is_representative(1));
    ASSERT_FALSE(sets.is_representative(2));
    ASSERT_FALSE(sets.is_representative(3));
    ASSERT_FALSE(sets.is_representative(5));
    ASSERT_EQ(sets.const_find(1), 4);
    ASSERT_EQ(sets.const_find(2), 4);
    ASSERT_EQ(sets.const_find(3), 4);
    ASSERT_EQ(sets.const_find(4), 4);
    ASSERT_EQ(sets.const_find(5), 4);
    ASSERT_EQ(sets.get_set_size(1), 5);
    ASSERT_EQ(sets.get_set_size(2), 5);
    ASSERT_EQ(sets.get_set_size(3), 5);
    ASSERT_EQ(sets.get_set_size(4), 5);
    ASSERT_EQ(sets.get_set_size(5), 5);
    ASSERT_EQ(sets.get_path_length(1), 2);
    ASSERT_EQ(sets.get_path_length(2), 1);
    ASSERT_EQ(sets.get_path_length(3), 1);
    ASSERT_EQ(sets.get_path_length(4), 0);
    ASSERT_EQ(sets.get_path_length(5), 1);
    // path-compress 1
    ASSERT_EQ(sets.find(1), 4);
    ASSERT_EQ(sets.get_path_length(1), 1);
}


}  // namespace
