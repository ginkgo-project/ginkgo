/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class IndexSet : public ::testing::Test {
protected:
    using value_type = T;
    IndexSet() : exec(gko::OmpExecutor::create()) {}

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    static void assert_equal_to_original(gko::IndexSet<T> &a)
    {
        ASSERT_EQ(a.get_size(), 10);
    }


    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_CASE(IndexSet, gko::test::IndexTypes);


TYPED_TEST(IndexSet, CanBeEmpty)
{
    auto empty = gko::IndexSet<TypeParam>{};
    ASSERT_EQ(empty.get_size(), 0);
    ASSERT_TRUE(empty.is_empty());
}


TYPED_TEST(IndexSet, CanBeConstructedWithSize)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_TRUE(idx_set.is_empty());
}


TYPED_TEST(IndexSet, CanBeCopyConstructed)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2(idx_set);
    ASSERT_EQ(idx_set2, idx_set);
}


TYPED_TEST(IndexSet, CanBeMoveConstructed)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2(std::move(idx_set));
    this->assert_equal_to_original(idx_set2);
}


TYPED_TEST(IndexSet, CanBeCopyAssigned)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2 = idx_set;
    ASSERT_EQ(idx_set2, idx_set);
}


TYPED_TEST(IndexSet, CanBeMoveAssigned)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2 = std::move(idx_set);
    this->assert_equal_to_original(idx_set2);
}


TYPED_TEST(IndexSet, KnowsItsSize)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    ASSERT_EQ(idx_set.get_size(), 10);
}


TYPED_TEST(IndexSet, CanSetNewSize)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    ASSERT_EQ(idx_set.get_size(), 10);
    idx_set.set_size(12);
    ASSERT_EQ(idx_set.get_size(), 12);
}


TYPED_TEST(IndexSet, CanStoreSubsets)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 5);
    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_elems(), 5);
}


TYPED_TEST(IndexSet, FailsforIncorrectIndexSize)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    ASSERT_THROW(idx_set.add_subset(0, 15), gko::ConditionUnsatisfied);
}


TYPED_TEST(IndexSet, CanStoreNonContiguousSubsets)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 3);
    idx_set.add_subset(6, 10);
    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_elems(), 7);
    ASSERT_FALSE(idx_set.is_contiguous());
}


TYPED_TEST(IndexSet, KnowsNumberOfItsSubsets)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 3);
    idx_set.add_subset(6, 10);
    ASSERT_EQ(idx_set.get_num_subsets(), 2);
    ASSERT_FALSE(idx_set.is_contiguous());
}


TYPED_TEST(IndexSet, CanMergeContinuousSubsets)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 5);
    idx_set.add_subset(4, 10);
    ASSERT_EQ(idx_set.get_num_subsets(), 1);
    ASSERT_TRUE(idx_set.is_contiguous());
}


TYPED_TEST(IndexSet, CanAddAnIndex)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 3);
    idx_set.add_subset(4, 10);
    idx_set.add_index(3);
    idx_set.merge();
    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_elems(), 10);
    ASSERT_TRUE(idx_set.is_contiguous());
}


TYPED_TEST(IndexSet, CanAddRangeOfIndices)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    auto idx_set2 = gko::IndexSet<TypeParam>{this->exec, 7};
    idx_set.add_subset(0, 3);
    idx_set2.add_subset(1, 5);
    ASSERT_EQ(idx_set2.get_num_elems(), 4);
    idx_set2.add_indices(idx_set);
    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set2.get_num_elems(), 5);
}


TYPED_TEST(IndexSet, CanAddRangeOfIndicesWithIterators)
{
    auto idx_set2 = gko::IndexSet<TypeParam>{this->exec, 11};
    auto indices = std::vector<TypeParam>{0, 2, 1, 6, 8};
    idx_set2.add_subset(1, 5);
    ASSERT_EQ(idx_set2.get_num_elems(), 4);
    idx_set2.add_indices(indices.begin(), indices.end());
    ASSERT_EQ(idx_set2.get_num_elems(), 7);
    ASSERT_TRUE(idx_set2.is_element(4));
    ASSERT_TRUE(idx_set2.is_element(6));
    ASSERT_TRUE(idx_set2.is_element(8));
    ASSERT_FALSE(idx_set2.is_element(5));
}


TYPED_TEST(IndexSet, KnowsItsElements)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 3);
    ASSERT_TRUE(idx_set.is_element(0));
    ASSERT_FALSE(idx_set.is_element(3));
}


TYPED_TEST(IndexSet, CanGetGlobalIndex)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 3);
    idx_set.add_subset(5, 7);
    ASSERT_EQ(idx_set.get_num_elems(), 5);
    ASSERT_EQ(idx_set.get_global_index(4), 6);
}


TYPED_TEST(IndexSet, CanGetLocalIndex)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10};
    idx_set.add_subset(0, 3);
    idx_set.add_subset(5, 7);
    ASSERT_EQ(idx_set.get_num_elems(), 5);
    ASSERT_EQ(idx_set.get_local_index(5), 3);
}


TYPED_TEST(IndexSet, CanGetLargestSubsetStartIndex)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 40};
    idx_set.add_subset(0, 4);
    idx_set.add_subset(5, 7);
    idx_set.add_subset(10, 30);
    idx_set.merge();
    ASSERT_EQ(
        idx_set.get_global_index(idx_set.get_largest_subset_starting_index()),
        10);
}


TYPED_TEST(IndexSet, CanGetLargestElementInSet)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 40};
    idx_set.add_subset(0, 4);
    idx_set.add_subset(10, 30);
    idx_set.add_subset(5, 7);
    ASSERT_EQ(idx_set.get_largest_element_in_set(), 29);
}


TYPED_TEST(IndexSet, CanCompareIndexSets)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set2 = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set3 = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set4 = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set5 = gko::IndexSet<TypeParam>{this->exec, 11};
    idx_set.add_subset(2, 10);
    idx_set2.add_subset(2, 10);
    idx_set3.add_subset(3, 11);
    idx_set4.add_subset(5, 7);
    ASSERT_THROW(idx_set == idx_set5, gko::ConditionUnsatisfied);
    ASSERT_TRUE(idx_set == idx_set2);
    ASSERT_TRUE(idx_set2 != idx_set3);
    ASSERT_FALSE(idx_set4 == idx_set);
}


TYPED_TEST(IndexSet, CanGetIntersectionOfSets)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set2 = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set3 = gko::IndexSet<TypeParam>{this->exec, 12};
    idx_set.add_subset(0, 5);
    idx_set2.add_subset(2, 10);
    idx_set3.add_subset(2, 5);
    auto idx_set4 = (idx_set & idx_set2);
    ASSERT_EQ(idx_set4.get_num_elems(), 3);
    ASSERT_TRUE(idx_set3 == idx_set4);
}


TYPED_TEST(IndexSet, CanSubtractASet)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set2 = gko::IndexSet<TypeParam>{this->exec, 12};
    auto idx_set3 = gko::IndexSet<TypeParam>{this->exec, 12};
    idx_set.add_subset(0, 7);
    idx_set2.add_subset(4, 10);
    idx_set.subtract_set(idx_set2);
    idx_set3.add_subset(0, 4);
    ASSERT_EQ(idx_set.get_num_elems(), 4);
    ASSERT_TRUE(idx_set3 == idx_set);
}


TYPED_TEST(IndexSet, CanBeCleared)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 12};
    idx_set.add_subset(0, 7);
    ASSERT_EQ(idx_set.get_num_elems(), 7);
    idx_set.clear();
    ASSERT_EQ(idx_set.get_num_elems(), 0);
}


TYPED_TEST(IndexSet, ReturnsBeginOfIndexSet)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 15};
    idx_set.add_subset(3, 7);
    idx_set.add_subset(8, 13);

    ASSERT_EQ(*idx_set.begin(), 3);
}


TYPED_TEST(IndexSet, ReturnsSpecificElementOfIndexSet)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 55};
    idx_set.add_subset(3, 7);
    idx_set.add_subset(25, 43);
    ASSERT_EQ(*idx_set.at(5), 5);
    ASSERT_EQ(*idx_set.at(24), 25);
    ASSERT_THROW(*idx_set.at(65), gko::ConditionUnsatisfied);
}


TYPED_TEST(IndexSet, ReturnsFirstIntervalOfSet)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 15};
    idx_set.add_subset(3, 7);
    idx_set.add_subset(8, 13);
    auto first_int = idx_set.get_first_interval();
    ASSERT_EQ((*first_int).get_num_elems(), 4);
}


TYPED_TEST(IndexSet, KnowsIndicesWithinIntervals)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 15};
    idx_set.add_subset(2, 5);
    idx_set.add_subset(6, 8);
    idx_set.add_subset(10, 14);
    auto first_int = idx_set.get_first_interval();
    ASSERT_EQ(*(*first_int).begin(), 2);
    ASSERT_EQ((*first_int).last(), 4);
    ASSERT_EQ(*(*++first_int).begin(), 6);
    ASSERT_EQ((*first_int).last(), 7);
    ASSERT_EQ(*(*++first_int).begin(), 10);
    ASSERT_EQ((*first_int).last(), 13);
}


TYPED_TEST(IndexSet, CanIncrementBetweenIntervals)
{
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 15};
    idx_set.add_subset(3, 7);
    idx_set.add_subset(8, 14);
    auto first_int = idx_set.get_first_interval();
    ASSERT_EQ((*(++first_int)).get_num_elems(), 6);
}


}  // namespace
