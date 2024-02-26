// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/index_set.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class index_set : public ::testing::Test {
protected:
    using value_type = T;
    index_set() : exec(gko::ReferenceExecutor::create()) {}

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    static void assert_equal_index_sets(gko::index_set<T>& a,
                                        gko::index_set<T>& b)
    {
        ASSERT_EQ(a.get_size(), b.get_size());
        ASSERT_EQ(a.get_num_subsets(), b.get_num_subsets());
        if (a.get_num_subsets() > 0) {
            for (auto i = 0; i < a.get_num_subsets(); ++i) {
                EXPECT_EQ(a.get_subsets_begin()[i], b.get_subsets_begin()[i]);
                EXPECT_EQ(a.get_subsets_end()[i], b.get_subsets_end()[i]);
                EXPECT_EQ(a.get_superset_indices()[i],
                          b.get_superset_indices()[i]);
            }
        }
    }

    static void assert_equal_arrays(const T num_elems, const T* a, const T* b)
    {
        if (num_elems > 0) {
            for (auto i = 0; i < num_elems; ++i) {
                EXPECT_EQ(a[i], b[i]);
            }
        }
    }

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(index_set, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(index_set, KnowsItsExecutor)
{
    auto idx_set = gko::index_set<TypeParam>{this->exec};

    ASSERT_EQ(this->exec, idx_set.get_executor());
}


TYPED_TEST(index_set, CanBeCopyConstructed)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    gko::index_set<TypeParam> idx_set2(idx_set);

    ASSERT_EQ(idx_set2.get_executor(), idx_set.get_executor());
    this->assert_equal_index_sets(idx_set2, idx_set);
}


TYPED_TEST(index_set, CanBeMoveConstructed)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    gko::index_set<TypeParam> idx_set2(std::move(idx_set));

    ASSERT_EQ(idx_set2.get_executor(), this->exec);
    ASSERT_EQ(idx_set.get_size(), 0);
    ASSERT_EQ(idx_set2.get_size(), 10);
}


TYPED_TEST(index_set, CanBeCopyAssigned)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    gko::index_set<TypeParam> idx_set2 = idx_set;

    ASSERT_EQ(idx_set2.get_executor(), idx_set.get_executor());
    this->assert_equal_index_sets(idx_set2, idx_set);
}


TYPED_TEST(index_set, CanBeMoveAssigned)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    gko::index_set<TypeParam> idx_set2 = std::move(idx_set);

    ASSERT_EQ(idx_set2.get_executor(), this->exec);
    ASSERT_EQ(idx_set.get_size(), 0);
    ASSERT_EQ(idx_set2.get_size(), 10);
}


TYPED_TEST(index_set, KnowsItsSize)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_size(), 10);
}


TYPED_TEST(index_set, CanBeConstructedFromIndices)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_subsets(), 3);
    ASSERT_EQ(idx_set.get_num_subsets(), begin_comp.get_size());
    auto num_subsets = idx_set.get_num_subsets();
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_begin(),
                              begin_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_end(),
                              end_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_superset_indices(),
                              superset_comp.get_data());
}


TYPED_TEST(index_set, CanBeConvertedToGlobalIndices)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    auto out_arr = idx_set.to_global_indices();

    GKO_ASSERT_ARRAY_EQ(idx_arr, out_arr);
}


TYPED_TEST(index_set, CanBeConstructedFromNonSortedIndices)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {9, 1, 4, 2, 6, 8, 0, 7}};
    auto begin_comp = gko::array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_subsets(), 3);
    ASSERT_EQ(idx_set.get_num_subsets(), begin_comp.get_size());
    auto num_subsets = idx_set.get_num_subsets();
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_begin(),
                              begin_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_end(),
                              end_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_superset_indices(),
                              superset_comp.get_data());
}


TYPED_TEST(index_set, CanDetectContiguousindex_sets)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 3, 4, 5, 6}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_subsets(), 1);
    ASSERT_TRUE(idx_set.is_contiguous());
}


TYPED_TEST(index_set, CanDetectNonContiguousindex_sets)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 3, 4, 5, 6}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_subsets(), 2);
    ASSERT_FALSE(idx_set.is_contiguous());
}


TYPED_TEST(index_set, CanDetectElementInindex_set)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 3, 4, 5, 6}};

    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_subsets(), 2);
    ASSERT_TRUE(idx_set.contains(4));
    ASSERT_FALSE(idx_set.contains(2));
}

TYPED_TEST(index_set, CanGetGlobalIndex)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_elems(), 8);
    EXPECT_EQ(idx_set.get_global_index(0), 0);
    EXPECT_EQ(idx_set.get_global_index(1), 1);
    EXPECT_EQ(idx_set.get_global_index(2), 2);
    EXPECT_EQ(idx_set.get_global_index(3), 4);
    EXPECT_EQ(idx_set.get_global_index(4), 6);
    EXPECT_EQ(idx_set.get_global_index(5), 7);
    EXPECT_EQ(idx_set.get_global_index(6), 8);
    EXPECT_EQ(idx_set.get_global_index(7), 9);
}


TYPED_TEST(index_set, CanGetGlobalIndexFromSortedArrays)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto lidx_arr = gko::array<TypeParam>{this->exec, {0, 1, 4, 6, 7}};
    auto gidx_arr = gko::array<TypeParam>{this->exec, {0, 1, 6, 8, 9}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_gidx = idx_set.map_local_to_global(lidx_arr, true);

    this->assert_equal_arrays(gidx_arr.get_size(),
                              idx_set_gidx.get_const_data(),
                              gidx_arr.get_const_data());
}


TYPED_TEST(index_set, CanGetGlobalIndexFromUnsortedArrays)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto lidx_arr = gko::array<TypeParam>{this->exec, {4, 7, 0, 6, 1}};
    auto gidx_arr = gko::array<TypeParam>{this->exec, {6, 9, 0, 8, 1}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_gidx = idx_set.map_local_to_global(lidx_arr);

    this->assert_equal_arrays(gidx_arr.get_size(),
                              idx_set_gidx.get_const_data(),
                              gidx_arr.get_const_data());
}


TYPED_TEST(index_set, CanGetLocalIndex)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_elems(), 8);
    EXPECT_EQ(idx_set.get_local_index(6), 4);
    EXPECT_EQ(idx_set.get_local_index(7), 5);
    EXPECT_EQ(idx_set.get_local_index(0), 0);
    EXPECT_EQ(idx_set.get_local_index(8), 6);
    EXPECT_EQ(idx_set.get_local_index(4), 3);
}


TYPED_TEST(index_set, CanDetectNonExistentIndices)
{
    auto idx_arr = gko::array<TypeParam>{
        this->exec, {0, 8, 1, 2, 3, 4, 6, 11, 9, 5, 7, 28, 39}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 45, idx_arr};

    ASSERT_EQ(idx_set.get_num_elems(), 13);
    EXPECT_EQ(idx_set.get_local_index(11), 10);
    EXPECT_EQ(idx_set.get_local_index(22), -1);
}


TYPED_TEST(index_set, CanGetLocalIndexFromSortedArrays)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto gidx_arr = gko::array<TypeParam>{this->exec, {0, 4, 6, 8, 9}};
    auto lidx_arr = gko::array<TypeParam>{this->exec, {0, 3, 4, 6, 7}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_lidx = idx_set.map_global_to_local(gidx_arr, true);

    this->assert_equal_arrays(lidx_arr.get_size(),
                              idx_set_lidx.get_const_data(),
                              lidx_arr.get_const_data());
}


TYPED_TEST(index_set, CanGetLocalIndexFromUnsortedArrays)
{
    auto idx_arr = gko::array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto gidx_arr = gko::array<TypeParam>{this->exec, {6, 0, 4, 8, 9}};
    auto lidx_arr = gko::array<TypeParam>{this->exec, {4, 0, 3, 6, 7}};
    auto idx_set = gko::index_set<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_lidx = idx_set.map_global_to_local(gidx_arr);

    this->assert_equal_arrays(lidx_arr.get_size(),
                              idx_set_lidx.get_const_data(),
                              lidx_arr.get_const_data());
}


}  // namespace
