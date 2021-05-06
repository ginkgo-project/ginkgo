/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/index_set.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class IndexSet : public ::testing::Test {
protected:
    using value_type = T;
    IndexSet() : exec(gko::ReferenceExecutor::create()) {}

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

    static void assert_equal_index_sets(gko::IndexSet<T> &a,
                                        gko::IndexSet<T> &b)
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

    static void assert_equal_arrays(const T num_elems, const T *a, const T *b)
    {
        if (num_elems > 0) {
            for (auto i = 0; i < num_elems; ++i) {
                EXPECT_EQ(a[i], b[i]);
            }
        }
    }

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(IndexSet, gko::test::IndexTypes);


TYPED_TEST(IndexSet, CanBeConstructedFromIndices)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::Array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::Array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::Array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_subsets(), 3);
    ASSERT_EQ(idx_set.get_num_subsets(), begin_comp.get_num_elems());
    auto num_subsets = idx_set.get_num_subsets();
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_begin(),
                              begin_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_end(),
                              end_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_superset_indices(),
                              superset_comp.get_data());
}


TYPED_TEST(IndexSet, CanBeConstructedFromNonSortedIndices)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {9, 1, 4, 2, 6, 8, 0, 7}};
    auto begin_comp = gko::Array<TypeParam>{this->exec, {0, 4, 6}};
    auto end_comp = gko::Array<TypeParam>{this->exec, {3, 5, 10}};
    auto superset_comp = gko::Array<TypeParam>{this->exec, {0, 3, 4, 8}};

    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_size(), 10);
    ASSERT_EQ(idx_set.get_num_subsets(), 3);
    ASSERT_EQ(idx_set.get_num_subsets(), begin_comp.get_num_elems());
    auto num_subsets = idx_set.get_num_subsets();
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_begin(),
                              begin_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_subsets_end(),
                              end_comp.get_data());
    this->assert_equal_arrays(num_subsets, idx_set.get_superset_indices(),
                              superset_comp.get_data());
}


TYPED_TEST(IndexSet, CanDetectContiguousIndexSets)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 3, 4, 5, 6}};

    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_subsets(), 1);
    ASSERT_TRUE(idx_set.is_contiguous());
}


TYPED_TEST(IndexSet, CanDetectNonContiguousIndexSets)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 3, 4, 5, 6}};

    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_GT(idx_set.get_num_subsets(), 1);
    ASSERT_FALSE(idx_set.is_contiguous());
}


TYPED_TEST(IndexSet, CanDetectElementInIndexSet)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 3, 4, 5, 6}};

    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_subsets(), 2);
    ASSERT_TRUE(idx_set.is_element(4));
    ASSERT_FALSE(idx_set.is_element(2));
}

TYPED_TEST(IndexSet, CanGetGlobalIndex)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};

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


TYPED_TEST(IndexSet, CanGetGlobalIndexFromSortedArrays)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto lidx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 4, 6, 7}};
    auto gidx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 6, 8, 9}};
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_gidx = idx_set.get_global_indices(lidx_arr, true);

    this->assert_equal_arrays(gidx_arr.get_num_elems(),
                              idx_set_gidx.get_const_data(),
                              gidx_arr.get_const_data());
}


TYPED_TEST(IndexSet, CanGetGlobalIndexFromUnsortedArrays)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto lidx_arr = gko::Array<TypeParam>{this->exec, {4, 7, 0, 6, 1}};
    auto gidx_arr = gko::Array<TypeParam>{this->exec, {6, 9, 0, 8, 1}};
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_gidx = idx_set.get_global_indices(lidx_arr);

    this->assert_equal_arrays(gidx_arr.get_num_elems(),
                              idx_set_gidx.get_const_data(),
                              gidx_arr.get_const_data());
}


TYPED_TEST(IndexSet, CanGetLocalIndex)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};

    ASSERT_EQ(idx_set.get_num_elems(), 8);
    EXPECT_EQ(idx_set.get_local_index(6), 4);
    EXPECT_EQ(idx_set.get_local_index(7), 5);
    EXPECT_EQ(idx_set.get_local_index(0), 0);
    EXPECT_EQ(idx_set.get_local_index(8), 6);
    EXPECT_EQ(idx_set.get_local_index(4), 3);
}


TYPED_TEST(IndexSet, CanDetectNonExistentIndices)
{
    auto idx_arr = gko::Array<TypeParam>{
        this->exec, {0, 8, 1, 2, 3, 4, 6, 11, 9, 5, 7, 28, 39}};
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 45, idx_arr};

    ASSERT_EQ(idx_set.get_num_elems(), 13);
    EXPECT_EQ(idx_set.get_local_index(11), 10);
    EXPECT_EQ(idx_set.get_local_index(22), -1);
}


TYPED_TEST(IndexSet, CanGetLocalIndexFromSortedArrays)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto gidx_arr = gko::Array<TypeParam>{this->exec, {0, 4, 6, 8, 9}};
    auto lidx_arr = gko::Array<TypeParam>{this->exec, {0, 3, 4, 6, 7}};
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_lidx = idx_set.get_local_indices(gidx_arr, true);

    this->assert_equal_arrays(lidx_arr.get_num_elems(),
                              idx_set_lidx.get_const_data(),
                              lidx_arr.get_const_data());
}


TYPED_TEST(IndexSet, CanGetLocalIndexFromUnsortedArrays)
{
    auto idx_arr = gko::Array<TypeParam>{this->exec, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto gidx_arr = gko::Array<TypeParam>{this->exec, {6, 0, 4, 8, 9}};
    auto lidx_arr = gko::Array<TypeParam>{this->exec, {4, 0, 3, 6, 7}};
    auto idx_set = gko::IndexSet<TypeParam>{this->exec, 10, idx_arr};
    ASSERT_EQ(idx_set.get_num_elems(), 8);

    auto idx_set_lidx = idx_set.get_local_indices(gidx_arr);

    this->assert_equal_arrays(lidx_arr.get_num_elems(),
                              idx_set_lidx.get_const_data(),
                              lidx_arr.get_const_data());
}


}  // namespace
