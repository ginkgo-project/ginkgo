// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/index_set.hpp>


#include <algorithm>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>


#include "hip/test/utils.hip.hpp"


namespace {


class index_set : public HipTestFixture {
protected:
    using T = int;

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
};


TEST_F(index_set, CanBeCopiedBetweenExecutors)
{
    auto idx_arr = gko::array<T>{ref, {0, 1, 2, 4, 6, 7, 8, 9}};
    auto begin_comp = gko::array<T>{ref, {0, 4, 6}};
    auto end_comp = gko::array<T>{ref, {3, 5, 10}};
    auto superset_comp = gko::array<T>{ref, {0, 3, 4, 8}};

    auto idx_set = gko::index_set<T>{ref, 10, idx_arr};
    auto hip_idx_set = gko::index_set<T>(exec, idx_set);
    auto host_idx_set = gko::index_set<T>(ref, hip_idx_set);

    ASSERT_EQ(hip_idx_set.get_executor(), exec);
    this->assert_equal_index_sets(host_idx_set, idx_set);
}


}  // namespace
