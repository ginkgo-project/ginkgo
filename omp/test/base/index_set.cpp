// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/index_set.hpp>


#include <random>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/index_set_kernels.hpp"
#include "core/test/utils.hpp"


namespace {


template <typename T>
class index_set : public ::testing::Test {
protected:
    using index_type = T;
    index_set()
        : omp(gko::OmpExecutor::create()), ref(gko::ReferenceExecutor::create())
    {}

    gko::array<index_type> setup_random_indices(int num_indices = 100)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<index_type> dist(0, num_indices);
        std::vector<index_type> index_vec(num_indices);
        for (auto& i : index_vec) {
            i = dist(gen);
        }
        auto rand_index_arr = gko::array<index_type>(
            this->ref, index_vec.data(), num_indices + index_vec.data());
        return std::move(rand_index_arr);
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

    std::shared_ptr<const gko::OmpExecutor> omp;
    std::shared_ptr<const gko::ReferenceExecutor> ref;
};

TYPED_TEST_SUITE(index_set, gko::test::IndexTypes, TypenameNameGenerator);


TYPED_TEST(index_set, PopulateSubsetsIsEquivalentToReferenceForUnsortedInput)
{
    auto rand_arr = this->setup_random_indices(512);
    auto ref_begin_comp = gko::array<TypeParam>{this->ref};
    auto ref_end_comp = gko::array<TypeParam>{this->ref};
    auto ref_superset_comp = gko::array<TypeParam>{this->ref};
    auto omp_begin_comp = gko::array<TypeParam>{this->omp};
    auto omp_end_comp = gko::array<TypeParam>{this->omp};
    auto omp_superset_comp = gko::array<TypeParam>{this->omp};

    gko::kernels::reference::idx_set::populate_subsets(
        this->ref, TypeParam(520), &rand_arr, &ref_begin_comp, &ref_end_comp,
        &ref_superset_comp, false);
    gko::kernels::omp::idx_set::populate_subsets(
        this->omp, TypeParam(520), &rand_arr, &omp_begin_comp, &omp_end_comp,
        &omp_superset_comp, false);

    GKO_ASSERT_ARRAY_EQ(ref_begin_comp, omp_begin_comp);
    GKO_ASSERT_ARRAY_EQ(ref_end_comp, omp_end_comp);
    GKO_ASSERT_ARRAY_EQ(ref_superset_comp, omp_superset_comp);
}


TYPED_TEST(index_set, PopulateSubsetsIsEquivalentToReferenceForSortedInput)
{
    auto rand_arr = this->setup_random_indices(512);
    std::sort(rand_arr.get_data(), rand_arr.get_data() + rand_arr.get_size());
    auto ref_begin_comp = gko::array<TypeParam>{this->ref};
    auto ref_end_comp = gko::array<TypeParam>{this->ref};
    auto ref_superset_comp = gko::array<TypeParam>{this->ref};
    auto omp_begin_comp = gko::array<TypeParam>{this->omp};
    auto omp_end_comp = gko::array<TypeParam>{this->omp};
    auto omp_superset_comp = gko::array<TypeParam>{this->omp};

    gko::kernels::reference::idx_set::populate_subsets(
        this->ref, TypeParam(520), &rand_arr, &ref_begin_comp, &ref_end_comp,
        &ref_superset_comp, false);
    gko::kernels::omp::idx_set::populate_subsets(
        this->omp, TypeParam(520), &rand_arr, &omp_begin_comp, &omp_end_comp,
        &omp_superset_comp, false);

    GKO_ASSERT_ARRAY_EQ(ref_begin_comp, omp_begin_comp);
    GKO_ASSERT_ARRAY_EQ(ref_end_comp, omp_end_comp);
    GKO_ASSERT_ARRAY_EQ(ref_superset_comp, omp_superset_comp);
}


TYPED_TEST(index_set, IndicesContainsIsEquivalentToReference)
{
    auto rand_arr = this->setup_random_indices(512);
    auto ref_idx_set = gko::index_set<TypeParam>(this->ref, 520, rand_arr);
    auto omp_idx_set = gko::index_set<TypeParam>(this->omp, 520, rand_arr);

    auto ref_indices_arr = this->setup_random_indices(73);
    auto ref_validity_arr = gko::array<bool>(this->omp, 73);
    gko::kernels::reference::idx_set::compute_validity(
        this->ref, &ref_indices_arr, &ref_validity_arr);
    auto omp_indices_arr = gko::array<TypeParam>(this->omp, ref_indices_arr);
    auto omp_validity_arr = gko::array<bool>(this->omp, 73);
    gko::kernels::omp::idx_set::compute_validity(this->omp, &omp_indices_arr,
                                                 &omp_validity_arr);

    GKO_ASSERT_ARRAY_EQ(ref_validity_arr, omp_validity_arr);
}


TYPED_TEST(index_set, GetGlobalIndicesIsEquivalentToReference)
{
    auto rand_arr = this->setup_random_indices(512);
    auto rand_global_arr = this->setup_random_indices(256);
    auto ref_idx_set = gko::index_set<TypeParam>(this->ref, 520, rand_arr);
    auto ref_begin_comp = gko::array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_begin(),
        ref_idx_set.get_subsets_begin() + ref_idx_set.get_num_subsets()};
    auto ref_end_comp = gko::array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_end(),
        ref_idx_set.get_subsets_end() + ref_idx_set.get_num_subsets()};
    auto ref_superset_comp = gko::array<TypeParam>{
        this->ref, ref_idx_set.get_superset_indices(),
        ref_idx_set.get_superset_indices() + ref_idx_set.get_num_subsets()};
    auto omp_idx_set = gko::index_set<TypeParam>(this->omp, 520, rand_arr);
    auto omp_begin_comp = gko::array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_begin(),
        omp_idx_set.get_subsets_begin() + omp_idx_set.get_num_subsets()};
    auto omp_end_comp = gko::array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_end(),
        omp_idx_set.get_subsets_end() + omp_idx_set.get_num_subsets()};
    auto omp_superset_comp = gko::array<TypeParam>{
        this->omp, omp_idx_set.get_superset_indices(),
        omp_idx_set.get_superset_indices() + omp_idx_set.get_num_subsets()};

    auto ref_local_arr =
        gko::array<TypeParam>{this->ref, rand_global_arr.get_size()};
    gko::kernels::reference::idx_set::global_to_local(
        this->ref, TypeParam(520), ref_idx_set.get_num_subsets(),
        ref_idx_set.get_subsets_begin(), ref_idx_set.get_subsets_end(),
        ref_idx_set.get_superset_indices(),
        static_cast<TypeParam>(rand_global_arr.get_size()),
        rand_global_arr.get_const_data(), ref_local_arr.get_data(), false);
    auto omp_local_arr =
        gko::array<TypeParam>{this->omp, rand_global_arr.get_size()};
    gko::kernels::omp::idx_set::global_to_local(
        this->omp, TypeParam(520), omp_idx_set.get_num_subsets(),
        omp_idx_set.get_subsets_begin(), omp_idx_set.get_subsets_end(),
        omp_idx_set.get_superset_indices(),
        static_cast<TypeParam>(rand_global_arr.get_size()),
        rand_global_arr.get_const_data(), omp_local_arr.get_data(), false);

    ASSERT_EQ(rand_global_arr.get_size(), omp_local_arr.get_size());
    GKO_ASSERT_ARRAY_EQ(ref_local_arr, omp_local_arr);
}


TYPED_TEST(index_set, GetLocalIndicesIsEquivalentToReference)
{
    auto rand_arr = this->setup_random_indices(512);
    auto rand_local_arr = this->setup_random_indices(256);
    auto ref_idx_set = gko::index_set<TypeParam>(this->ref, 520, rand_arr);
    auto ref_begin_comp = gko::array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_begin(),
        ref_idx_set.get_subsets_begin() + ref_idx_set.get_num_subsets()};
    auto ref_end_comp = gko::array<TypeParam>{
        this->ref, ref_idx_set.get_subsets_end(),
        ref_idx_set.get_subsets_end() + ref_idx_set.get_num_subsets()};
    auto ref_superset_comp = gko::array<TypeParam>{
        this->ref, ref_idx_set.get_superset_indices(),
        ref_idx_set.get_superset_indices() + ref_idx_set.get_num_subsets()};
    auto omp_idx_set = gko::index_set<TypeParam>(this->omp, 520, rand_arr);
    auto omp_begin_comp = gko::array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_begin(),
        omp_idx_set.get_subsets_begin() + omp_idx_set.get_num_subsets()};
    auto omp_end_comp = gko::array<TypeParam>{
        this->omp, omp_idx_set.get_subsets_end(),
        omp_idx_set.get_subsets_end() + omp_idx_set.get_num_subsets()};
    auto omp_superset_comp = gko::array<TypeParam>{
        this->omp, omp_idx_set.get_superset_indices(),
        omp_idx_set.get_superset_indices() + omp_idx_set.get_num_subsets()};

    auto ref_global_arr =
        gko::array<TypeParam>{this->ref, rand_local_arr.get_size()};
    gko::kernels::reference::idx_set::local_to_global(
        this->ref, ref_idx_set.get_num_subsets(),
        ref_idx_set.get_subsets_begin(), ref_idx_set.get_superset_indices(),
        static_cast<TypeParam>(rand_local_arr.get_size()),
        rand_local_arr.get_const_data(), ref_global_arr.get_data(), false);
    auto omp_global_arr =
        gko::array<TypeParam>{this->omp, rand_local_arr.get_size()};
    gko::kernels::omp::idx_set::local_to_global(
        this->omp, omp_idx_set.get_num_subsets(),
        omp_idx_set.get_subsets_begin(), omp_idx_set.get_superset_indices(),
        static_cast<TypeParam>(rand_local_arr.get_size()),
        rand_local_arr.get_const_data(), omp_global_arr.get_data(), false);

    ASSERT_EQ(rand_local_arr.get_size(), omp_global_arr.get_size());
    GKO_ASSERT_ARRAY_EQ(ref_global_arr, omp_global_arr);
}


}  // namespace
