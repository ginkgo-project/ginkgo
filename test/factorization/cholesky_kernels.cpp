// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/cholesky_kernels.hpp"

#include <algorithm>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>

#include "core/components/disjoint_sets.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "core/utils/matrix_utils.hpp"
#include "ginkgo/core/base/types.hpp"
#include "matrices/config.hpp"
#include "test/utils/common_fixture.hpp"


namespace {


template <typename ValueIndexType>
class CholeskySymbolic : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using elimination_forest =
        gko::factorization::elimination_forest<index_type>;

    CholeskySymbolic() : tmp{ref}, dtmp{exec}
    {
        matrices.emplace_back(
            "example small",
            gko::initialize<matrix_type>(
                {{1, 0, 1, 0}, {0, 1, 0, 1}, {1, 0, 1, 0}, {0, 1, 0, 1}}, ref));
        // this is the example from Liu 1990 https://doi.org/10.1137/0611010.
        // "The Role of Elimination Trees in Sparse Factorization"
        matrices.emplace_back("example", gko::initialize<matrix_type>(
                                             {{1, 0, 1, 0, 0, 0, 0, 1, 0, 0},
                                              {0, 1, 0, 0, 1, 0, 0, 0, 0, 1},
                                              {1, 0, 1, 0, 0, 0, 1, 0, 0, 0},
                                              {0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
                                              {0, 1, 0, 0, 1, 0, 0, 0, 1, 1},
                                              {0, 0, 0, 0, 0, 1, 1, 1, 0, 0},
                                              {0, 0, 1, 0, 0, 1, 1, 0, 0, 0},
                                              {1, 0, 0, 0, 0, 1, 0, 1, 1, 1},
                                              {0, 0, 0, 1, 1, 0, 0, 1, 1, 0},
                                              {0, 1, 0, 1, 1, 0, 0, 1, 0, 1}},
                                             ref));
        matrices.emplace_back("separable", gko::initialize<matrix_type>(
                                               {{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {1, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
                                                {0, 0, 0, 1, 1, 1, 0, 0, 0, 1},
                                                {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 1, 1, 0, 1},
                                                {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                                                {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                                                {0, 0, 0, 0, 1, 0, 1, 0, 1, 1}},
                                               ref));
        matrices.emplace_back(
            "missing diagonal",
            gko::initialize<matrix_type>({{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                                          {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                                          {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
                                          {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
                                          {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
                                          {0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                                          {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}},
                                         ref));
        std::ifstream ani1_stream{gko::matrices::location_ani1_mtx};
        matrices.emplace_back("ani1", gko::read<matrix_type>(ani1_stream, ref));
        std::ifstream ani1_amd_stream{gko::matrices::location_ani1_amd_mtx};
        matrices.emplace_back("ani1_amd",
                              gko::read<matrix_type>(ani1_amd_stream, ref));
        std::ifstream ani4_stream{gko::matrices::location_ani4_mtx};
        matrices.emplace_back("ani4", gko::read<matrix_type>(ani4_stream, ref));
        std::ifstream ani4_amd_stream{gko::matrices::location_ani4_amd_mtx};
        matrices.emplace_back("ani4_amd",
                              gko::read<matrix_type>(ani4_amd_stream, ref));
    }

    void assert_equal_forests(elimination_forest& lhs, elimination_forest& rhs,
                              bool check_postorder = false)
    {
        GKO_ASSERT_ARRAY_EQ(lhs.parents, rhs.parents);
        GKO_ASSERT_ARRAY_EQ(lhs.children, rhs.children);
        GKO_ASSERT_ARRAY_EQ(lhs.child_ptrs, rhs.child_ptrs);
        if (check_postorder) {
            GKO_ASSERT_ARRAY_EQ(lhs.postorder, rhs.postorder);
            GKO_ASSERT_ARRAY_EQ(lhs.postorder_parents, rhs.postorder_parents);
            GKO_ASSERT_ARRAY_EQ(lhs.inv_postorder, rhs.inv_postorder);
        }
    }

    index_type check_mst(const std::unique_ptr<matrix_type>& mst) const
    {
        gko::matrix_data<value_type, index_type> mst_data;
        const auto size = mst->get_size();
        mst->write(mst_data);
        gko::disjoint_sets<index_type> sets(this->ref, size[0]);
        index_type weight_sum{};
        // need an IIFE because the assertions would return something
        [&] {
            for (const auto entry : mst_data.nonzeros) {
                ASSERT_GT(entry.row, entry.column);
                weight_sum += entry.row;
                const auto row_rep = sets.find(entry.row);
                const auto col_rep = sets.find(entry.column);
                ASSERT_NE(row_rep, col_rep);
                sets.join(row_rep, col_rep);
            }
        }();
        return weight_sum;
    }

    std::vector<std::pair<std::string, std::unique_ptr<const matrix_type>>>
        matrices;
    gko::array<index_type> tmp;
    gko::array<index_type> dtmp;
};

#ifdef GKO_COMPILING_OMP
using Types = gko::test::ValueIndexTypesBase;
#elif defined(GKO_COMPILING_CUDA)
// CUDA doesn't support long indices for sorting, and the triangular solvers
// seem broken
using Types = gko::test::cartesian_type_product_t<gko::test::ValueTypes,
                                                  ::testing::Types<gko::int32>>;
#else
// HIP only supports real types and int32
using Types = gko::test::cartesian_type_product_t<gko::test::RealValueTypesBase,
                                                  ::testing::Types<gko::int32>>;
#endif

TYPED_TEST_SUITE(CholeskySymbolic, Types, PairTypenameNameGenerator);


TYPED_TEST(CholeskySymbolic, KernelComputeChildrenIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);

        gko::kernels::reference::elimination_forest::compute_children(
            this->ref, forest->parents.get_const_data(), ssize,
            forest->child_ptrs.get_data(), forest->children.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::
            compute_children(this->exec, dforest->parents.get_const_data(),
                             ssize, dforest->child_ptrs.get_data(),
                             dforest->children.get_data());

        GKO_ASSERT_ARRAY_EQ(forest->child_ptrs, dforest->child_ptrs);
        GKO_ASSERT_ARRAY_EQ(forest->children, dforest->children);
    }
}


TYPED_TEST(CholeskySymbolic, KernelComputeSubtreeSizesIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> subtree_sizes{this->ref, size};
        gko::array<index_type> dsubtree_sizes{this->exec, size};

        gko::kernels::reference::elimination_forest::compute_subtree_sizes(
            this->ref, forest->child_ptrs.get_const_data(),
            forest->children.get_const_data(), ssize, subtree_sizes.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::
            compute_subtree_sizes(this->exec,
                                  dforest->child_ptrs.get_const_data(),
                                  dforest->children.get_const_data(), ssize,
                                  dsubtree_sizes.get_data());

        GKO_ASSERT_ARRAY_EQ(subtree_sizes, dsubtree_sizes);
    }
}


TYPED_TEST(CholeskySymbolic, KernelComputeEulerPathSizesIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> subtree_sizes{this->ref, size};
        gko::array<index_type> dsubtree_sizes{this->exec, size};

        gko::kernels::reference::elimination_forest::
            compute_subtree_euler_path_sizes(
                this->ref, forest->child_ptrs.get_const_data(),
                forest->children.get_const_data(), ssize,
                subtree_sizes.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::
            compute_subtree_euler_path_sizes(
                this->exec, dforest->child_ptrs.get_const_data(),
                dforest->children.get_const_data(), ssize,
                dsubtree_sizes.get_data());

        GKO_ASSERT_ARRAY_EQ(subtree_sizes, dsubtree_sizes);
    }
}


TYPED_TEST(CholeskySymbolic, KernelComputeLevelsIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> levels{this->ref, size};
        gko::array<index_type> dlevels{this->exec, size};

        gko::kernels::reference::elimination_forest::compute_levels(
            this->ref, forest->parents.get_const_data(), ssize,
            levels.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::compute_levels(
            this->exec, dforest->parents.get_const_data(), ssize,
            dlevels.get_data());

        GKO_ASSERT_ARRAY_EQ(levels, dlevels);
    }
}


TYPED_TEST(CholeskySymbolic, KernelComputePostorderIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> subtree_sizes{this->ref, size};
        gko::kernels::reference::elimination_forest::compute_subtree_sizes(
            this->ref, forest->child_ptrs.get_const_data(),
            forest->children.get_const_data(), ssize, subtree_sizes.get_data());
        gko::array<index_type> dsubtree_sizes{this->exec, subtree_sizes};
        gko::array<index_type> postorder{this->ref, size};
        gko::array<index_type> dpostorder{this->exec, size};
        gko::array<index_type> inv_postorder{this->ref, size};
        gko::array<index_type> dinv_postorder{this->exec, size};

        gko::kernels::reference::elimination_forest::compute_postorder(
            this->ref, forest->child_ptrs.get_const_data(),
            forest->children.get_const_data(), ssize,
            subtree_sizes.get_const_data(), postorder.get_data(),
            inv_postorder.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::
            compute_postorder(this->exec, dforest->child_ptrs.get_const_data(),
                              dforest->children.get_const_data(), ssize,
                              dsubtree_sizes.get_const_data(),
                              dpostorder.get_data(), dinv_postorder.get_data());

        GKO_ASSERT_ARRAY_EQ(inv_postorder, dinv_postorder);
        GKO_ASSERT_ARRAY_EQ(postorder, dpostorder);
    }
}


TYPED_TEST(CholeskySymbolic, KernelMapPostorderIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);

        gko::kernels::reference::elimination_forest::map_postorder(
            this->ref, forest->parents.get_const_data(),
            forest->child_ptrs.get_const_data(),
            forest->children.get_const_data(), ssize,
            forest->inv_postorder.get_const_data(),
            forest->postorder_parents.get_data(),
            forest->postorder_child_ptrs.get_data(),
            forest->postorder_children.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::map_postorder(
            this->exec, dforest->parents.get_const_data(),
            dforest->child_ptrs.get_const_data(),
            dforest->children.get_const_data(), ssize,
            dforest->inv_postorder.get_const_data(),
            dforest->postorder_parents.get_data(),
            dforest->postorder_child_ptrs.get_data(),
            dforest->postorder_children.get_data());

        GKO_ASSERT_ARRAY_EQ(forest->postorder_parents,
                            dforest->postorder_parents);
        GKO_ASSERT_ARRAY_EQ(forest->postorder_child_ptrs,
                            dforest->postorder_child_ptrs);
        GKO_ASSERT_ARRAY_EQ(forest->postorder_children,
                            dforest->postorder_children);
    }
}


TYPED_TEST(CholeskySymbolic, KernelPointerDoubleIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> grandparents{this->ref, size};
        gko::array<index_type> dgrandparents{this->exec, size};

        gko::kernels::reference::elimination_forest::pointer_double(
            this->ref, forest->parents.get_const_data(), ssize,
            grandparents.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::pointer_double(
            this->exec, dforest->parents.get_const_data(), ssize,
            dgrandparents.get_data());

        GKO_ASSERT_ARRAY_EQ(grandparents, dgrandparents);
    }
}


TYPED_TEST(CholeskySymbolic, KernelComputeEulerPathIsEquivalentToRef)
{
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(mtx->get_size()[0]);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> euler_path_sizes{this->ref, size};
        gko::kernels::reference::elimination_forest::
            compute_subtree_euler_path_sizes(
                this->ref, forest->child_ptrs.get_const_data(),
                forest->children.get_const_data(), ssize,
                euler_path_sizes.get_data());
        gko::array<index_type> deuler_path_sizes{this->exec, euler_path_sizes};
        gko::array<index_type> levels{this->ref, size};
        gko::kernels::reference::elimination_forest::compute_levels(
            this->ref, forest->parents.get_const_data(), ssize,
            levels.get_data());
        gko::array<index_type> dlevels{this->exec, levels};
        gko::array<index_type> euler_path{this->ref, 2 * size + 1};
        gko::array<index_type> deuler_path{this->exec, 2 * size + 1};
        gko::array<index_type> euler_levels{this->ref, 2 * size + 1};
        gko::array<index_type> deuler_levels{this->exec, 2 * size + 1};
        gko::array<index_type> euler_first{this->ref, size};
        gko::array<index_type> deuler_first{this->exec, size};

        gko::kernels::reference::elimination_forest::compute_euler_path(
            this->ref, forest->child_ptrs.get_const_data(),
            forest->children.get_const_data(), ssize,
            euler_path_sizes.get_const_data(), levels.get_data(),
            euler_path.get_data(), euler_first.get_data(),
            euler_levels.get_data());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::
            compute_euler_path(this->exec, dforest->child_ptrs.get_const_data(),
                               dforest->children.get_const_data(), ssize,
                               deuler_path_sizes.get_const_data(),
                               dlevels.get_data(), deuler_path.get_data(),
                               deuler_first.get_data(),
                               deuler_levels.get_data());

        GKO_ASSERT_ARRAY_EQ(euler_first, deuler_first);
        GKO_ASSERT_ARRAY_EQ(euler_path, deuler_path);
        GKO_ASSERT_ARRAY_EQ(euler_levels, deuler_levels);
    }
}


TYPED_TEST(CholeskySymbolic, KernelComputeSkeletonTree)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        // check for correctness: is mtx symmetric?
        GKO_ASSERT_MTX_EQ_SPARSITY(mtx, gko::as<matrix_type>(mtx->transpose()));
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size();
        const auto skeleton = matrix_type::create(this->ref, size, size[0]);
        const auto dskeleton = matrix_type::create(this->exec, size, size[0]);

        gko::kernels::reference::elimination_forest::compute_skeleton_tree(
            this->ref, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            size[0], skeleton->get_row_ptrs(), skeleton->get_col_idxs());
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::
            compute_skeleton_tree(this->exec, dmtx->get_const_row_ptrs(),
                                  dmtx->get_const_col_idxs(), size[0],
                                  dskeleton->get_row_ptrs(),
                                  dskeleton->get_col_idxs());

        // check that the created graphs are trees and have the same edge sum
        const auto weight_sum = this->check_mst(skeleton);
        const auto dweight_sum = this->check_mst(dskeleton);
        ASSERT_EQ(weight_sum, dweight_sum);
        // while the MSTs may be different, they should produce the same
        // elimination forest as the original matrix
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> skeleton_forest;
        std::unique_ptr<elimination_forest> dskeleton_forest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(skeleton.get(),
                                                       skeleton_forest);
        gko::factorization::compute_elimination_forest(dskeleton.get(),
                                                       dskeleton_forest);
        // the parents array fully determines the elimination forest
        GKO_ASSERT_ARRAY_EQ(forest->parents, skeleton_forest->parents);
        GKO_ASSERT_ARRAY_EQ(skeleton_forest->parents,
                            dskeleton_forest->parents);
    }
}


TYPED_TEST(CholeskySymbolic, KernelComputeEliminationForest)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        // check for correctness: is mtx symmetric?
        GKO_ASSERT_MTX_EQ_SPARSITY(mtx, gko::as<matrix_type>(mtx->transpose()));
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto size = mtx->get_size()[0];
        const auto ssize = static_cast<index_type>(size);
        elimination_forest forest{this->ref, ssize};
        elimination_forest dforest{this->exec, ssize};

        gko::kernels::reference::elimination_forest::compute(
            this->ref, mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            size, forest);
        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::compute(
            this->exec, dmtx->get_const_row_ptrs(), dmtx->get_const_col_idxs(),
            size, dforest);

        GKO_ASSERT_ARRAY_EQ(forest.parents, dforest.parents);
    }
}


TYPED_TEST(CholeskySymbolic, KernelSymbolicCount)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> row_nnz{this->ref, mtx->get_size()[0]};
        gko::array<index_type> drow_nnz{this->exec, mtx->get_size()[0]};

        gko::kernels::reference::cholesky::symbolic_count(
            this->ref, mtx.get(), *forest, row_nnz.get_data(), this->tmp);
        gko::kernels::GKO_DEVICE_NAMESPACE::cholesky::symbolic_count(
            this->exec, dmtx.get(), *dforest, drow_nnz.get_data(), this->dtmp);

        GKO_ASSERT_ARRAY_EQ(drow_nnz, row_nnz);
    }
}


TYPED_TEST(CholeskySymbolic, KernelSymbolicFactorize)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using value_type = typename TestFixture::value_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        const auto num_rows = mtx->get_size()[0];
        std::unique_ptr<elimination_forest> forest;
        gko::factorization::compute_elimination_forest(mtx.get(), forest);
        gko::array<index_type> row_ptrs{this->ref, num_rows + 1};
        gko::kernels::reference::cholesky::symbolic_count(
            this->ref, mtx.get(), *forest, row_ptrs.get_data(), this->tmp);
        gko::kernels::reference::components::prefix_sum_nonnegative(
            this->ref, row_ptrs.get_data(), num_rows + 1);
        const auto nnz =
            static_cast<gko::size_type>(row_ptrs.get_const_data()[num_rows]);
        auto l_factor = matrix_type::create(
            this->ref, mtx->get_size(), gko::array<value_type>{this->ref, nnz},
            gko::array<index_type>{this->ref, nnz}, row_ptrs);
        auto dl_factor = matrix_type::create(
            this->exec, mtx->get_size(),
            gko::array<value_type>{this->exec, nnz},
            gko::array<index_type>{this->exec, nnz}, row_ptrs);
        // need to call the device kernels to initialize dtmp
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::compute_elimination_forest(dmtx.get(), dforest);
        gko::array<index_type> dtmp_ptrs{this->exec, num_rows + 1};
        gko::kernels::GKO_DEVICE_NAMESPACE::cholesky::symbolic_count(
            this->exec, dmtx.get(), *dforest, dtmp_ptrs.get_data(), this->dtmp);

        gko::kernels::reference::cholesky::symbolic_factorize(
            this->ref, mtx.get(), *forest, l_factor.get(), this->tmp);
        gko::kernels::GKO_DEVICE_NAMESPACE::cholesky::symbolic_factorize(
            this->exec, dmtx.get(), *dforest, dl_factor.get(), this->dtmp);

        GKO_ASSERT_MTX_EQ_SPARSITY(dl_factor, l_factor);
    }
}


TYPED_TEST(CholeskySymbolic, SymbolicFactorize)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        const auto dmtx = gko::clone(this->exec, mtx);
        std::unique_ptr<matrix_type> factors;
        std::unique_ptr<matrix_type> dfactors;
        std::unique_ptr<elimination_forest> forest;
        std::unique_ptr<elimination_forest> dforest;
        gko::factorization::symbolic_cholesky(mtx.get(), true, factors, forest);
        gko::factorization::symbolic_cholesky(mtx.get(), true, dfactors,
                                              dforest);

        GKO_ASSERT_MTX_EQ_SPARSITY(dfactors, factors);
        this->assert_equal_forests(*forest, *dforest, true);
    }
}


TYPED_TEST(CholeskySymbolic, KernelForestFromFactorWorks)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    for (const auto& pair : this->matrices) {
        SCOPED_TRACE(pair.first);
        const auto& mtx = pair.second;
        std::unique_ptr<matrix_type> factors;
        std::unique_ptr<elimination_forest> forest;
        gko::factorization::symbolic_cholesky(mtx.get(), true, factors, forest);
        const auto dfactors = gko::clone(this->exec, factors);
        gko::array<index_type> dparents{this->exec, mtx->get_size()[0]};

        gko::kernels::GKO_DEVICE_NAMESPACE::elimination_forest::from_factor(
            this->exec, dfactors.get(), dparents.get_data());

        GKO_ASSERT_ARRAY_EQ(forest->parents, dparents);
    }
}


template <typename ValueIndexType>
class Cholesky : public CommonTestFixture {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using factory_type =
        gko::experimental::factorization::Cholesky<value_type, index_type>;
    using matrix_type = typename factory_type::matrix_type;
    using sparsity_pattern_type = typename factory_type::sparsity_pattern_type;
    using elimination_forest =
        gko::factorization::elimination_forest<index_type>;

    Cholesky() : lookup{ref}, dlookup{exec} {}

    void initialize_data(const char* mtx_filename,
                         const char* mtx_chol_filename)
    {
        std::ifstream s_mtx{mtx_filename};
        mtx = gko::read<matrix_type>(s_mtx, ref);
        dmtx = gko::clone(exec, mtx);
        num_rows = mtx->get_size()[0];
        std::ifstream s_mtx_chol{mtx_chol_filename};
        auto mtx_chol_data = gko::read_raw<value_type, index_type>(s_mtx_chol);
        auto nnz = mtx_chol_data.nonzeros.size();
        // add missing upper diagonal entries
        // (values not important, only pattern important)
        gko::utils::make_symmetric(mtx_chol_data);
        mtx_chol_data.sort_row_major();
        mtx_chol = matrix_type::create(ref);
        mtx_chol->read(mtx_chol_data);
        lookup = gko::matrix::csr::build_lookup(mtx_chol.get());

        dlookup = lookup;
        dmtx_chol = gko::clone(exec, mtx_chol);
        mtx_chol_sparsity = sparsity_pattern_type::create(ref);
        mtx_chol_sparsity->copy_from(mtx_chol.get());
        dmtx_chol_sparsity = sparsity_pattern_type::create(exec);
        dmtx_chol_sparsity->copy_from(mtx_chol_sparsity.get());
        gko::factorization::compute_elimination_forest(mtx_chol.get(), forest);
        gko::factorization::compute_elimination_forest(dmtx_chol.get(),
                                                       dforest);
    }

    void forall_matrices(std::function<void()> fn)
    {
        {
            SCOPED_TRACE("ani1");
            this->initialize_data(gko::matrices::location_ani1_mtx,
                                  gko::matrices::location_ani1_chol_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani1_amd");
            this->initialize_data(gko::matrices::location_ani1_amd_mtx,
                                  gko::matrices::location_ani1_amd_chol_mtx);
            fn();
        }
        {
#ifndef GINKGO_FAST_TESTS
            SCOPED_TRACE("ani4");
            this->initialize_data(gko::matrices::location_ani4_mtx,
                                  gko::matrices::location_ani4_chol_mtx);
            fn();
#endif
        }
        {
#ifndef GINKGO_FAST_TESTS
            SCOPED_TRACE("ani4_amd");
            this->initialize_data(gko::matrices::location_ani4_amd_mtx,
                                  gko::matrices::location_ani4_amd_chol_mtx);
            fn();
#endif
        }
    }

    gko::size_type num_rows;
    std::shared_ptr<matrix_type> mtx;
    std::shared_ptr<matrix_type> mtx_chol;
    std::unique_ptr<elimination_forest> forest;
    std::shared_ptr<sparsity_pattern_type> mtx_chol_sparsity;
    std::shared_ptr<matrix_type> dmtx;
    std::shared_ptr<matrix_type> dmtx_chol;
    std::unique_ptr<elimination_forest> dforest;
    std::shared_ptr<sparsity_pattern_type> dmtx_chol_sparsity;
    gko::matrix::csr::lookup_data<index_type> lookup;
    gko::matrix::csr::lookup_data<index_type> dlookup;
};

TYPED_TEST_SUITE(Cholesky, Types, PairTypenameNameGenerator);


TYPED_TEST(Cholesky, KernelInitializeIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        const auto nnz = this->mtx_chol->get_num_stored_elements();
        std::fill_n(this->mtx_chol->get_values(), nnz, gko::zero<value_type>());
        gko::kernels::GKO_DEVICE_NAMESPACE::components::fill_array(
            this->exec, this->dmtx_chol->get_values(), nnz,
            gko::zero<value_type>());
        gko::array<index_type> diag_idxs{this->ref, this->num_rows};
        gko::array<index_type> ddiag_idxs{this->exec, this->num_rows};
        gko::array<index_type> transpose_idxs{this->ref, nnz};
        gko::array<index_type> dtranspose_idxs{this->exec, nnz};

        gko::kernels::reference::cholesky::initialize(
            this->ref, this->mtx.get(),
            this->lookup.storage_offsets.get_const_data(),
            this->lookup.row_descs.get_const_data(),
            this->lookup.storage.get_const_data(), diag_idxs.get_data(),
            transpose_idxs.get_data(), this->mtx_chol.get());
        gko::kernels::GKO_DEVICE_NAMESPACE::cholesky::initialize(
            this->exec, this->dmtx.get(),
            this->dlookup.storage_offsets.get_const_data(),
            this->dlookup.row_descs.get_const_data(),
            this->dlookup.storage.get_const_data(), ddiag_idxs.get_data(),
            dtranspose_idxs.get_data(), this->dmtx_chol.get());

        GKO_ASSERT_MTX_NEAR(this->dmtx_chol, this->dmtx_chol, 0.0);
        GKO_ASSERT_ARRAY_EQ(diag_idxs, ddiag_idxs);
    });
}


TYPED_TEST(Cholesky, KernelFactorizeIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        const auto nnz = this->mtx_chol->get_num_stored_elements();
        gko::array<index_type> diag_idxs{this->ref, this->num_rows};
        gko::array<index_type> ddiag_idxs{this->exec, this->num_rows};
        gko::array<index_type> transpose_idxs{this->ref, nnz};
        gko::array<index_type> dtranspose_idxs{this->exec, nnz};
        gko::array<int> tmp{this->ref};
        gko::array<int> dtmp{this->exec};
        gko::kernels::reference::cholesky::initialize(
            this->ref, this->mtx.get(),
            this->lookup.storage_offsets.get_const_data(),
            this->lookup.row_descs.get_const_data(),
            this->lookup.storage.get_const_data(), diag_idxs.get_data(),
            transpose_idxs.get_data(), this->mtx_chol.get());
        gko::kernels::GKO_DEVICE_NAMESPACE::cholesky::initialize(
            this->exec, this->dmtx.get(),
            this->dlookup.storage_offsets.get_const_data(),
            this->dlookup.row_descs.get_const_data(),
            this->dlookup.storage.get_const_data(), ddiag_idxs.get_data(),
            dtranspose_idxs.get_data(), this->dmtx_chol.get());

        gko::kernels::reference::cholesky::factorize(
            this->ref, this->lookup.storage_offsets.get_const_data(),
            this->lookup.row_descs.get_const_data(),
            this->lookup.storage.get_const_data(), diag_idxs.get_const_data(),
            transpose_idxs.get_const_data(), this->mtx_chol.get(), true, tmp);
        gko::kernels::GKO_DEVICE_NAMESPACE::cholesky::factorize(
            this->exec, this->dlookup.storage_offsets.get_const_data(),
            this->dlookup.row_descs.get_const_data(),
            this->dlookup.storage.get_const_data(), ddiag_idxs.get_const_data(),
            dtranspose_idxs.get_const_data(), this->dmtx_chol.get(), true,
            dtmp);

        GKO_ASSERT_MTX_NEAR(this->mtx_chol, this->dmtx_chol,
                            r<value_type>::value);
    });
}


TYPED_TEST(Cholesky, GenerateWithUnknownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .on(this->exec);

        auto factors = factory->generate(this->mtx);
        auto dfactors = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(factors->get_combined(),
                                   dfactors->get_combined());
        GKO_ASSERT_MTX_NEAR(factors->get_combined(), dfactors->get_combined(),
                            r<value_type>::value);
    });
}


TYPED_TEST(Cholesky, GenerateWithKnownSparsityIsEquivalentToRef)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices([this] {
        auto factory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .with_symbolic_factorization(this->mtx_chol_sparsity)
                .on(this->ref);
        auto dfactory =
            gko::experimental::factorization::Cholesky<value_type,
                                                       index_type>::build()
                .with_symbolic_factorization(this->dmtx_chol_sparsity)
                .on(this->exec);

        auto factors = factory->generate(this->mtx);
        auto dfactors = dfactory->generate(this->dmtx);

        GKO_ASSERT_MTX_EQ_SPARSITY(this->dmtx_chol_sparsity,
                                   dfactors->get_combined());
        GKO_ASSERT_MTX_NEAR(factors->get_combined(), dfactors->get_combined(),
                            r<value_type>::value);
    });
}


}  // namespace
