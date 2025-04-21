// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/cholesky_kernels.hpp"

#include <algorithm>
#include <initializer_list>
#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/factorization/cholesky.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/identity.hpp>

#include "core/base/index_range.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/elimination_forest_kernels.hpp"
#include "core/factorization/symbolic.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/assertions.hpp"
#include "ginkgo/core/base/types.hpp"
#include "matrices/config.hpp"


template <typename ValueIndexType>
class Cholesky : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using matrix_type = gko::matrix::Csr<value_type, index_type>;
    using sparsity_matrix_type =
        gko::matrix::SparsityCsr<value_type, index_type>;
    using elimination_forest =
        gko::factorization::elimination_forest<index_type>;

    Cholesky()
        : ref(gko::ReferenceExecutor::create()),
          tmp{ref},
          ref_row_nnz{ref},
          lookup{ref}
    {}

    std::unique_ptr<matrix_type> combined_factor(
        gko::ptr_param<const matrix_type> l_factor)
    {
        auto one = gko::initialize<gko::matrix::Dense<value_type>>(
            {gko::one<value_type>()}, ref);
        auto id = gko::matrix::Identity<value_type>::create(
            ref, l_factor->get_size()[0]);
        auto result = gko::as<matrix_type>(l_factor->conj_transpose());
        l_factor->apply(one, id, one, result);
        gko::matrix_data<value_type, index_type> data;
        result->write(data);
        for (auto& entry : data.nonzeros) {
            if (entry.row == entry.column) {
                entry.value /= value_type{2.0};
            }
        }
        result->read(data);
        return result;
    }

    void setup(
        std::initializer_list<std::initializer_list<value_type>> mtx_list,
        std::initializer_list<std::initializer_list<value_type>> factor_list)
    {
        mtx = gko::initialize<matrix_type>(mtx_list, ref);
        l_factor_ref = gko::initialize<matrix_type>(factor_list, ref);
        setup_impl();
    }

    void setup(const char* name_mtx, const char* name_factor)
    {
        std::ifstream stream{name_mtx};
        std::ifstream ref_stream{name_factor};
        mtx = gko::read<matrix_type>(stream, this->ref);
        l_factor_ref = gko::read<matrix_type>(ref_stream, this->ref);
        setup_impl();
    }

    void setup_impl()
    {
        num_rows = mtx->get_size()[0];
        combined_ref = combined_factor(l_factor_ref.get());
        l_factor = matrix_type::create(ref, l_factor_ref->get_size(),
                                       l_factor_ref->get_num_stored_elements());
        combined = matrix_type::create(ref, combined_ref->get_size(),
                                       combined_ref->get_num_stored_elements());
        gko::factorization::compute_elimination_forest(l_factor_ref.get(),
                                                       forest);
        // init sparsity lookup
        ref->copy(num_rows + 1, l_factor_ref->get_const_row_ptrs(),
                  l_factor->get_row_ptrs());
        ref->copy(l_factor_ref->get_num_stored_elements(),
                  l_factor_ref->get_const_col_idxs(), l_factor->get_col_idxs());
        ref->copy(num_rows + 1, combined_ref->get_const_row_ptrs(),
                  combined->get_row_ptrs());
        ref->copy(combined_ref->get_num_stored_elements(),
                  combined_ref->get_const_col_idxs(), combined->get_col_idxs());

        ref_row_nnz.resize_and_reset(num_rows);
        const auto ref_row_ptrs = l_factor_ref->get_const_row_ptrs();
        for (gko::size_type row = 0; row < num_rows; row++) {
            ref_row_nnz.get_data()[row] =
                ref_row_ptrs[row + 1] - ref_row_ptrs[row];
        }

        lookup = gko::matrix::csr::build_lookup(combined.get());
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

    void forall_matrices(std::function<void()> fn, bool non_spd)
    {
        {
            SCOPED_TRACE("ani1");
            this->setup(gko::matrices::location_ani1_mtx,
                        gko::matrices::location_ani1_chol_mtx);
            fn();
        }
        {
            SCOPED_TRACE("ani1_amd");
            this->setup(gko::matrices::location_ani1_amd_mtx,
                        gko::matrices::location_ani1_amd_chol_mtx);
            fn();
        }
        {
            // structurally this is the example from Liu 1990
            // "The Role of Elimination Trees in Sparse Factorization"
            // https://doi.org/10.1137/0611010.
            SCOPED_TRACE("example");
            this->setup(
                {{4, 0, 1, 0, 0, 0, 0, 1, 0, 0},
                 {0, 4, 0, 0, 1, 0, 0, 0, 0, 1},
                 {1, 0, 4.25, 0, 0, 0, 1, 0, 0, 0},
                 {0, 0, 0, 4, 0, 0, 0, 0, 1, 1},
                 {0, 1, 0, 0, 4.25, 0, 0, 0, 1, 1},
                 {0, 0, 0, 0, 0, 4, 2, 4, 0, 0},
                 {0, 0, 1, 0, 0, 2, 5.25, 0, 0, 0},
                 {1, 0, 0, 0, 0, 4, 0, 8, 1, 1},
                 {0, 0, 0, 1, 1, 0, 0, 1, 4, 0},
                 {0, 1, 0, 1, 1, 0, 0, 1, 0, 4}},
                {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                 {0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                 {0.5, 0, 2, 0, 0, 0, 0, 0, 0, 0},
                 {0, 0, 0, 2, 0, 0, 0, 0, 0, 0},
                 {0, 0.5, 0, 0, 2, 0, 0, 0, 0, 0},
                 {0, 0, 0, 0, 0, 2, 0, 0, 0, 0},
                 {0, 0, 0.5, 0, 0, 1, 2, 0, 0, 0},
                 {0.5, 0, -0.125, 0, 0, 2, -0.96875, 1.67209402770897, 0, 0},
                 {0, 0, 0, 0.5, 0.5, 0, 0, 0.598052491922453, 1.7726627476498,
                  0},
                 {0, 0.5, 0, 0.5, 0.375, 0, 0, 0.598052491922453,
                  -0.448571948696326, 1.67346688755653}});
            fn();
        }
        {
            SCOPED_TRACE("separable");
            this->setup({{4, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                         {0, 4, 2, 0, 0, 0, 0, 0, 0, 0},
                         {1, 2, 5.25, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 4, 1, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 4.25, 1, 0, 0, 0, 4},
                         {0, 0, 0, 0, 1, 4.25, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 4, 1, 0, 4},
                         {0, 0, 0, 0, 0, 0, 1, 4.25, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 4, 1},
                         {0, 0, 0, 0, 4, 0, 4, 0, 1, 17.75}},
                        {{2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0, 2, 0, 0, 0, 0, 0, 0, 0, 0},
                         {0.5, 1, 2, 0, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 2, 0, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0.5, 2, 0, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0.5, 2, 0, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 2, 0, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0.5, 2, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 2, 0},
                         {0, 0, 0, 0, 2, -0.5, 2, -0.5, 0.5, 3}});
            fn();
        }
        if (non_spd) {
            SCOPED_TRACE("missing diagonal");
            this->setup({{1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                         {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
                         {1, 1, 0, 1, 0, 0, 0, 0, 0, 0},
                         {0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
                         {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
                         {0, 0, 0, 0, 1, 1, 1, 0, 0, 0},
                         {0, 0, 0, 0, 0, 1, 1, 1, 0, 1},
                         {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1},
                         {0, 0, 0, 0, 0, 0, 1, 0, 1, 0}},
                        {{1., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
                         {0., 1., 0., 0., 0., 0., 0., 0., 0., 0.},
                         {1., 1., 1., 0., 0., 0., 0., 0., 0., 0.},
                         {0., 0., 1., 1., 0., 0., 0., 0., 0., 0.},
                         {0., 0., 0., 1., 1., 0., 0., 0., 0., 0.},
                         {0., 0., 0., 0., 1., 1., 0., 0., 0., 0.},
                         {0., 0., 0., 0., 0., 1., 1., 0., 0., 0.},
                         {0., 0., 0., 0., 0., 0., 1., 1., 0., 0.},
                         {0., 0., 0., 0., 0., 0., 0., 0., 1., 0.},
                         {0., 0., 0., 0., 0., 0., 1., 1., 1., 1.}});
            fn();
        }
    }

    std::shared_ptr<const gko::ReferenceExecutor> ref;
    gko::size_type num_rows;
    gko::array<index_type> tmp;
    gko::array<index_type> ref_row_nnz;
    gko::matrix::csr::lookup_data<index_type> lookup;
    std::shared_ptr<matrix_type> mtx;
    std::unique_ptr<elimination_forest> forest;
    std::shared_ptr<matrix_type> l_factor;
    std::shared_ptr<matrix_type> l_factor_ref;
    std::shared_ptr<matrix_type> combined;
    std::shared_ptr<matrix_type> combined_ref;
};

TYPED_TEST_SUITE(Cholesky, gko::test::ValueIndexTypes,
                 PairTypenameNameGenerator);


template <typename IndexType>
void ref_compute_children(const IndexType* parents, IndexType size,
                          IndexType* child_ptrs, IndexType* children)
{
    std::vector<std::vector<IndexType>> child_vectors(size + 1);
    for (const auto i : gko::irange{size}) {
        const auto parent = parents[i];
        child_vectors[parent].push_back(i);
    }
    child_ptrs[0] = 0;
    for (const auto i : gko::irange{size + 1}) {
        child_ptrs[i + 1] =
            child_ptrs[i] + static_cast<IndexType>(child_vectors[i].size());
        std::copy(child_vectors[i].begin(), child_vectors[i].end(),
                  children + child_ptrs[i]);
    }
}


template <typename IndexType>
void ref_compute_euler_walk(const IndexType* child_ptrs,
                            const IndexType* children, IndexType node,
                            IndexType level, std::vector<IndexType>& path,
                            std::vector<IndexType>& levels, IndexType* first)
{
    if (level >= 0) {
        first[node] = static_cast<IndexType>(path.size());
        path.emplace_back(node);
        levels.emplace_back(level);
    }
    const auto child_begin = child_ptrs[node];
    const auto child_end = child_ptrs[node + 1];
    for (const auto child_idx : gko::irange{child_begin, child_end}) {
        const auto child = children[child_idx];
        ref_compute_euler_walk(child_ptrs, children, child, level + 1, path,
                               levels, first);
        if (level >= 0) {
            path.emplace_back(node);
            levels.emplace_back(level);
        }
    }
}


template <typename IndexType>
void ref_compute_levels(const IndexType* parents, IndexType size,
                        IndexType* levels)
{
    for (auto node = size - 1; node >= 0; node--) {
        const auto parent = parents[node];
        // root nodes are attached to pseudo-root at index size
        levels[node] = parent == size ? IndexType{} : levels[parent] + 1;
    }
}


TYPED_TEST(Cholesky, KernelComputeEliminationForest)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            const auto size = this->mtx->get_size()[0];
            const auto ssize = static_cast<index_type>(size);
            this->forest =
                std::make_unique<elimination_forest>(this->ref, ssize);
            this->forest->parents.fill(-1);
            this->forest->child_ptrs.fill(-1);
            this->forest->children.fill(-1);
            this->forest->levels.fill(-1);
            this->forest->euler_walk.fill(-1);
            this->forest->euler_levels.fill(-1);
            this->forest->euler_first.fill(-1);
            this->forest->postorder.fill(-1);
            this->forest->inv_postorder.fill(-1);
            this->forest->postorder_parents.fill(-1);
            const auto parents = this->forest->parents.get_const_data();

            gko::kernels::reference::elimination_forest::compute(
                this->ref, this->mtx->get_const_row_ptrs(),
                this->mtx->get_const_col_idxs(), size, *this->forest);


            // child pointers and indices
            gko::array<index_type> ref_child_ptrs{this->ref, size + 2};
            gko::array<index_type> ref_children{this->ref, size};
            ref_compute_children(parents, ssize, ref_child_ptrs.get_data(),
                                 ref_children.get_data());
            GKO_ASSERT_ARRAY_EQ(this->forest->child_ptrs, ref_child_ptrs);
            GKO_ASSERT_ARRAY_EQ(this->forest->children, ref_children);
            // levels
            gko::array<index_type> ref_levels{this->ref, size};
            ref_levels.fill(-1);
            ref_compute_levels(parents, ssize, ref_levels.get_data());
            GKO_ASSERT_ARRAY_EQ(this->forest->levels, ref_levels);
            // check euler walk for correctness
            std::vector<index_type> ref_euler_walk;
            std::vector<index_type> ref_euler_levels;
            gko::array<index_type> ref_euler_first{this->ref, size};
            ref_euler_first.fill(-1);
            ref_compute_euler_walk(
                ref_child_ptrs.get_const_data(), ref_children.get_const_data(),
                ssize, index_type{-1}, ref_euler_walk, ref_euler_levels,
                ref_euler_first.get_data());
            ref_euler_walk.resize(2 * size - 1, -1);
            ref_euler_levels.resize(2 * size - 1, -1);
            const gko::array<index_type> ref_euler_walk_array{
                this->ref, ref_euler_walk.begin(), ref_euler_walk.end()};
            const gko::array<index_type> ref_euler_levels_array{
                this->ref, ref_euler_levels.begin(), ref_euler_levels.end()};
            GKO_ASSERT_ARRAY_EQ(this->forest->euler_walk, ref_euler_walk_array);
            GKO_ASSERT_ARRAY_EQ(this->forest->euler_levels,
                                ref_euler_levels_array);
            GKO_ASSERT_ARRAY_EQ(this->forest->euler_first, ref_euler_first);
        },
        true);
}


TYPED_TEST(Cholesky, KernelComputeSkeletonTreeIsEquivalentToOriginalMatrix)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            auto skeleton = matrix_type::create(
                this->ref, this->mtx->get_size(), this->mtx->get_size()[0]);
            std::unique_ptr<elimination_forest> skeleton_forest;
            gko::factorization::compute_elimination_forest(this->mtx.get(),
                                                           this->forest);
            gko::kernels::reference::elimination_forest::compute_skeleton_tree(
                this->ref, this->mtx->get_const_row_ptrs(),
                this->mtx->get_const_col_idxs(), this->mtx->get_size()[0],
                skeleton->get_row_ptrs(), skeleton->get_col_idxs());

            gko::factorization::compute_elimination_forest(skeleton.get(),
                                                           skeleton_forest);

            this->assert_equal_forests(*skeleton_forest, *this->forest);
        },
        true);
}


TYPED_TEST(Cholesky, KernelSymbolicCount)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            gko::factorization::compute_elimination_forest(this->mtx.get(),
                                                           this->forest);
            gko::array<index_type> row_nnz{this->ref, this->num_rows};

            gko::kernels::reference::cholesky::symbolic_count(
                this->ref, this->mtx.get(), *this->forest, row_nnz.get_data(),
                this->tmp);

            GKO_ASSERT_ARRAY_EQ(row_nnz, this->ref_row_nnz);
        },
        true);
}


TYPED_TEST(Cholesky, KernelSymbolicFactorize)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            gko::factorization::compute_elimination_forest(this->mtx.get(),
                                                           this->forest);
            gko::kernels::reference::cholesky::symbolic_count(
                this->ref, this->mtx.get(), *this->forest,
                this->l_factor->get_row_ptrs(), this->tmp);
            gko::kernels::reference::components::prefix_sum_nonnegative(
                this->ref, this->l_factor->get_row_ptrs(), this->num_rows + 1);

            gko::kernels::reference::cholesky::symbolic_factorize(
                this->ref, this->mtx.get(), *this->forest, this->l_factor.get(),
                this->tmp);

            GKO_ASSERT_MTX_EQ_SPARSITY(this->l_factor, this->l_factor_ref);
        },
        true);
}


TYPED_TEST(Cholesky, SymbolicFactorize)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    this->forall_matrices(
        [this] {
            std::unique_ptr<matrix_type> combined_factor;
            std::unique_ptr<elimination_forest> forest;
            gko::factorization::symbolic_cholesky(this->mtx.get(), true,
                                                  combined_factor, forest);

            GKO_ASSERT_MTX_EQ_SPARSITY(combined_factor, this->combined_ref);
        },
        true);
}


TYPED_TEST(Cholesky, SymbolicFactorizeOnlyLower)
{
    using matrix_type = typename TestFixture::matrix_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    this->forall_matrices(
        [this] {
            std::unique_ptr<matrix_type> l_factor;
            std::unique_ptr<elimination_forest> forest;
            gko::factorization::symbolic_cholesky(this->mtx.get(), false,
                                                  l_factor, forest);

            GKO_ASSERT_MTX_EQ_SPARSITY(l_factor, this->l_factor_ref);
        },
        true);
}


TYPED_TEST(Cholesky, KernelForestFromFactorPlusPostprocessing)
{
    using matrix_type = typename TestFixture::matrix_type;
    using index_type = typename TestFixture::index_type;
    using elimination_forest = typename TestFixture::elimination_forest;
    this->forall_matrices(
        [this] {
            std::unique_ptr<matrix_type> combined_factor;
            std::unique_ptr<elimination_forest> forest_ref;
            gko::factorization::symbolic_cholesky(this->mtx.get(), true,
                                                  combined_factor, forest_ref);
            gko::array<index_type> parents{this->ref, this->num_rows};

            gko::kernels::reference::elimination_forest::from_factor(
                this->ref, combined_factor.get(), parents.get_data());

            GKO_ASSERT_ARRAY_EQ(forest_ref->parents, parents);
        },
        true);
}


TYPED_TEST(Cholesky, KernelInitializeWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            std::fill_n(this->combined->get_values(),
                        this->combined->get_num_stored_elements(),
                        gko::zero<value_type>());
            gko::array<index_type> diag_idxs{this->ref, this->num_rows};
            gko::array<index_type> transpose_idxs{
                this->ref, this->combined->get_num_stored_elements()};

            gko::kernels::reference::cholesky::initialize(
                this->ref, this->mtx.get(),
                this->lookup.storage_offsets.get_const_data(),
                this->lookup.row_descs.get_const_data(),
                this->lookup.storage.get_const_data(), diag_idxs.get_data(),
                transpose_idxs.get_data(), this->combined.get());

            GKO_ASSERT_MTX_NEAR(this->mtx, this->combined, 0.0);
            for (gko::size_type row = 0; row < this->num_rows; row++) {
                const auto diag_pos = diag_idxs.get_const_data()[row];
                const auto begin_pos =
                    this->combined->get_const_row_ptrs()[row];
                const auto end_pos =
                    this->combined->get_const_row_ptrs()[row + 1];
                ASSERT_GE(diag_pos, begin_pos);
                ASSERT_LT(diag_pos, end_pos);
                ASSERT_EQ(this->combined->get_const_col_idxs()[diag_pos], row);
                for (auto nz = begin_pos; nz < end_pos; nz++) {
                    const auto trans_pos = transpose_idxs.get_const_data()[nz];
                    const auto col = this->combined->get_const_col_idxs()[nz];
                    ASSERT_GE(trans_pos,
                              this->combined->get_const_row_ptrs()[col]);
                    ASSERT_LT(trans_pos,
                              this->combined->get_const_row_ptrs()[col + 1]);
                    ASSERT_EQ(this->combined->get_const_col_idxs()[trans_pos],
                              row);
                    ASSERT_EQ(transpose_idxs.get_const_data()[trans_pos], nz);
                }
            }
        },
        true);
}


TYPED_TEST(Cholesky, KernelFactorizeWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            gko::array<index_type> diag_idxs{this->ref, this->num_rows};
            gko::array<index_type> transpose_idxs{
                this->ref, this->combined->get_num_stored_elements()};
            gko::array<int> tmp{this->ref};
            gko::kernels::reference::cholesky::initialize(
                this->ref, this->mtx.get(),
                this->lookup.storage_offsets.get_const_data(),
                this->lookup.row_descs.get_const_data(),
                this->lookup.storage.get_const_data(), diag_idxs.get_data(),
                transpose_idxs.get_data(), this->combined.get());

            gko::kernels::reference::cholesky::factorize(
                this->ref, this->lookup.storage_offsets.get_const_data(),
                this->lookup.row_descs.get_const_data(),
                this->lookup.storage.get_const_data(), diag_idxs.get_data(),
                transpose_idxs.get_data(), this->combined.get(), true, tmp);

            GKO_ASSERT_MTX_NEAR(this->combined, this->combined_ref,
                                r<value_type>::value);
        },
        false);
}


TYPED_TEST(Cholesky, FactorizeWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            auto factory =
                gko::experimental::factorization::Cholesky<value_type,
                                                           index_type>::build()
                    .on(this->ref);

            auto cholesky = factory->generate(this->mtx);

            GKO_ASSERT_MTX_NEAR(cholesky->get_combined(), this->combined_ref,
                                r<value_type>::value);
            ASSERT_EQ(cholesky->get_storage_type(),
                      gko::experimental::factorization::storage_type::
                          symm_combined_cholesky);
            ASSERT_EQ(cholesky->get_lower_factor(), nullptr);
            ASSERT_EQ(cholesky->get_upper_factor(), nullptr);
            ASSERT_EQ(cholesky->get_diagonal(), nullptr);
        },
        false);
}


TYPED_TEST(Cholesky, FactorizeWithKnownSparsityWorks)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    this->forall_matrices(
        [this] {
            auto pattern = gko::share(
                gko::matrix::SparsityCsr<value_type, index_type>::create(
                    this->ref));
            pattern->copy_from(this->combined_ref.get());
            auto factory =
                gko::experimental::factorization::Cholesky<value_type,
                                                           index_type>::build()
                    .with_symbolic_factorization(pattern)
                    .on(this->ref);

            auto cholesky = factory->generate(this->mtx);

            GKO_ASSERT_MTX_NEAR(cholesky->get_combined(), this->combined_ref,
                                r<value_type>::value);
            ASSERT_EQ(cholesky->get_storage_type(),
                      gko::experimental::factorization::storage_type::
                          symm_combined_cholesky);
            ASSERT_EQ(cholesky->get_lower_factor(), nullptr);
            ASSERT_EQ(cholesky->get_upper_factor(), nullptr);
            ASSERT_EQ(cholesky->get_diagonal(), nullptr);
        },
        false);
}
